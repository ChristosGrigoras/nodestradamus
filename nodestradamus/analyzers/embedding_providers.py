"""Embedding provider abstraction for pluggable embedding backends.

Supports local (sentence-transformers) and API-based (Mistral) providers.

Configuration via environment variables:
    NODESTRADAMUS_EMBEDDING_PROVIDER: Provider to use ("local" or "mistral")
    NODESTRADAMUS_EMBEDDING_MODEL: Model name override (provider-specific)
    MISTRAL_API_KEY: API key for Mistral provider
"""

import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np

from nodestradamus.logging import ProgressBar, logger


@dataclass
class EmbeddingResult:
    """Result of encoding texts, with tracking of successes and failures.

    Attributes:
        embeddings: numpy array of shape (n_success, dimensions).
        success_indices: Indices of input texts that were successfully embedded.
        skipped_indices: Indices of input texts that failed to embed.
        skipped_reasons: Reason for each skipped text (parallel to skipped_indices).
    """

    embeddings: np.ndarray
    success_indices: list[int] = field(default_factory=list)
    skipped_indices: list[int] = field(default_factory=list)
    skipped_reasons: list[str] = field(default_factory=list)

# Default models
_DEFAULT_LOCAL_MODEL = "jinaai/jina-embeddings-v2-base-code"
_MISTRAL_MODEL = "codestral-embed"

# Cached provider instance
_provider: "EmbeddingProvider | None" = None


def _get_provider_name() -> str:
    """Get provider name from environment (read at call time, not import time)."""
    return os.getenv("NODESTRADAMUS_EMBEDDING_PROVIDER", "local").lower()


def get_expected_model_name() -> str:
    """Get expected model name from environment without instantiating provider.

    Useful for cache validation to avoid loading models or checking API keys
    when just comparing model names.

    Returns:
        Model identifier string matching what the provider's model_name property returns.
    """
    provider_name = _get_provider_name()
    if provider_name == "mistral":
        return f"mistral:{_MISTRAL_MODEL}"
    else:
        model = os.getenv("NODESTRADAMUS_EMBEDDING_MODEL", _DEFAULT_LOCAL_MODEL)
        return f"local:{model}"


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier for cache validation."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        ...

    @abstractmethod
    def encode(self, texts: list[str]) -> EmbeddingResult:
        """Encode texts into embeddings with graceful failure handling.

        Args:
            texts: List of text strings to encode.

        Returns:
            EmbeddingResult with embeddings for successful texts and
            tracking of any skipped texts.
        """
        ...


class LocalProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers.

    Uses jinaai/jina-embeddings-v2-base-code by default, configurable
    via NODESTRADAMUS_EMBEDDING_MODEL environment variable.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._model_name = os.getenv("NODESTRADAMUS_EMBEDDING_MODEL", _DEFAULT_LOCAL_MODEL)
        self._dimensions: int | None = None

    @property
    def model_name(self) -> str:
        return f"local:{self._model_name}"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Load model to get dimensions
            model = self._get_model()
            self._dimensions = model.get_sentence_embedding_dimension()
        return self._dimensions

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                ) from e
            logger.info("  Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("  Model loaded successfully")
        return self._model

    def encode(self, texts: list[str]) -> EmbeddingResult:
        """Encode texts using sentence-transformers.

        Local provider handles all content gracefully, so all texts succeed.
        """
        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                success_indices=[],
                skipped_indices=[],
                skipped_reasons=[],
            )

        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return EmbeddingResult(
            embeddings=embeddings,
            success_indices=list(range(len(texts))),
            skipped_indices=[],
            skipped_reasons=[],
        )


class MistralProvider(EmbeddingProvider):
    """Mistral Codestral Embed API provider.

    Requires MISTRAL_API_KEY environment variable.
    Uses codestral-embed model optimized for code.

    Parallel API calls can be configured via NODESTRADAMUS_EMBEDDING_WORKERS env var.
    Default is 4 workers for ~3.5x speedup over sequential processing.
    """

    API_URL = "https://api.mistral.ai/v1/embeddings"
    MODEL = _MISTRAL_MODEL
    DIMENSIONS = 1536  # Codestral Embed actual output
    MAX_BATCH_SIZE = 64  # API limit per request
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    DEFAULT_WORKERS = 4  # Parallel API calls (benchmarked: 3.5x speedup)

    def __init__(self) -> None:
        self._api_key = os.getenv("MISTRAL_API_KEY")
        if not self._api_key:
            raise ValueError(
                "MISTRAL_API_KEY environment variable is required for Mistral provider. "
                "Get your API key at: https://console.mistral.ai/api-keys"
            )
        self._client: Any = None
        self._client_lock = Lock()
        self._max_workers = int(os.getenv("NODESTRADAMUS_EMBEDDING_WORKERS", self.DEFAULT_WORKERS))

    @property
    def model_name(self) -> str:
        return f"mistral:{self.MODEL}"

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS

    def _get_client(self) -> Any:
        """Lazy-load httpx client (thread-safe)."""
        if self._client is None:
            with self._client_lock:
                # Double-check after acquiring lock
                if self._client is None:
                    try:
                        import httpx
                    except ImportError as e:
                        raise ImportError(
                            "httpx is required for Mistral API. "
                            "Install with: pip install httpx"
                        ) from e
                    self._client = httpx.Client(
                        timeout=60.0,
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                    )
        return self._client

    def _call_api(self, texts: list[str]) -> list[list[float]] | None:
        """Call Mistral embeddings API with retry logic.

        Returns:
            List of embeddings on success, None on failure.
        """
        client = self._get_client()

        for attempt in range(self.MAX_RETRIES):
            try:
                response = client.post(
                    self.API_URL,
                    json={
                        "model": self.MODEL,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        "  Mistral API error (attempt %d/%d): %s",
                        attempt + 1,
                        self.MAX_RETRIES,
                        str(e),
                    )
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    # Return None to signal failure (caller will handle)
                    return None
        return None

    def _embed_batch_with_fallback(
        self,
        texts: list[str],
        original_indices: list[int],
    ) -> tuple[list[list[float]], list[int], list[int], list[str]]:
        """Embed a batch, using binary search to isolate bad texts on failure.

        Args:
            texts: Texts to embed.
            original_indices: Original indices of these texts in the full input.

        Returns:
            Tuple of (embeddings, success_indices, skipped_indices, skipped_reasons).
        """
        # Try the whole batch first
        result = self._call_api(texts)
        if result is not None:
            return result, original_indices, [], []

        # Batch failed - if single text, it's the bad one
        if len(texts) == 1:
            reason = "API rejected content (400 Bad Request)"
            logger.warning("  Skipping chunk at index %d: %s", original_indices[0], reason)
            return [], [], original_indices, [reason]

        # Binary search to isolate bad texts
        mid = len(texts) // 2
        left_texts = texts[:mid]
        left_indices = original_indices[:mid]
        right_texts = texts[mid:]
        right_indices = original_indices[mid:]

        # Recursively process both halves
        left_emb, left_ok, left_skip, left_reasons = self._embed_batch_with_fallback(
            left_texts, left_indices
        )
        right_emb, right_ok, right_skip, right_reasons = self._embed_batch_with_fallback(
            right_texts, right_indices
        )

        return (
            left_emb + right_emb,
            left_ok + right_ok,
            left_skip + right_skip,
            left_reasons + right_reasons,
        )

    def encode(self, texts: list[str]) -> EmbeddingResult:
        """Encode texts using Mistral Codestral Embed API.

        Uses parallel API calls for ~3.5x speedup (configurable via
        NODESTRADAMUS_EMBEDDING_WORKERS env var, default 4).

        Uses binary search to isolate and skip problematic texts that
        cause API errors, rather than failing the entire batch.
        """
        if not texts:
            return EmbeddingResult(
                embeddings=np.array([]),
                success_indices=[],
                skipped_indices=[],
                skipped_reasons=[],
            )

        # Prepare batches
        batches: list[tuple[int, list[str], list[int]]] = []
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i : i + self.MAX_BATCH_SIZE]
            batch_indices = list(range(i, i + len(batch)))
            batches.append((i, batch, batch_indices))

        # Results indexed by batch position for ordered reassembly
        results: dict[int, tuple[list[list[float]], list[int], list[int], list[str]]] = {}

        # Process batches in parallel
        with ProgressBar(total=len(texts), desc="Embedding chunks", unit="chunks") as pbar:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                # Submit all batch jobs
                future_to_batch = {
                    executor.submit(
                        self._embed_batch_with_fallback, batch, batch_indices
                    ): (batch_idx, len(batch))
                    for batch_idx, batch, batch_indices in batches
                }

                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    batch_idx, batch_len = future_to_batch[future]
                    try:
                        result = future.result()
                        results[batch_idx] = result
                    except Exception as e:
                        # Batch failed completely - mark all as skipped
                        logger.warning("  Batch %d failed: %s", batch_idx, str(e))
                        batch_indices = batches[batch_idx // self.MAX_BATCH_SIZE][2]
                        results[batch_idx] = (
                            [],
                            [],
                            batch_indices,
                            [f"Batch failed: {e}"] * len(batch_indices),
                        )
                    pbar.update(batch_len)

        # Reassemble results in original order
        all_embeddings: list[list[float]] = []
        all_success: list[int] = []
        all_skipped: list[int] = []
        all_reasons: list[str] = []

        for batch_idx, _, _ in batches:
            emb, ok, skip, reasons = results[batch_idx]
            all_embeddings.extend(emb)
            all_success.extend(ok)
            all_skipped.extend(skip)
            all_reasons.extend(reasons)

        if all_skipped:
            logger.warning(
                "  Skipped %d chunks due to API errors (out of %d total)",
                len(all_skipped),
                len(texts),
            )

        return EmbeddingResult(
            embeddings=np.array(all_embeddings, dtype=np.float32) if all_embeddings else np.array([]),
            success_indices=all_success,
            skipped_indices=all_skipped,
            skipped_reasons=all_reasons,
        )


def get_embedding_provider() -> EmbeddingProvider:
    """Get the configured embedding provider.

    Returns:
        EmbeddingProvider instance based on NODESTRADAMUS_EMBEDDING_PROVIDER env var.

    Raises:
        ValueError: If provider name is unknown or misconfigured.
    """
    global _provider

    if _provider is not None:
        return _provider

    provider_name = _get_provider_name()

    if provider_name == "local":
        logger.info("  Using local embedding provider (sentence-transformers)")
        _provider = LocalProvider()
    elif provider_name == "mistral":
        logger.info("  Using Mistral embedding provider (codestral-embed)")
        _provider = MistralProvider()
    else:
        raise ValueError(
            f"Unknown embedding provider: {provider_name}. "
            f"Supported providers: local, mistral"
        )

    return _provider


def reset_provider() -> None:
    """Reset the cached provider instance.

    Useful for testing or when changing configuration at runtime.
    """
    global _provider
    _provider = None
