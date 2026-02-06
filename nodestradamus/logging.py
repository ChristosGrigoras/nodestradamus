"""Logging configuration for Nodestradamus MCP server.

Logs to stderr for visibility in Cursor's MCP logs (stdout is used for protocol).
Provides tqdm progress bars when available for visual feedback.
"""

import logging
import os
import sys
import time
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any, TypeVar

# tqdm for progress bars (optional but recommended)
# Disable with NODESTRADAMUS_DISABLE_PROGRESS=1 for MCP mode or non-TTY environments
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _tqdm = None  # type: ignore[assignment, misc]
    _HAS_TQDM = False

# Check if progress bars should be disabled
# - NODESTRADAMUS_DISABLE_PROGRESS=1 explicitly disables
# - Non-TTY stderr also disables (common in MCP mode)
_DISABLE_PROGRESS = (
    os.getenv("NODESTRADAMUS_DISABLE_PROGRESS", "").lower() in ("1", "true", "yes")
    or not sys.stderr.isatty()
)

T = TypeVar("T")

# Create logger that outputs to stderr
logger = logging.getLogger("nodestradamus")
logger.setLevel(logging.INFO)

# Only add handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[Nodestradamus] %(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_progress(
    message: str,
    current: int | None = None,
    total: int | None = None,
    level: int = logging.INFO,
) -> None:
    """Log a progress message.

    Args:
        message: The progress message.
        current: Current item number (optional).
        total: Total items (optional).
        level: Log level (default INFO).
    """
    if current is not None and total is not None:
        pct = (current / total * 100) if total > 0 else 0
        logger.log(level, "%s [%d/%d %.1f%%]", message, current, total, pct)
    elif current is not None:
        logger.log(level, "%s [%d]", message, current)
    else:
        logger.log(level, message)


class TimingContext:
    """Context object that captures elapsed time from an operation.

    Attributes:
        elapsed: Elapsed time in seconds (set after context exits).
        elapsed_ms: Elapsed time in milliseconds (set after context exits).
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def start(self) -> None:
        """Start the timer."""
        self._start = time.perf_counter()

    def stop(self) -> None:
        """Stop the timer and record elapsed time."""
        self.elapsed = time.perf_counter() - self._start
        self.elapsed_ms = self.elapsed * 1000


@contextmanager
def log_operation(
    operation: str,
    details: dict[str, Any] | None = None,
) -> Generator[TimingContext, None, None]:
    """Context manager for logging operation start/end with timing.

    Args:
        operation: Name of the operation.
        details: Optional details dict to include in start message.

    Yields:
        TimingContext object with elapsed time after context exits.

    Example:
        with log_operation("analyze_deps", {"repo": "/path/to/repo"}) as timing:
            # do the analysis
        print(f"Took {timing.elapsed_ms:.1f}ms")
    """
    details_str = ""
    if details:
        details_str = " " + " ".join(f"{k}={v}" for k, v in details.items())

    logger.info("▶ Starting %s%s", operation, details_str)

    ctx = TimingContext()
    ctx.start()

    try:
        yield ctx
    except Exception as e:
        ctx.stop()
        logger.error("✗ %s failed after %.2fs: %s", operation, ctx.elapsed, e)
        raise
    else:
        ctx.stop()
        logger.info("✓ Completed %s in %.2fs", operation, ctx.elapsed)


class ProgressTracker:
    """Track and log progress for long-running operations.

    Logs progress at configurable intervals to avoid log spam.
    """

    def __init__(
        self,
        operation: str,
        total: int,
        log_every: int = 100,
        log_every_pct: float = 10.0,
    ):
        """Initialize progress tracker.

        Args:
            operation: Name of the operation being tracked.
            total: Total number of items to process.
            log_every: Log every N items (default: 100).
            log_every_pct: Also log at every N% milestone (default: 10%).
        """
        self.operation = operation
        self.total = total
        self.log_every = log_every
        self.log_every_pct = log_every_pct
        self.current = 0
        self.start_time = time.perf_counter()
        self.last_pct_logged = 0

    def update(self, increment: int = 1) -> None:
        """Update progress counter and log if needed.

        Args:
            increment: Number of items completed (default: 1).
        """
        self.current += increment

        # Log at intervals
        should_log = False

        # Check count-based logging
        if self.current % self.log_every == 0:
            should_log = True

        # Check percentage-based logging
        if self.total > 0:
            current_pct = (self.current / self.total) * 100
            if current_pct >= self.last_pct_logged + self.log_every_pct:
                should_log = True
                self.last_pct_logged = int(current_pct / self.log_every_pct) * self.log_every_pct

        if should_log:
            elapsed = time.perf_counter() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / rate if rate > 0 else 0

            if self.total > 0:
                pct = (self.current / self.total) * 100
                logger.info(
                    "  %s: %d/%d (%.1f%%) - %.1f/s, ~%.1fs remaining",
                    self.operation,
                    self.current,
                    self.total,
                    pct,
                    rate,
                    remaining,
                )
            else:
                logger.info(
                    "  %s: %d processed - %.1f/s",
                    self.operation,
                    self.current,
                    rate,
                )

    def finish(self) -> None:
        """Log completion of the operation."""
        elapsed = time.perf_counter() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        logger.info(
            "  %s: completed %d items in %.2fs (%.1f/s)",
            self.operation,
            self.current,
            elapsed,
            rate,
        )


# =============================================================================
# Progress Bar Support (tqdm)
# =============================================================================


def progress_bar[T](
    iterable: Iterable[T],
    desc: str | None = None,
    total: int | None = None,
    unit: str = "it",
    disable: bool = False,
) -> Iterable[T]:
    """Wrap an iterable with a progress bar.

    Uses tqdm when available, falls back to plain iteration otherwise.
    Progress is shown on stderr to avoid interfering with MCP protocol.
    Automatically disabled when stderr is not a TTY (e.g., MCP mode).

    Args:
        iterable: The iterable to wrap.
        desc: Description shown before the progress bar.
        total: Total number of items (required for generators).
        unit: Unit name for the items (e.g., "files", "chunks").
        disable: If True, disable progress bar entirely.

    Returns:
        Wrapped iterable that shows progress.

    Example:
        for file in progress_bar(files, desc="Parsing", unit="files"):
            process(file)
    """
    # Disable if explicitly requested, no tqdm, or running in MCP/non-TTY mode
    if disable or not _HAS_TQDM or _DISABLE_PROGRESS:
        # Fallback: just return the iterable with optional logging
        if not disable and total and total > 100:
            logger.info("  %s: processing %d %s...", desc or "Progress", total, unit)
        return iterable

    return _tqdm(
        iterable,
        desc=f"  {desc}" if desc else None,
        total=total,
        unit=unit,
        file=sys.stderr,
        ncols=80,
        leave=False,  # Clean up after completion
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )


class ProgressBar:
    """Context manager for manual progress bar updates.

    Use when you can't iterate directly but need to update progress manually.

    Example:
        with ProgressBar(total=100, desc="Processing") as pbar:
            for batch in batches:
                process(batch)
                pbar.update(len(batch))
    """

    def __init__(
        self,
        total: int,
        desc: str | None = None,
        unit: str = "it",
        disable: bool = False,
    ):
        """Initialize progress bar.

        Args:
            total: Total number of items to process.
            desc: Description shown before the progress bar.
            unit: Unit name for the items.
            disable: If True, disable progress bar entirely.
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self._pbar: Any = None
        self._current = 0
        self._start_time = 0.0

    def __enter__(self) -> "ProgressBar":
        """Enter context and start progress bar."""
        self._start_time = time.perf_counter()
        # Use tqdm only if available and not in MCP/non-TTY mode
        if not self.disable and _HAS_TQDM and not _DISABLE_PROGRESS:
            self._pbar = _tqdm(
                total=self.total,
                desc=f"  {self.desc}" if self.desc else None,
                unit=self.unit,
                file=sys.stderr,
                ncols=80,
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        elif not self.disable and self.total > 100:
            logger.info("  %s: processing %d %s...", self.desc or "Progress", self.total, self.unit)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context and clean up progress bar."""
        if self._pbar is not None:
            self._pbar.close()
        elif not self.disable:
            elapsed = time.perf_counter() - self._start_time
            rate = self._current / elapsed if elapsed > 0 else 0
            logger.info(
                "  %s: completed %d %s in %.2fs (%.1f/s)",
                self.desc or "Progress",
                self._current,
                self.unit,
                elapsed,
                rate,
            )

    def update(self, n: int = 1) -> None:
        """Update progress by n items.

        Args:
            n: Number of items completed.
        """
        self._current += n
        if self._pbar is not None:
            self._pbar.update(n)

    def set_description(self, desc: str) -> None:
        """Update the description.

        Args:
            desc: New description.
        """
        self.desc = desc
        if self._pbar is not None:
            self._pbar.set_description(f"  {desc}")
