"""Tests for cache isolation and manage_cache tool."""

from pathlib import Path

import pytest

from nodestradamus.mcp.tools.definitions import MANAGE_CACHE_TOOL
from nodestradamus.utils.cache import get_cache_dir, get_repo_hash, parse_file_uri
from nodestradamus.utils.registry import (
    get_registry_path,
    get_repo_for_hash,
    list_registered_repos,
    load_registry,
    register_analysis,
    save_registry,
    unregister_analysis,
)


class TestCacheDir:
    """Tests for cache directory computation."""

    def test_get_cache_dir_without_workspace_returns_repo_local(self) -> None:
        """Without workspace_path, cache is in the analyzed repo."""
        repo_path = "/home/user/project"
        cache_dir = get_cache_dir(None, repo_path)

        assert cache_dir == Path("/home/user/project/.nodestradamus")

    def test_get_cache_dir_with_workspace_returns_workspace_scoped(self) -> None:
        """With workspace_path, cache is scoped to workspace."""
        workspace_path = "/home/user/workspace"
        repo_path = "/home/user/project"

        cache_dir = get_cache_dir(workspace_path, repo_path)

        # cache_dir is: <workspace>/.nodestradamus/cache/<hash>
        # cache_dir.parent is: <workspace>/.nodestradamus/cache
        # cache_dir.parent.parent is: <workspace>/.nodestradamus
        assert cache_dir.parent.parent == Path("/home/user/workspace/.nodestradamus")
        assert cache_dir.parent.name == "cache"
        # Subdirectory should be a 16-char hash
        assert len(cache_dir.name) == 16
        assert all(c in "0123456789abcdef" for c in cache_dir.name)

    def test_get_cache_dir_is_deterministic(self) -> None:
        """Same inputs produce same cache directory."""
        workspace = "/home/user/workspace"
        repo = "/home/user/project"

        cache1 = get_cache_dir(workspace, repo)
        cache2 = get_cache_dir(workspace, repo)

        assert cache1 == cache2

    def test_get_cache_dir_different_repos_different_dirs(self) -> None:
        """Different repos get different cache directories."""
        workspace = "/home/user/workspace"
        repo1 = "/home/user/project1"
        repo2 = "/home/user/project2"

        cache1 = get_cache_dir(workspace, repo1)
        cache2 = get_cache_dir(workspace, repo2)

        assert cache1 != cache2


class TestRepoHash:
    """Tests for repository hash computation."""

    def test_get_repo_hash_returns_16_chars(self) -> None:
        """Hash is exactly 16 characters."""
        repo_hash = get_repo_hash("/home/user/project")

        assert len(repo_hash) == 16
        assert all(c in "0123456789abcdef" for c in repo_hash)

    def test_get_repo_hash_is_deterministic(self) -> None:
        """Same repo path produces same hash."""
        hash1 = get_repo_hash("/home/user/project")
        hash2 = get_repo_hash("/home/user/project")

        assert hash1 == hash2

    def test_get_repo_hash_different_paths_different_hashes(self) -> None:
        """Different paths produce different hashes."""
        hash1 = get_repo_hash("/home/user/project1")
        hash2 = get_repo_hash("/home/user/project2")

        assert hash1 != hash2

    def test_get_repo_hash_resolves_paths(self) -> None:
        """Hash uses resolved absolute path."""
        # Note: These will resolve differently on actual filesystem
        # but the test verifies the hash changes based on path
        hash1 = get_repo_hash("/home/user/project")
        hash2 = get_repo_hash("/home/user/other/../project")

        # After resolution, these should be the same path
        assert hash1 == hash2


class TestParseFileUri:
    """Tests for file URI parsing."""

    def test_parse_file_uri_extracts_path(self) -> None:
        """File URIs are parsed correctly."""
        uri = "file:///home/user/workspace"
        path = parse_file_uri(uri)

        assert path == Path("/home/user/workspace")

    def test_parse_file_uri_returns_none_for_http(self) -> None:
        """Non-file URIs return None."""
        uri = "http://example.com/path"
        path = parse_file_uri(uri)

        assert path is None

    def test_parse_file_uri_handles_windows_paths(self) -> None:
        """Windows-style file URIs are parsed."""
        uri = "file:///C:/Users/user/workspace"
        path = parse_file_uri(uri)

        # On Unix, this will be parsed as a path starting with /C:
        assert path is not None
        assert "Users" in str(path)


class TestRegistry:
    """Tests for workspace registry."""

    def test_load_registry_returns_empty_for_missing(self, temp_dir: Path) -> None:
        """Loading nonexistent registry returns empty structure."""
        registry = load_registry(temp_dir)

        assert registry == {"repos": {}, "version": 1}

    def test_save_and_load_registry(self, temp_dir: Path) -> None:
        """Registry can be saved and loaded."""
        registry = {
            "repos": {
                "abc123": {
                    "path": "/home/user/project",
                    "hash": "abc123",
                    "last_analyzed": "2026-01-01T00:00:00",
                }
            },
            "version": 1,
        }

        save_registry(temp_dir, registry)
        loaded = load_registry(temp_dir)

        assert loaded == registry

    def test_registry_path_is_in_nodestradamus_dir(self, temp_dir: Path) -> None:
        """Registry file is in .nodestradamus directory."""
        registry_path = get_registry_path(temp_dir)

        assert registry_path.parent.name == ".nodestradamus"
        assert registry_path.name == "registry.json"

    def test_register_analysis_adds_entry(self, temp_dir: Path) -> None:
        """Registering analysis adds entry to registry."""
        repo_path = "/home/user/project"

        register_analysis(temp_dir, repo_path, cache_size=1024)

        registry = load_registry(temp_dir)
        repo_hash = get_repo_hash(repo_path)

        assert repo_hash in registry["repos"]
        entry = registry["repos"][repo_hash]
        assert entry["path"] == str(Path(repo_path).resolve())
        assert entry["cache_size"] == 1024

    def test_unregister_analysis_removes_entry(self, temp_dir: Path) -> None:
        """Unregistering analysis removes entry from registry."""
        repo_path = "/home/user/project"

        # First register
        register_analysis(temp_dir, repo_path)

        # Then unregister
        result = unregister_analysis(temp_dir, repo_path)

        assert result is True
        registry = load_registry(temp_dir)
        repo_hash = get_repo_hash(repo_path)
        assert repo_hash not in registry["repos"]

    def test_unregister_nonexistent_returns_false(self, temp_dir: Path) -> None:
        """Unregistering nonexistent repo returns False."""
        result = unregister_analysis(temp_dir, "/nonexistent/path")

        assert result is False

    def test_get_repo_for_hash_finds_registered(self, temp_dir: Path) -> None:
        """Can look up repo path by hash."""
        repo_path = "/home/user/project"
        register_analysis(temp_dir, repo_path)
        repo_hash = get_repo_hash(repo_path)

        found_path = get_repo_for_hash(temp_dir, repo_hash)

        assert found_path == str(Path(repo_path).resolve())

    def test_get_repo_for_hash_returns_none_for_unknown(self, temp_dir: Path) -> None:
        """Looking up unknown hash returns None."""
        found = get_repo_for_hash(temp_dir, "unknown_hash")

        assert found is None

    def test_list_registered_repos(self, temp_dir: Path) -> None:
        """Can list all registered repos."""
        register_analysis(temp_dir, "/home/user/project1")
        register_analysis(temp_dir, "/home/user/project2")

        repos = list_registered_repos(temp_dir)

        assert len(repos) == 2
        paths = [r["path"] for r in repos]
        assert str(Path("/home/user/project1").resolve()) in paths
        assert str(Path("/home/user/project2").resolve()) in paths


class TestManageCacheTool:
    """Tests for manage_cache MCP tool definition."""

    def test_tool_has_correct_name(self) -> None:
        """Tool name is manage_cache."""
        assert MANAGE_CACHE_TOOL.name == "manage_cache"

    def test_tool_has_mode_parameter(self) -> None:
        """Tool has mode parameter with correct enum values."""
        properties = MANAGE_CACHE_TOOL.inputSchema["properties"]

        assert "mode" in properties
        assert set(properties["mode"]["enum"]) == {"info", "clear", "list"}

    def test_tool_has_repo_path_parameter(self) -> None:
        """Tool has repo_path parameter."""
        properties = MANAGE_CACHE_TOOL.inputSchema["properties"]

        assert "repo_path" in properties

    def test_tool_has_workspace_path_parameter(self) -> None:
        """Tool has workspace_path parameter."""
        properties = MANAGE_CACHE_TOOL.inputSchema["properties"]

        assert "workspace_path" in properties

    def test_only_mode_is_required(self) -> None:
        """Only mode parameter is required."""
        required = MANAGE_CACHE_TOOL.inputSchema["required"]

        assert required == ["mode"]


class TestCacheIsolation:
    """Tests for cache isolation between workspaces."""

    def test_different_workspaces_different_cache_paths(self) -> None:
        """Same repo analyzed from different workspaces uses different caches."""
        workspace1 = "/home/user/workspace1"
        workspace2 = "/home/user/workspace2"
        repo = "/shared/project"

        cache1 = get_cache_dir(workspace1, repo)
        cache2 = get_cache_dir(workspace2, repo)

        # Caches should be in different workspaces
        assert "/workspace1/" in str(cache1)
        assert "/workspace2/" in str(cache2)
        assert cache1 != cache2

    def test_same_workspace_same_repo_same_cache(self) -> None:
        """Same repo from same workspace uses same cache."""
        workspace = "/home/user/workspace"
        repo = "/shared/project"

        cache1 = get_cache_dir(workspace, repo)
        cache2 = get_cache_dir(workspace, repo)

        assert cache1 == cache2


class TestCacheManagementIntegration:
    """Integration tests for cache management."""

    @pytest.fixture
    def setup_caches(self, temp_dir: Path):
        """Set up test caches in a workspace."""
        workspace = temp_dir / "workspace"
        workspace.mkdir()

        # Create some fake caches
        cache_root = workspace / ".nodestradamus" / "cache"
        cache_root.mkdir(parents=True)

        # Create fake embeddings files for two "repos"
        for repo_name in ["project1", "project2"]:
            repo_hash = get_repo_hash(f"/fake/{repo_name}")
            cache_dir = cache_root / repo_hash
            cache_dir.mkdir()
            embeddings_file = cache_dir / "embeddings.npz"
            embeddings_file.write_bytes(b"fake embeddings data")

            # Register in registry
            register_analysis(workspace, f"/fake/{repo_name}", cache_size=21)

        return workspace

    def test_caches_can_be_listed(self, setup_caches: Path) -> None:
        """Caches in workspace can be listed."""
        workspace = setup_caches
        cache_root = workspace / ".nodestradamus" / "cache"

        # Should have two cache directories
        cache_dirs = list(cache_root.iterdir())
        assert len(cache_dirs) == 2

    def test_registry_tracks_caches(self, setup_caches: Path) -> None:
        """Registry tracks all analyzed repos."""
        workspace = setup_caches
        repos = list_registered_repos(workspace)

        assert len(repos) == 2
        paths = [r["path"] for r in repos]
        assert str(Path("/fake/project1").resolve()) in paths
        assert str(Path("/fake/project2").resolve()) in paths

    def test_cache_clear_removes_files(self, setup_caches: Path) -> None:
        """Clearing cache removes the cache directory."""
        workspace = setup_caches
        repo_path = "/fake/project1"
        repo_hash = get_repo_hash(repo_path)

        cache_dir = workspace / ".nodestradamus" / "cache" / repo_hash
        assert cache_dir.exists()

        # Unregister and remove cache
        unregister_analysis(workspace, repo_path)

        # Manually remove cache (as the tool would do)
        import shutil
        shutil.rmtree(cache_dir)

        assert not cache_dir.exists()

        # Registry should no longer have this repo
        repos = list_registered_repos(workspace)
        paths = [r["path"] for r in repos]
        assert str(Path(repo_path).resolve()) not in paths
