"""Public deployment files remain safe for a fresh clone."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_default_compose_starts_without_host_credentials():
    compose = (REPO_ROOT / "docker-compose.yml").read_text()

    assert "required: false" in compose
    assert "/creds/claude.json" not in compose
    assert "/creds/codex.json" not in compose
    assert "/health" in compose


def test_oauth_mounts_are_explicit_and_read_only():
    override = (REPO_ROOT / "docker-compose.oauth.yml").read_text()

    assert "${HOME}/.claude/.credentials.json:/creds/claude.json:ro" in override
    assert "${HOME}/.codex/auth.json:/creds/codex.json:ro" in override


def test_hive_override_joins_the_existing_private_network():
    override = (REPO_ROOT / "docker-compose.hive.yml").read_text()

    assert "external: true" in override
    assert "name: hivemind" in override
    assert "inference-proxy:" in override


def test_runtime_secret_files_are_ignored():
    ignored = (REPO_ROOT / ".gitignore").read_text().splitlines()

    assert ".env" in ignored
    assert "*.pem" in ignored
    assert "data/" in ignored
    assert "*.log" in ignored
