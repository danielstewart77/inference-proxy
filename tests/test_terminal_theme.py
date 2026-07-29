"""The proxy UI stays aligned with the shared Hive terminal theme."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = REPO_ROOT / "app" / "templates"


def test_base_uses_terminal_palette_typography_and_brand():
    base = (TEMPLATES / "_base.html").read_text()

    assert "#dce8f0" in base
    assert "#162030" in base
    assert "#38bdf8" in base
    assert "Fira Code" in base
    assert "brand-service" in base
    assert "[proxy &gt;" in base
    assert "cursor-blink" in base


def test_login_uses_terminal_access_gate_treatment():
    login = (TEMPLATES / "admin" / "login.html").read_text()

    assert "hive mind access gate" in login
    assert "login-panel" in login
    assert "login-kicker" in login
    assert "Enter Proxy" in login
    assert "Azure OpenAI Proxy" not in login


def test_status_chart_uses_terminal_accent():
    status = (TEMPLATES / "status.html").read_text()

    assert "'#38bdf8'" in status
    assert "'#4f8cff'" not in status
