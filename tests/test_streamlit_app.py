"""Smoke test for the Streamlit app itself.

Phase 13 was originally "verified" by curling the server for HTTP 200 --
which does NOT execute the app script (Streamlit only runs it when a
client connects). That gap let a real startup failure ship: Streamlit's
hot-reload watcher walks sys.modules, transformers' lazy __getattr__ turns
that inspection into real imports of unrelated vision models, and one of
them requires torchvision (deliberately not installed -- this project is
text-only). See .streamlit/config.toml.

AppTest runs the script in-process the way a real browser session does,
so this catches script-level breakage for real.
"""

import pytest
from streamlit.testing.v1 import AppTest

import config

# AppTest resolves a relative path against the CALLING file (tests/), not cwd.
_APP_PATH = str(config.PROJECT_ROOT / "streamlit_app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(_APP_PATH, default_timeout=180)
    at.run()
    return at


def test_app_script_runs_without_exception(app):
    assert not app.exception, f"app raised: {[str(e) for e in app.exception]}"


def test_app_renders_title_and_chat_input(app):
    assert any("Car Sales Assistant" in t.value for t in app.title)
    assert len(app.chat_input) == 1


def test_app_sidebar_controls_present(app):
    labels = [cb.label for cb in app.checkbox]
    assert any("debug" in label.lower() for label in labels)
    assert any("expansion" in label.lower() or "hyde" in label.lower() for label in labels)


def test_file_watcher_disabled_so_transformers_is_never_module_walked():
    # The actual fix for the torchvision/zoedepth crash: app_session.py only
    # constructs the module-walking watcher when this is != "none".
    assert config_option("server.fileWatcherType") == "none"


def config_option(name: str):
    from streamlit import config as st_config

    st_config.get_config_options()
    return st_config.get_option(name)
