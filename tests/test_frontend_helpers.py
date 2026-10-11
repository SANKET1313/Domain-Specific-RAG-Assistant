"""Tests for frontend helper functions and static checks."""
import ast
import os
from unittest.mock import MagicMock, patch

import pytest
import requests


def test_get_backend_url_default():
    from frontend.app import get_backend_url

    with patch.dict(os.environ, {}, clear=True):
        url = get_backend_url()
        assert url == "http://127.0.0.1:8000"


def test_get_backend_url_custom_and_strip_slash():
    from frontend.app import get_backend_url

    with patch.dict(os.environ, {"API_URL": "https://domain-rag-backend.onrender.com/"}):
        url = get_backend_url()
        assert url == "https://domain-rag-backend.onrender.com"


def test_check_backend_health_invalid_url():
    with patch("frontend.app.API_URL", "not_a_valid_url"):
        from frontend.app import check_backend_health

        is_ok, msg = check_backend_health()
        assert is_ok is False
        assert "Invalid API_URL format" in msg


def test_check_backend_health_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("frontend.app.API_URL", "http://127.0.0.1:8000"):
        with patch("requests.get", return_value=mock_resp):
            from frontend.app import check_backend_health

            is_ok, msg = check_backend_health()
            assert is_ok is True
            assert msg == "Backend healthy"


def test_check_backend_health_connection_error():
    with patch("frontend.app.API_URL", "http://127.0.0.1:8000"):
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Refused")):
            from frontend.app import check_backend_health

            is_ok, msg = check_backend_health()
            assert is_ok is False
            assert "Connection refused" in msg


def test_check_backend_health_timeout():
    with patch("frontend.app.API_URL", "http://127.0.0.1:8000"):
        with patch("requests.get", side_effect=requests.exceptions.Timeout("Timed out")):
            from frontend.app import check_backend_health

            is_ok, msg = check_backend_health()
            assert is_ok is False
            assert "timed out" in msg


def test_check_backend_health_http_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.reason = "Service Unavailable"
    with patch("frontend.app.API_URL", "http://127.0.0.1:8000"):
        with patch("requests.get", return_value=mock_resp):
            from frontend.app import check_backend_health

            is_ok, msg = check_backend_health()
            assert is_ok is False
            assert "503" in msg


def test_no_widgets_inside_cached_functions():
    """Static check ensuring no st.* widget calls exist inside @st.cache_data or @st.cache_resource."""
    with open("frontend/app.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    widget_names = {
        "file_uploader",
        "button",
        "selectbox",
        "multiselect",
        "text_input",
        "checkbox",
        "radio",
        "slider",
    }

    cached_functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_cached = any(
                (isinstance(dec, ast.Attribute) and dec.attr in ("cache_data", "cache_resource"))
                or (isinstance(dec, ast.Name) and "cache" in dec.id)
                for dec in node.decorator_list
            )
            if is_cached:
                cached_functions.append(node)

    for func in cached_functions:
        for inner in ast.walk(func):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                if inner.func.attr in widget_names:
                    pytest.fail(f"Widget st.{inner.func.attr} found inside cached function '{func.name}'")


def test_unique_file_uploader_keys():
    """Static check ensuring every st.file_uploader call has a distinct key."""
    with open("frontend/app.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    uploader_keys = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "file_uploader":
                # Find the 'key' keyword argument
                key_node = next((kw for kw in node.keywords if kw.arg == "key"), None)
                assert key_node is not None, "Found st.file_uploader without an explicit key"
                assert isinstance(key_node.value, ast.Constant), "Uploader key must be a constant string"
                uploader_keys.append(key_node.value.value)

    assert len(uploader_keys) == len(set(uploader_keys)), f"Duplicate uploader keys detected: {uploader_keys}"
    assert len(uploader_keys) == 1, f"Expected exactly 1 main uploader, found {len(uploader_keys)}"
    assert uploader_keys[0] == "main_document_uploader"
