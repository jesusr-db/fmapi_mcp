# tests/test_tools.py
import pytest
import httpx
from unittest.mock import patch


GEMINI = "databricks-gemini-2-5-flash"
LLAMA = "databricks-meta-llama-3-3-70b-instruct"
FAKE_CREDS = ("https://test.databricks.net", "fake-token")


def make_mock_stream(chunks: list[str]):
    """Async generator that yields text chunks, matching stream_invocations signature."""
    async def _gen(*args, **kwargs):
        for text in chunks:
            yield text
    return _gen


def make_failing_stream(chunks: list[str], error: Exception):
    """Async generator that yields some chunks then raises."""
    async def _gen(*args, **kwargs):
        for text in chunks:
            yield text
        raise error
    return _gen


@pytest.mark.asyncio
async def test_text_only_returns_concatenated_chunks():
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=make_mock_stream(["Hello", " world"])),
    ):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(endpoint=GEMINI, prompt="say hello")
    assert result == "Hello world"


@pytest.mark.asyncio
async def test_system_prompt_included_as_first_message():
    captured = {}

    async def _capture(host, token, endpoint, messages, max_tokens=None):
        captured["messages"] = messages
        yield "ok"

    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=_capture),
    ):
        from fmapi_mcp.tools import call_fmapi
        await call_fmapi(endpoint=GEMINI, prompt="hi", system="You are helpful")

    assert captured["messages"][0] == {"role": "system", "content": "You are helpful"}
    assert captured["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_no_system_omits_system_message():
    captured = {}

    async def _capture(host, token, endpoint, messages, max_tokens=None):
        captured["messages"] = messages
        yield "ok"

    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=_capture),
    ):
        from fmapi_mcp.tools import call_fmapi
        await call_fmapi(endpoint=GEMINI, prompt="hi")

    assert all(m["role"] != "system" for m in captured["messages"])
    assert len(captured["messages"]) == 1


@pytest.mark.asyncio
async def test_max_tokens_passed_when_provided():
    captured = {}

    async def _capture(host, token, endpoint, messages, max_tokens=None):
        captured["max_tokens"] = max_tokens
        yield "ok"

    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=_capture),
    ):
        from fmapi_mcp.tools import call_fmapi
        await call_fmapi(endpoint=GEMINI, prompt="hi", max_tokens=256)

    assert captured["max_tokens"] == 256


@pytest.mark.asyncio
async def test_max_tokens_omitted_when_not_provided():
    captured = {}

    async def _capture(host, token, endpoint, messages, max_tokens=None):
        captured["max_tokens"] = max_tokens
        yield "ok"

    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=_capture),
    ):
        from fmapi_mcp.tools import call_fmapi
        await call_fmapi(endpoint=GEMINI, prompt="hi")

    assert captured["max_tokens"] is None


@pytest.mark.asyncio
async def test_llama_rejects_image_files(tmp_path):
    img = tmp_path / "photo.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    with patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS):
        from fmapi_mcp.tools import call_fmapi, LLAMA_ENDPOINT
        result = await call_fmapi(
            endpoint=LLAMA_ENDPOINT,
            prompt="describe this",
            files=[str(img)],
        )
    assert "does not support image files" in result
    assert "ask-gemini" in result or "ask-gpt4o" in result


@pytest.mark.asyncio
async def test_llama_accepts_text_files(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("some text")
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=make_mock_stream(["summarized"])),
    ):
        from fmapi_mcp.tools import call_fmapi, LLAMA_ENDPOINT
        result = await call_fmapi(
            endpoint=LLAMA_ENDPOINT,
            prompt="summarize",
            files=[str(txt)],
        )
    assert result == "summarized"


@pytest.mark.asyncio
async def test_missing_file_returns_error_string():
    with patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(
            endpoint=GEMINI,
            prompt="hi",
            files=["/tmp/nonexistent_fmapi_xyz_abc.txt"],
        )
    assert "File not found" in result


@pytest.mark.asyncio
async def test_image_file_included_in_user_content(tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    captured = {}

    async def _capture(host, token, endpoint, messages, max_tokens=None):
        captured["messages"] = messages
        yield "described"

    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=_capture),
    ):
        from fmapi_mcp.tools import call_fmapi
        await call_fmapi(endpoint=GEMINI, prompt="describe", files=[str(img)])

    user_content = captured["messages"][-1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0] == {"type": "text", "text": "describe"}
    assert user_content[1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_rate_limit_returns_readable_message():
    def _raise(*args, **kwargs):
        async def _gen():
            raise httpx.HTTPStatusError(
                "rate limited",
                request=httpx.Request("POST", "https://test.databricks.net"),
                response=httpx.Response(429, text="too many requests"),
            )
            yield  # make it an async generator
        return _gen()

    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=_raise),
    ):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(endpoint=GEMINI, prompt="hi")
    assert "Rate limited" in result
    assert GEMINI in result


@pytest.mark.asyncio
async def test_not_found_returns_readable_message():
    def _raise(*args, **kwargs):
        async def _gen():
            raise httpx.HTTPStatusError(
                "not found",
                request=httpx.Request("POST", "https://test.databricks.net"),
                response=httpx.Response(404, text="not found"),
            )
            yield
        return _gen()

    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=_raise),
    ):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(endpoint="bad-endpoint", prompt="hi")
    assert "not found" in result.lower()
    assert "bad-endpoint" in result


@pytest.mark.asyncio
async def test_auth_error_returns_readable_message():
    def _raise(*args, **kwargs):
        async def _gen():
            raise httpx.HTTPStatusError(
                "unauthorized",
                request=httpx.Request("POST", "https://test.databricks.net"),
                response=httpx.Response(401, text="unauthorized"),
            )
            yield
        return _gen()

    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=_raise),
    ):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(endpoint=GEMINI, prompt="hi")
    assert "Auth error" in result


@pytest.mark.asyncio
async def test_empty_string_system_is_included_not_omitted():
    captured = {}

    async def _capture(host, token, endpoint, messages, max_tokens=None):
        captured["messages"] = messages
        yield "ok"

    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=_capture),
    ):
        from fmapi_mcp.tools import call_fmapi
        await call_fmapi(endpoint=GEMINI, prompt="hi", system="")

    assert captured["messages"][0] == {"role": "system", "content": ""}
    assert captured["messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_api_status_error_returns_readable_message():
    def _raise(*args, **kwargs):
        async def _gen():
            raise httpx.HTTPStatusError(
                "server error",
                request=httpx.Request("POST", "https://test.databricks.net"),
                response=httpx.Response(500, text="internal server error"),
            )
            yield
        return _gen()

    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=_raise),
    ):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(endpoint=GEMINI, prompt="hi")
    assert "500" in result
    assert "API error" in result


@pytest.mark.asyncio
async def test_stream_interruption_returns_partial_with_annotation():
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.stream_invocations", new=make_failing_stream(["partial"], RuntimeError("connection reset"))),
    ):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(endpoint=GEMINI, prompt="hi")
    assert "partial" in result
    assert "[Stream interrupted:" in result
