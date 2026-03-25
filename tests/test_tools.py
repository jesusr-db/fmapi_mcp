# tests/test_tools.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# Helper: create a mock async stream that yields text chunks
def make_mock_stream(chunks: list[str]):
    async def _gen():
        for text in chunks:
            choice = MagicMock()
            choice.delta.content = text
            chunk = MagicMock()
            chunk.choices = [choice]
            yield chunk
    return _gen()


def mock_client_with_stream(chunks: list[str]) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=make_mock_stream(chunks))
    return client


GEMINI = "databricks-gemini-2-0-flash"
LLAMA = "databricks-meta-llama-3-3-70b-instruct"
FAKE_CREDS = ("https://test.databricks.net", "fake-token")


@pytest.mark.asyncio
async def test_text_only_returns_concatenated_chunks():
    client = mock_client_with_stream(["Hello", " world"])
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
    ):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(endpoint=GEMINI, prompt="say hello")
    assert result == "Hello world"


@pytest.mark.asyncio
async def test_system_prompt_included_as_first_message():
    client = mock_client_with_stream(["ok"])
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
    ):
        from fmapi_mcp.tools import call_fmapi
        await call_fmapi(endpoint=GEMINI, prompt="hi", system="You are helpful")
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are helpful"}
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_no_system_omits_system_message():
    client = mock_client_with_stream(["ok"])
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
    ):
        from fmapi_mcp.tools import call_fmapi
        await call_fmapi(endpoint=GEMINI, prompt="hi")
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert all(m["role"] != "system" for m in messages)
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_max_tokens_passed_when_provided():
    client = mock_client_with_stream(["ok"])
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
    ):
        from fmapi_mcp.tools import call_fmapi
        await call_fmapi(endpoint=GEMINI, prompt="hi", max_tokens=256)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs.get("max_tokens") == 256


@pytest.mark.asyncio
async def test_max_tokens_omitted_when_not_provided():
    client = mock_client_with_stream(["ok"])
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
    ):
        from fmapi_mcp.tools import call_fmapi
        await call_fmapi(endpoint=GEMINI, prompt="hi")
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "max_tokens" not in kwargs


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
    client = mock_client_with_stream(["summarized"])
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
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
    client = mock_client_with_stream(["described"])
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
    ):
        from fmapi_mcp.tools import call_fmapi
        await call_fmapi(endpoint=GEMINI, prompt="describe", files=[str(img)])
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    user_content = messages[-1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0] == {"type": "text", "text": "describe"}
    assert user_content[1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_rate_limit_returns_readable_message():
    from openai import RateLimitError
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    err = RateLimitError("rate limited", response=mock_response, body={})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=err)
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
    ):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(endpoint=GEMINI, prompt="hi")
    assert "Rate limited" in result
    assert GEMINI in result


@pytest.mark.asyncio
async def test_not_found_returns_readable_message():
    from openai import NotFoundError
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.headers = {}
    err = NotFoundError("not found", response=mock_response, body={})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=err)
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
    ):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(endpoint="bad-endpoint", prompt="hi")
    assert "not found" in result.lower()
    assert "bad-endpoint" in result


@pytest.mark.asyncio
async def test_auth_error_returns_readable_message():
    from openai import AuthenticationError
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}
    err = AuthenticationError("unauthorized", response=mock_response, body={})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=err)
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
    ):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(endpoint=GEMINI, prompt="hi")
    assert "Auth error" in result


@pytest.mark.asyncio
async def test_empty_string_system_is_included_not_omitted():
    client = mock_client_with_stream(["ok"])
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
    ):
        from fmapi_mcp.tools import call_fmapi
        await call_fmapi(endpoint=GEMINI, prompt="hi", system="")
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": ""}
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_api_status_error_returns_readable_message():
    from openai import APIStatusError
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.headers = {}
    err = APIStatusError("server error", response=mock_response, body={})
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=err)
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
    ):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(endpoint=GEMINI, prompt="hi")
    assert "500" in result
    assert "API error" in result


@pytest.mark.asyncio
async def test_stream_interruption_returns_partial_with_annotation():
    async def _failing_stream():
        choice = MagicMock()
        choice.delta.content = "partial"
        chunk = MagicMock()
        chunk.choices = [choice]
        yield chunk
        raise RuntimeError("connection reset")

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_failing_stream())
    with (
        patch("fmapi_mcp.tools.get_credentials", return_value=FAKE_CREDS),
        patch("fmapi_mcp.tools.make_client", return_value=client),
    ):
        from fmapi_mcp.tools import call_fmapi
        result = await call_fmapi(endpoint=GEMINI, prompt="hi")
    assert "partial" in result
    assert "[Stream interrupted:" in result
