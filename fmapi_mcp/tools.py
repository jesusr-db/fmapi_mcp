"""Core tool handler for fmapi-mcp.

call_fmapi() is the shared implementation for all four MCP tools.
Returns the full response text (streaming accumulated) or an error string.
No exceptions escape — every error case returns a human-readable string.
"""
import httpx

from fmapi_mcp.auth import get_credentials
from fmapi_mcp.client import stream_invocations
from fmapi_mcp.files import FileError, build_file_parts, has_image_parts

GEMINI_ENDPOINT = "databricks-gemini-2-5-flash"
GPT4O_ENDPOINT = "databricks-gpt-5-4"
LLAMA_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"


async def call_fmapi(
    endpoint: str,
    prompt: str,
    system: str | None = None,
    files: list[str] | None = None,
    max_tokens: int | None = None,
) -> str:
    """Invoke a Databricks FMAPI endpoint and return the response text.

    Streams the response internally, accumulating chunks into a single string.
    On mid-stream error: returns partial text + "[Stream interrupted: ...]" annotation.
    All other errors: returns a descriptive error string (no exceptions raised).

    Args:
        endpoint: FMAPI serving endpoint name (e.g., databricks-gemini-2-5-flash)
        prompt: The user query or task
        system: Optional system prompt (omitted from messages if None)
        files: Optional local file paths — images or text files
        max_tokens: Optional output token limit (omitted from request if None)

    Returns:
        Response text string, or human-readable error string.
    """
    # Validate: text-only models reject image files
    if files and endpoint == LLAMA_ENDPOINT and has_image_parts(files):
        return (
            "ask-llama does not support image files — "
            "use ask-gemini or ask-gpt4o for multimodal tasks"
        )

    # Build file content parts
    file_parts: list[dict] = []
    if files:
        try:
            file_parts = build_file_parts(files)
        except FileError as e:
            return str(e)

    # Construct user message content
    if file_parts:
        content: str | list[dict] = [{"type": "text", "text": prompt}] + file_parts
    else:
        content = prompt

    # Build messages list (system prompt only included if provided)
    messages: list[dict] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    # Re-fetch credentials on every call (handles OAuth token refresh)
    host, token = get_credentials()

    chunks: list[str] = []
    try:
        async for chunk in stream_invocations(host, token, endpoint, messages, max_tokens):
            chunks.append(chunk)
        return "".join(chunks)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 401:
            return "Auth error — token may have expired, re-check ~/.databrickscfg"
        if status == 404:
            return f"Endpoint '{endpoint}' not found in workspace — check the endpoint name"
        if status == 429:
            return f"Rate limited by endpoint '{endpoint}' — retry in a moment"
        return f"API error ({status}): {e.response.text[:200]}"
    except httpx.RequestError as e:
        return f"Network error: {e}"
    except Exception as e:  # noqa: BLE001
        partial = "".join(chunks)
        if partial:
            return f"{partial}\n\n[Stream interrupted: {e}]"
        return f"Unexpected error: {e}"
