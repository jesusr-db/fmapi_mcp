"""Core tool handler for fmapi-mcp.

call_fmapi() is the shared implementation for all four MCP tools.
Returns the full response text (streaming accumulated) or an error string.
No exceptions escape — every error case returns a human-readable string.
"""
from openai import APIStatusError, AuthenticationError, NotFoundError, RateLimitError

from fmapi_mcp.auth import get_credentials
from fmapi_mcp.client import make_client
from fmapi_mcp.files import FileError, build_file_parts, has_image_parts

GEMINI_ENDPOINT = "databricks-gemini-2-0-flash"
GPT4O_ENDPOINT = "databricks-gpt-4o"
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
        endpoint: FMAPI serving endpoint name (e.g., databricks-gemini-2-0-flash)
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
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    # Re-fetch credentials on every call (handles OAuth token refresh)
    host, token = get_credentials()
    client = make_client(host, token, endpoint)

    # Build inference kwargs — omit max_tokens if not provided
    create_kwargs: dict = {
        "model": "placeholder",
        "messages": messages,
        "stream": True,
    }
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens

    try:
        stream = await client.chat.completions.create(**create_kwargs)
        chunks: list[str] = []
        try:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
        except Exception as e:  # noqa: BLE001
            partial = "".join(chunks)
            return f"{partial}\n\n[Stream interrupted: {e}]"
        return "".join(chunks)

    except RateLimitError:
        return f"Rate limited by endpoint '{endpoint}' — retry in a moment"
    except NotFoundError:
        return f"Endpoint '{endpoint}' not found in workspace — check the endpoint name"
    except AuthenticationError:
        return "Auth error — token may have expired, re-check ~/.databrickscfg"
    except APIStatusError as e:
        return f"API error ({e.status_code}): {e.message}"
    except Exception as e:  # noqa: BLE001
        return f"Unexpected error: {e}"
