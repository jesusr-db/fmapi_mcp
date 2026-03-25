"""Async streaming client for Databricks FMAPI /invocations endpoints.

Databricks serving endpoints only accept POST to /invocations — the
OpenAI SDK's /chat/completions path returns 404. We use httpx directly
and parse the SSE stream manually (same format as OpenAI streaming).
"""
import json
from collections.abc import AsyncIterator

import httpx


async def stream_invocations(
    host: str,
    token: str,
    endpoint_name: str,
    messages: list[dict],
    max_tokens: int | None = None,
) -> AsyncIterator[str]:
    """Stream text chunks from a Databricks FMAPI endpoint.

    Args:
        host: Databricks workspace host (e.g., https://adb-xxx.azuredatabricks.net)
        token: Databricks PAT or OAuth token
        endpoint_name: FMAPI serving endpoint name
        messages: OpenAI-format messages list
        max_tokens: Optional output token limit

    Yields:
        Text content chunks as they arrive.

    Raises:
        httpx.HTTPStatusError: on non-2xx responses (caller maps to user-friendly strings)
        httpx.RequestError: on network errors
    """
    url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    body: dict = {"messages": messages, "stream": True}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
