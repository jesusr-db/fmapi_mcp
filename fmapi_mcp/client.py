"""Per-call AsyncOpenAI client factory for fmapi-mcp.

Databricks FMAPI serving endpoints are OpenAI-compatible. Setting
base_url = f"{host}/serving-endpoints/{endpoint_name}" causes the
OpenAI SDK to POST to f"{base_url}/chat/completions", which Databricks
routes correctly for OpenAI-compatible endpoints.

A fresh client is created per call so credential changes (OAuth refresh)
are picked up immediately.
"""
from openai import AsyncOpenAI


def make_client(host: str, token: str, endpoint_name: str) -> AsyncOpenAI:
    """Create an AsyncOpenAI client targeting a specific FMAPI endpoint.

    Args:
        host: Databricks workspace host (e.g., https://adb-xxx.azuredatabricks.net)
        token: Databricks PAT or OAuth token
        endpoint_name: FMAPI serving endpoint name (e.g., databricks-gemini-2-0-flash)

    Returns:
        AsyncOpenAI pointed at {host}/serving-endpoints/{endpoint_name}
    """
    base_url = f"{host}/serving-endpoints/{endpoint_name}"
    return AsyncOpenAI(base_url=base_url, api_key=token)
