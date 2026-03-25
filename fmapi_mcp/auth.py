"""Databricks authentication for fmapi-mcp.

Called at startup to validate credentials, and on every request to
re-fetch host+token (handles OAuth token expiry transparently).
"""
from databricks.sdk import WorkspaceClient


def init_auth() -> tuple[str, str]:
    """Validate Databricks credentials at startup.

    Reads ~/.databrickscfg (default profile, or DATABRICKS_CONFIG_PROFILE env var).
    Also accepts DATABRICKS_HOST + DATABRICKS_TOKEN env vars.

    Returns (host, token) on success.
    Raises SystemExit with a readable message on failure.
    """
    try:
        client = WorkspaceClient()
        host = client.config.host
        token = client.config.token
        if not host or not token:
            raise ValueError("host or token resolved to empty string")
        return host, token
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"Databricks auth failed: {e}") from e


def get_credentials() -> tuple[str, str]:
    """Re-fetch host+token for each inference call.

    Creates a fresh WorkspaceClient so the SDK handles OAuth token refresh
    transparently. PAT tokens (common in ~/.databrickscfg) don't expire,
    but OAuth tokens do — this ensures they're always fresh.
    """
    client = WorkspaceClient()
    return client.config.host, client.config.token
