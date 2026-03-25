"""Databricks authentication for fmapi-mcp.

Called at startup to validate credentials, and on every request to
re-fetch host+token (handles OAuth token expiry transparently).

Supports both PAT tokens (config.token) and OAuth / databricks-cli auth
(where config.token is empty but config.authenticate() still works).
"""
import os

from databricks.sdk import WorkspaceClient


def _resolve_credentials(client: WorkspaceClient) -> tuple[str, str]:
    """Extract (host, bearer_token) from a WorkspaceClient.

    Works for PAT, OAuth, and databricks-cli auth types.
    Raises ValueError if either value cannot be resolved.
    """
    host = client.config.host
    if not host:
        raise ValueError("host resolved to empty string")

    # PAT: token is stored directly
    token = client.config.token
    if token:
        return host, token

    # OAuth / databricks-cli: extract Bearer token from auth headers
    headers: dict[str, str] = client.config.authenticate()
    auth_header = headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return host, auth_header[7:]

    raise ValueError(
        "Could not extract Bearer token — check auth_type in ~/.databrickscfg"
    )


def _make_client() -> WorkspaceClient:
    """Create a WorkspaceClient, stripping DATABRICKS_TOKEN if no host is set.

    When DATABRICKS_TOKEN is in the environment but DATABRICKS_HOST is not,
    the SDK raises an error instead of falling back to ~/.databrickscfg.
    We unset the orphaned token so the SDK falls through to the config file.
    """
    token_only = (
        os.environ.get("DATABRICKS_TOKEN")
        and not os.environ.get("DATABRICKS_HOST")
    )
    if token_only:
        token = os.environ.pop("DATABRICKS_TOKEN")
        try:
            return WorkspaceClient()
        finally:
            os.environ["DATABRICKS_TOKEN"] = token
    return WorkspaceClient()


def init_auth() -> tuple[str, str]:
    """Validate Databricks credentials at startup.

    Reads ~/.databrickscfg (default profile, or DATABRICKS_CONFIG_PROFILE env var).
    Also accepts DATABRICKS_HOST + DATABRICKS_TOKEN env vars.

    Returns (host, token) on success.
    Raises SystemExit with a readable message on failure.
    """
    try:
        client = _make_client()
        return _resolve_credentials(client)
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"Databricks auth failed: {e}") from e


def get_credentials() -> tuple[str, str]:
    """Re-fetch host+token for each inference call.

    Creates a fresh WorkspaceClient so the SDK handles OAuth token refresh
    transparently. OAuth tokens expire; this ensures they're always fresh.

    Raises RuntimeError if credentials are unavailable after startup.
    """
    try:
        client = _make_client()
        return _resolve_credentials(client)
    except Exception as e:
        raise RuntimeError(f"Databricks credentials unavailable: {e}") from e
