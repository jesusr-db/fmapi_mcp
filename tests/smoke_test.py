"""Manual integration test against a live Databricks workspace.

NOT run in CI — run manually with: python tests/smoke_test.py

Prerequisites:
  - Valid ~/.databrickscfg with a Databricks host + token
  - OR: DATABRICKS_HOST + DATABRICKS_TOKEN env vars set
  - The databricks-gemini-2-0-flash endpoint must be enabled in the workspace
"""
import asyncio

from fmapi_mcp.tools import call_fmapi


async def main() -> None:
    print("Smoke test: ask-fmapi → databricks-gemini-2-0-flash")
    result = await call_fmapi(
        endpoint="databricks-gemini-2-0-flash",
        prompt="What is 2+2? Answer in one word only.",
    )
    print(f"Response: {result!r}")
    assert result.strip(), "Got empty response — endpoint may be unavailable"
    print("PASS")


if __name__ == "__main__":
    asyncio.run(main())
