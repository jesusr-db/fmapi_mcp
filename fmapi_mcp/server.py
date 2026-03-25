"""MCP server entry point for fmapi-mcp.

Registers four tools:
  ask-gemini  → databricks-gemini-2-5-flash            (text + images)
  ask-gpt4o   → databricks-gpt-5-4                     (text + images)
  ask-llama   → databricks-meta-llama-3-3-70b-instruct (text only)
  ask-fmapi   → any FMAPI endpoint by name             (generic)

Run via: uvx fmapi-mcp
Add to Claude Code: claude mcp add fmapi -- uvx fmapi-mcp
"""
from mcp.server.fastmcp import FastMCP

from fmapi_mcp.auth import init_auth
from fmapi_mcp.tools import GEMINI_ENDPOINT, GPT4O_ENDPOINT, LLAMA_ENDPOINT, call_fmapi

app = FastMCP("fmapi")


@app.tool(
    name="ask-gemini",
    description=(
        "Ask Gemini 2.0 Flash via Databricks FMAPI. "
        "Supports text prompts and image files (.jpg, .png, .gif, .webp). "
        "Use for multimodal tasks, general questions, and code."
    ),
)
async def ask_gemini(
    prompt: str,
    system: str | None = None,
    files: list[str] | None = None,
    max_tokens: int | None = None,
) -> str:
    return await call_fmapi(GEMINI_ENDPOINT, prompt, system, files, max_tokens)


@app.tool(
    name="ask-gpt4o",
    description=(
        "Ask GPT-4o via Databricks FMAPI. "
        "Supports text prompts and image files (.jpg, .png, .gif, .webp). "
        "Use for multimodal tasks, general questions, and code."
    ),
)
async def ask_gpt4o(
    prompt: str,
    system: str | None = None,
    files: list[str] | None = None,
    max_tokens: int | None = None,
) -> str:
    return await call_fmapi(GPT4O_ENDPOINT, prompt, system, files, max_tokens)


@app.tool(
    name="ask-llama",
    description=(
        "Ask Llama 3.3 70B Instruct via Databricks FMAPI. "
        "Text only — does not support image files. "
        "Use for text analysis, code review, summarization."
    ),
)
async def ask_llama(
    prompt: str,
    system: str | None = None,
    files: list[str] | None = None,
    max_tokens: int | None = None,
) -> str:
    return await call_fmapi(LLAMA_ENDPOINT, prompt, system, files, max_tokens)


@app.tool(
    name="ask-fmapi",
    description=(
        "Ask any Databricks FMAPI endpoint by name. "
        "Use when you need a specific model not covered by ask-gemini/ask-gpt4o/ask-llama. "
        "Example model names: databricks-claude-3-5-sonnet, databricks-mixtral-8x7b-instruct."
    ),
)
async def ask_fmapi_tool(
    model: str,
    prompt: str,
    system: str | None = None,
    files: list[str] | None = None,
    max_tokens: int | None = None,
) -> str:
    return await call_fmapi(model, prompt, system, files, max_tokens)


def main() -> None:
    """Entry point for uvx / CLI invocation.

    Validates Databricks credentials first — exits with a clear message on failure.
    Then starts the MCP server over stdio.
    """
    init_auth()
    app.run()


if __name__ == "__main__":
    main()
