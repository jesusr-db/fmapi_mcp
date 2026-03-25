# fmapi-mcp

An MCP server that exposes [Databricks Foundation Model API](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) endpoints to Claude Code. Lets Claude delegate tasks — text analysis, code review, multimodal reasoning — to models like Gemini 2.0 Flash, GPT-4o, and Llama 3.3 hosted on your Databricks workspace.

---

## Prerequisites

- A Databricks workspace with Foundation Model API enabled
- `~/.databrickscfg` configured with a valid host and token
- Python 3.11+
- `uvx` (ships with `uv`: `pip install uv`)

---

## Installation

Add to Claude Code:

```bash
claude mcp add fmapi -- uvx fmapi-mcp
```

To use a non-default Databricks profile:

```bash
DATABRICKS_CONFIG_PROFILE=my-profile claude mcp add fmapi -- uvx fmapi-mcp
```

You can also authenticate via environment variables instead of `~/.databrickscfg`:

```bash
DATABRICKS_HOST=https://your-workspace.azuredatabricks.net \
DATABRICKS_TOKEN=your-pat-token \
claude mcp add fmapi -- uvx fmapi-mcp
```

---

## Tools

| Tool | Model | Multimodal |
|------|-------|-----------|
| `ask-gemini` | Gemini 2.0 Flash (`databricks-gemini-2-0-flash`) | Yes |
| `ask-gpt4o` | GPT-4o (`databricks-gpt-4o`) | Yes |
| `ask-llama` | Llama 3.3 70B Instruct (`databricks-meta-llama-3-3-70b-instruct`) | Text only |
| `ask-fmapi` | Any FMAPI endpoint (specify by name) | Depends on model |

All tools accept:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | yes | The question or task |
| `system` | string | no | System prompt |
| `files` | string[] | no | Local file paths (images or text) |
| `max_tokens` | integer | no | Max output tokens |

`ask-fmapi` additionally requires `model` (the FMAPI endpoint name).

---

## Sample Queries

Once installed, you can ask Claude to use these tools directly in conversation:

### Text generation

```
Use ask-gemini to summarize this paragraph in one sentence:
"The Databricks Lakehouse Platform unifies data engineering, analytics, and AI
into a single platform built on an open data architecture."
```

```
Use ask-llama to write a Python function that checks if a string is a palindrome.
```

### Code review

```
Use ask-gpt4o to review the following code for bugs and style issues:

def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)
```

### With a system prompt

```
Use ask-gemini with system prompt "You are a concise technical writer" to explain
what a Merkle tree is in two sentences.
```

### With files (multimodal)

```
Use ask-gemini with files=["/path/to/screenshot.png"] to describe what's in this image.
```

```
Use ask-gpt4o with files=["/path/to/diagram.png"] and prompt
"What architecture pattern does this diagram show?"
```

### Text file analysis

```
Use ask-llama with files=["/path/to/logs.txt"] to identify any errors in these logs.
```

### Generic endpoint (ask-fmapi)

```
Use ask-fmapi with model="databricks-claude-3-5-sonnet" to explain the CAP theorem.
```

```
Use ask-fmapi with model="databricks-mixtral-8x7b-instruct" and max_tokens=100
to write a haiku about distributed systems.
```

---

## File Support

| File type | Sent as | Supported by |
|-----------|---------|--------------|
| `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp` | Base64 image | `ask-gemini`, `ask-gpt4o`, `ask-fmapi` (multimodal models) |
| `.txt`, `.md`, `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.csv`, `.xml` | Inline text | All tools |
| `.pdf` | Not supported | — |

`ask-llama` does not accept image files. Pass images to `ask-gemini` or `ask-gpt4o` instead.

---

## Development

```bash
git clone https://github.com/jesusr-db/fmapi_mcp.git
cd fmapi_mcp

python -m venv .venv && source .venv/bin/activate
pip install mcp databricks-sdk openai pytest pytest-asyncio

PYTHONPATH=. pytest tests/test_tools.py tests/test_files.py -v
```

For live workspace testing:

```bash
PYTHONPATH=. python tests/smoke_test.py
```
