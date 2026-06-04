# ============================================================
# Challenge 5 - Builders Skill Sprint
# MCP Chatbot using Strands SDK + Ollama (llama3.2:3b)
#
# What is MCP?
#   Model Context Protocol (MCP) is an open standard that lets
#   AI agents connect to external tool servers.  An MCP server
#   exposes a set of tools; the agent discovers and calls them
#   at runtime without needing to know how they are implemented.
#
# What this challenge builds:
#   • A local MCP server   – exposes four tools over stdio
#   • A Strands agent      – discovers and uses those MCP tools
#   • An interactive REPL  – lets you chat and watch tool calls happen
#
# How the pieces connect:
#   User → Chat loop → Strands Agent → MCPClient → MCP Server
#                                 ↑ tool results come back ↑
#
# Architecture overview:
#   ┌────────────────────────────────────────────────────────┐
#   │  starter.py  (this file)                               │
#   │                                                        │
#   │   ┌──────────────────┐   stdio   ┌──────────────────┐  │
#   │   │  Strands Agent   │ ◄──────► │   MCP Server     │  │
#   │   │  (llama3.2:3b)   │           │  (same process,  │  │
#   │   └──────────────────┘           │   subprocess)    │  │
#   │                                  │                  │  │
#   │                                  │  Tools exposed:  │  │
#   │                                  │  • calculator    │  │
#   │                                  │  • get_weather   │  │
#   │                                  │  • get_time      │  │
#   │                                  │  • text_stats    │  │
#   └────────────────────────────────────────────────────────┘
# ============================================================


# ============================================================
# SETUP – Install required packages (run once in your terminal)
# ============================================================
#
#   pip install strands-agents strands-agents-tools
#   pip install mcp                  # MCP Python SDK
#   pip install ollama
#
# Make sure Ollama is running and llama3.2:3b is pulled:
#   ollama serve                     # start Ollama (if not already running)
#   ollama pull llama3.2:3b          # download the model (one-time)
#
# Run this file with:
#   python starter.py
#
# ============================================================


# ─────────────────────────────────────────────────────────────
# Standard-library imports
# ─────────────────────────────────────────────────────────────
import sys                     # for sys.executable (MCP server subprocess)
import asyncio                 # MCP client runs async; we bridge it here
from datetime import datetime  # used by the get_time tool
import textwrap                # used by the text_stats tool


# ─────────────────────────────────────────────────────────────
# MCP SDK imports
# ─────────────────────────────────────────────────────────────
# mcp.server.fastmcp      – high-level decorator-based server builder
# mcp.client.stdio        – stdio transport for the MCP client side
# MCPClient               – Strands built-in client that talks to MCP servers
from mcp.server.fastmcp import FastMCP
from mcp.client.stdio import stdio_client, StdioServerParameters
from strands.tools.mcp import MCPClient


# ─────────────────────────────────────────────────────────────
# Strands imports
# ─────────────────────────────────────────────────────────────
from strands import Agent                        # core agent loop
from strands.models.ollama import OllamaModel   # local Ollama adapter


# ============================================================
# SECTION 1 – MCP Server Definition
# ============================================================
# We define a local MCP server using FastMCP.
# FastMCP turns ordinary Python functions into MCP-compatible
# tools by reading their name, type hints, and docstring.
#
# This server will be launched as a subprocess by MCPClient
# (see Section 4).  Communication happens over stdin/stdout
# using the MCP JSON-RPC protocol – no network port needed.
#
# The server is only active while MCPClient has it open.
# ============================================================

# Create the FastMCP server instance.
# The string argument is the server name (used in logs / discovery).
mcp_server = FastMCP("challenge5-local-server")


# ─────────────────────────────────────────────────────────────
# MCP Tool 1: Calculator
# ─────────────────────────────────────────────────────────────
# @mcp_server.tool() registers the function as an MCP tool.
# The agent reads the docstring to understand what the tool does
# and when to call it.
@mcp_server.tool()
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the numeric result.

    Use this tool for any arithmetic: addition, subtraction,
    multiplication, division, exponentiation, or mixed expressions.

    Args:
        expression: A math expression string, e.g. "2 + 3 * 4"

    Returns:
        The result as a string, or an error message.
    """
    # Restrict eval() to safe built-ins only – prevents code injection.
    safe_names = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "pow": pow,
        "min": min,
        "max": max,
    }
    try:
        result = eval(expression, safe_names)  # type: ignore[arg-type]
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


# ─────────────────────────────────────────────────────────────
# MCP Tool 2: Weather (mock)
# ─────────────────────────────────────────────────────────────
@mcp_server.tool()
def get_weather(city: str) -> str:
    """
    Return the current weather for a given city.

    Use this tool when the user asks about weather, temperature,
    or conditions in any city.

    Args:
        city: Name of the city, e.g. "London" or "Tokyo"

    Returns:
        A weather summary string (mock data).
    """
    mock_db: dict[str, str] = {
        "london":    "Cloudy, 15°C, humidity 80%, light drizzle.",
        "new york":  "Sunny, 22°C, humidity 55%, clear skies.",
        "tokyo":     "Partly cloudy, 18°C, humidity 70%, gentle breeze.",
        "sydney":    "Sunny, 25°C, humidity 60%, perfect beach weather.",
        "paris":     "Overcast, 13°C, humidity 75%, chance of showers.",
        "dubai":     "Hot and sunny, 38°C, humidity 40%, dry heat.",
        "toronto":   "Snowy, -2°C, humidity 85%, 5 cm of fresh snow.",
        "singapore": "Humid, 30°C, humidity 90%, afternoon thunderstorm likely.",
        "mumbai":    "Warm, 32°C, humidity 85%, partly cloudy.",
        "berlin":    "Cool, 10°C, humidity 70%, overcast.",
    }
    key = city.strip().lower()
    if key in mock_db:
        return f"Weather in {city.title()}: {mock_db[key]}"
    return (
        f"Weather in {city.title()}: 20°C, partly cloudy, humidity 65%. "
        "(Mock data – add this city to the mock_db for a real entry.)"
    )


# ─────────────────────────────────────────────────────────────
# MCP Tool 3: Get Current Time
# ─────────────────────────────────────────────────────────────
@mcp_server.tool()
def get_time(timezone: str = "local") -> str:
    """
    Return the current date and time.

    Use this tool whenever the user asks what time or date it is.

    Args:
        timezone: Currently supports "local" (default) or "utc".

    Returns:
        A formatted date-time string.
    """
    now_local = datetime.now()
    now_utc   = datetime.utcnow()

    if timezone.strip().lower() == "utc":
        return f"Current UTC time : {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"

    # Default: local machine time
    return f"Current local time: {now_local.strftime('%Y-%m-%d %H:%M:%S')}"


# ─────────────────────────────────────────────────────────────
# MCP Tool 4: Text Statistics
# ─────────────────────────────────────────────────────────────
@mcp_server.tool()
def text_stats(text: str) -> str:
    """
    Analyse a piece of text and return basic statistics.

    Use this tool when the user wants to count words, characters,
    sentences, or lines in any text they provide.

    Args:
        text: The text to analyse.

    Returns:
        A multi-line statistics summary.
    """
    if not text.strip():
        return "Error: No text provided."

    char_count    = len(text)
    char_no_space = len(text.replace(" ", ""))
    word_count    = len(text.split())
    line_count    = len(text.splitlines()) or 1
    # Count sentences by splitting on terminal punctuation.
    sentence_count = max(
        1,
        text.count(".") + text.count("!") + text.count("?"),
    )
    # Estimate average word length
    words = text.split()
    avg_word_len = (
        round(sum(len(w.strip(".,!?;:\"'")) for w in words) / len(words), 1)
        if words else 0
    )

    return (
        f"Text Statistics:\n"
        f"  Characters (with spaces)   : {char_count}\n"
        f"  Characters (without spaces): {char_no_space}\n"
        f"  Words                      : {word_count}\n"
        f"  Lines                      : {line_count}\n"
        f"  Sentences (approx.)        : {sentence_count}\n"
        f"  Average word length        : {avg_word_len} chars"
    )


# ============================================================
# SECTION 2 – MCP Server Entry-Point Guard
# ============================================================
# When MCPClient launches this file as a subprocess it passes
# the flag  --mcp-server  on the command line.
#
# Why a subprocess?
#   MCP uses the stdio transport: the client writes JSON-RPC
#   requests to stdin and reads responses from stdout.
#   Running the server in a subprocess keeps the server's I/O
#   separate from the interactive chat output in the main process.
#
# When --mcp-server is detected:
#   • The server starts, listens on stdin/stdout, and serves
#     tool calls until the client closes the connection.
#   • sys.exit() is called so none of the chat-loop code below
#     runs in the subprocess.
# ============================================================
if "--mcp-server" in sys.argv:
    # This branch runs ONLY in the subprocess launched by MCPClient.
    # run() blocks and handles MCP protocol messages over stdio.
    mcp_server.run(transport="stdio")
    sys.exit(0)


# ============================================================
# SECTION 3 – Ollama Model Configuration
# ============================================================
# Same pattern as previous challenges: OllamaModel points
# Strands at the local Ollama server.
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,   # lower → more consistent, reliable tool-use decisions
)


# ============================================================
# SECTION 4 – MCP Client Setup
# ============================================================
# MCPClient is Strands' built-in bridge between the agent and
# any MCP server.  It:
#   1. Launches the MCP server as a subprocess (using the
#      command we provide).
#   2. Performs the MCP handshake to list available tools.
#   3. Returns Strands-compatible tool objects the agent can call.
#
# How the subprocess command works:
#   sys.executable  → the Python interpreter currently running
#   __file__        → this script (starter.py)
#   "--mcp-server"  → flag that triggers the server branch above
#
# The MCPClient uses a context manager so it cleans up the
# subprocess automatically when the chat loop exits.
# ============================================================

# Build the command that MCPClient will use to start the server.
mcp_command = [sys.executable, __file__, "--mcp-server"]

print("=" * 60)
print("  Challenge 5 – MCP Chatbot  (llama3.2:3b + MCP Tools)")
print("=" * 60)
print()
print("Starting MCP server subprocess …")


# ============================================================
# SECTION 5 – Agent Creation with MCP Tools
# ============================================================
# We open the MCPClient as a context manager.
# Inside the `with` block:
#   • mcp_client.list_tools_sync() returns the tool objects.
#   • We pass those tools to the Agent constructor.
#   • The agent can now call any of the four MCP tools.
#
# The context manager ensures the subprocess is terminated
# cleanly when the `with` block exits (i.e. when we type "exit").
# ============================================================
with MCPClient(lambda: stdio_client(
    StdioServerParameters(
        command=mcp_command[0],
        args=mcp_command[1:],
    )
)) as mcp_client:

    # ── Discover tools from the MCP server ───────────────────
    # list_tools_sync() performs the MCP "tools/list" request
    # and converts the response into Strands tool objects.
    mcp_tools = mcp_client.list_tools_sync()

    print(f"MCP server connected.  {len(mcp_tools)} tools discovered:")
    for t in mcp_tools:
        # Each tool object has a .tool_name attribute.
        print(f"  • {t.tool_name}")
    print()

    # ── Create the agent with MCP tools ──────────────────────
    agent = Agent(
        model=ollama_model,
        tools=mcp_tools,          # tools come from the MCP server
        system_prompt=(
            "You are a helpful MCP chatbot powered by a local MCP server.\n\n"
            "Tools available via MCP:\n"
            "  • calculator  – evaluate any math expression\n"
            "  • get_weather – get weather info for a city\n"
            "  • get_time    – get the current date and time\n"
            "  • text_stats  – count words, characters, lines in text\n\n"
            "Always use the appropriate MCP tool when the user's request "
            "matches one of the tool capabilities.\n"
            "Be concise, accurate, and friendly."
        ),
    )

    # ── Print usage examples ──────────────────────────────────
    print("MCP tools are ready.  Try these example prompts:")
    print()
    print("  Calculator  → 'What is 15 * 48 + 200?'")
    print("  Weather     → 'What is the weather in Tokyo?'")
    print("  Time        → 'What time is it?'")
    print("  Text stats  → 'Count the words in: The quick brown fox'")
    print("  Mixed       → 'What is 2 ** 10 and what time is it?'")
    print()
    print("Type 'tools' to list available MCP tools.")
    print("Type 'exit' to quit.")
    print("=" * 60)
    print()

    # ============================================================
    # SECTION 6 – Interactive Chat Loop
    # ============================================================
    # Each iteration:
    #   (a) Read user input from stdin.
    #   (b) Handle the special 'tools' command.
    #   (c) Send the message to the agent.
    #       The agent decides whether to answer directly or call
    #       one (or more) MCP tools, then streams the final reply.
    # ============================================================
    while True:
        # ── (a) Read input ───────────────────────────────────
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl-C / Ctrl-D gracefully
            print("\nGoodbye!")
            break

        # ── Exit condition ────────────────────────────────────
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break

        # ── Skip blank lines ──────────────────────────────────
        if not user_input:
            continue

        # ── (b) Special command: list MCP tools ───────────────
        # Useful to inspect what the server is exposing at runtime.
        if user_input.lower() == "tools":
            print("\n── Available MCP Tools ──────────────────────")
            for t in mcp_tools:
                print(f"  • {t.tool_name}")
            print("─────────────────────────────────────────────\n")
            continue

        # ── (c) Send message to the agent ────────────────────
        # Strands manages the full reasoning loop:
        #   user message
        #     → model decides to call an MCP tool (or answer directly)
        #     → MCPClient forwards the call to the MCP server subprocess
        #     → tool result is returned to the model
        #     → model generates the final answer
        #     → answer is streamed to stdout
        print("Agent: ", end="", flush=True)
        agent(user_input)
        print()   # blank line for readability

# ── Cleanup note ─────────────────────────────────────────────
# When the `with MCPClient(…)` block exits here, the MCP server
# subprocess is terminated automatically.  No manual cleanup needed.
print("MCP server stopped.  Session ended.")
