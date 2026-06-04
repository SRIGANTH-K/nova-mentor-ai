# ============================================================
# Challenge 4 - Builders Skill Sprint
# Full Agent using Strands SDK + Ollama (llama3.2:3b)
#
# This challenge combines EVERYTHING from Challenges 1–3:
#   • Challenge 1 concept : basic agent + chat loop
#   • Challenge 2 concept : custom tools (calculator, weather, age)
#   • Challenge 3 concept : persistent memory with Mem0 + FAISS
#
# What the agent can do:
#   1. calculator tool       – evaluate math expressions safely
#   2. get_weather tool      – return mock weather for any city
#   3. calculate_age tool    – compute age from a birth year
#   4. Persistent memory     – remember user facts across turns
#                              (stored in a local FAISS vector DB)
#   5. User info recall      – retrieve relevant memories and
#                              inject them into every prompt
# ============================================================


# ============================================================
# SETUP – Install required packages (run once in your terminal)
# ============================================================
# pip install strands-agents strands-agents-tools
# pip install mem0ai
# pip install faiss-cpu          # CPU build; use faiss-gpu for CUDA
# pip install sentence-transformers  # local embeddings, no API key
# pip install ollama
# ============================================================


# ─────────────────────────────────────────────────────────────
# Standard-library imports
# ─────────────────────────────────────────────────────────────
import os                    # used to silence noisy log output
from datetime import datetime  # needed by the age calculator tool

# ─────────────────────────────────────────────────────────────
# Suppress verbose warnings from HuggingFace tokenizers
# (must happen BEFORE importing transformers / sentence_transformers)
# ─────────────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────────────────────────────────────────
# Strands imports
# ─────────────────────────────────────────────────────────────
from strands import Agent, tool          # Agent = core loop, tool = decorator
from strands.models.ollama import OllamaModel  # local Ollama adapter

# ─────────────────────────────────────────────────────────────
# Mem0 import – our persistent memory layer
# ─────────────────────────────────────────────────────────────
from mem0 import Memory


# ============================================================
# SECTION 1 – Tool Definitions
# ============================================================
# The @tool decorator turns a plain Python function into a
# Strands-compatible tool.  The agent reads the function name,
# type hints, and docstring to decide when and how to call it.
# Always write a clear docstring – it is the agent's manual.

# ------------------------------------------------------------
# Tool 1: Calculator
# Uses a restricted eval() so only safe math operations run.
# ------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic mathematical expression and return the result.

    Use this tool whenever the user asks you to perform arithmetic:
    addition, subtraction, multiplication, division, exponentiation,
    or any combination thereof.

    Args:
        expression: A math expression string, e.g. "2 + 3 * 4" or "10 / 2"

    Returns:
        The numeric result as a string, or an error message if the
        expression is invalid or contains unsafe code.
    """
    # Restrict eval() to safe math builtins only; block everything else.
    safe_names = {
        "__builtins__": {},          # no built-in functions
        "abs": abs,
        "round": round,
        "pow": pow,
        "min": min,
        "max": max,
    }
    try:
        result = eval(expression, safe_names)   # type: ignore[arg-type]
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


# ------------------------------------------------------------
# Tool 2: Weather
# Returns mock data so the demo works without an API key.
# In production, replace this with a real weather API call.
# ------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """
    Return the current weather report for a given city.

    Use this tool whenever the user asks about weather, temperature,
    or forecast for any location.

    Args:
        city: The name of the city to look up, e.g. "London" or "Tokyo"

    Returns:
        A short weather summary string for that city.
    """
    # Mock weather database – swap for a real API (e.g. OpenWeatherMap)
    # to get live data without any other code changes.
    mock_weather: dict[str, str] = {
        "london":    "Cloudy, 15°C, humidity 80%, light drizzle.",
        "new york":  "Sunny, 22°C, humidity 55%, clear skies.",
        "tokyo":     "Partly cloudy, 18°C, humidity 70%, gentle breeze.",
        "sydney":    "Sunny, 25°C, humidity 60%, perfect beach weather.",
        "paris":     "Overcast, 13°C, humidity 75%, chance of showers.",
        "dubai":     "Hot and sunny, 38°C, humidity 40%, dry heat.",
        "toronto":   "Snowy, -2°C, humidity 85%, 5 cm of fresh snow.",
        "singapore": "Humid, 30°C, humidity 90%, afternoon thunderstorm likely.",
    }

    key = city.strip().lower()
    if key in mock_weather:
        return f"Weather in {city.title()}: {mock_weather[key]}"

    # Generic fallback for any city not in the mock DB
    return (
        f"Weather in {city.title()}: 20°C, partly cloudy, humidity 65%. "
        "(Mock data – real API not configured)"
    )


# ------------------------------------------------------------
# Tool 3: Age Calculator
# Computes how old someone is based on the year they were born.
# ------------------------------------------------------------
@tool
def calculate_age(birth_year: int) -> str:
    """
    Calculate a person's current age based on their birth year.

    Use this tool when the user wants to know how old someone is,
    or asks how many years ago a given year was.

    Args:
        birth_year: The four-digit year of birth, e.g. 1990

    Returns:
        The calculated age in years, or an error message for invalid input.
    """
    current_year = datetime.now().year

    if birth_year < 1900:
        return "Error: Birth year seems too far in the past. Please provide a year after 1900."
    if birth_year > current_year:
        return f"Error: Birth year {birth_year} is in the future. Please enter a past year."

    age = current_year - birth_year
    return f"Someone born in {birth_year} is {age} years old in {current_year}."


# ============================================================
# SECTION 2 – Mem0 + FAISS Configuration
# ============================================================
# Mem0 is a memory layer that sits between the chat loop and
# the agent.  It:
#   • converts raw text into fact vectors (via the embedder)
#   • stores those vectors locally in a FAISS index
#   • retrieves the most relevant facts on every new turn
#
# All three sub-sections below point to LOCAL resources so the
# agent works 100 % offline with no paid API keys.

MEM0_CONFIG = {
    # ── Vector store ──────────────────────────────────────────
    # FAISS stores embedding vectors in flat files on disk.
    # The folder is created automatically on first run.
    "vector_store": {
        "provider": "faiss",
        "config": {
            "embedding_model_dims": 384,           # must match embedder output size
            "path": "./challenge4_memory_db",      # where the FAISS files are saved
        },
    },
    # ── Embedder ──────────────────────────────────────────────
    # all-MiniLM-L6-v2 is a fast, ~80 MB model that produces
    # 384-dimensional sentence vectors.  Downloaded automatically
    # on the first run via the sentence-transformers library.
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },
    # ── LLM used by Mem0 internally ───────────────────────────
    # Mem0 calls an LLM to EXTRACT structured facts from raw
    # conversation text before embedding them.  We reuse our
    # local Ollama model so nothing leaves the machine.
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "ollama_base_url": "http://localhost:11434",
            "temperature": 0,       # deterministic extraction is better for facts
            "max_tokens": 2000,
        },
    },
}

# Initialise Mem0 with the config above.
# First run: downloads the ~80 MB embedding model automatically.
print("Initialising memory layer (first run downloads ~80 MB embedding model) …")
memory = Memory.from_config(MEM0_CONFIG)
print("Memory layer ready.\n")


# ============================================================
# SECTION 3 – Ollama Model Configuration
# ============================================================
# OllamaModel tells Strands where to find the local Ollama
# server and which model to use for chat completions.
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,   # lower → more consistent tool-use decisions
)


# ============================================================
# SECTION 4 – Agent Setup
# ============================================================
# We register all three tools and provide a system prompt that:
#   • explains the available tools
#   • tells the agent how to use the injected memory block
#   • keeps tone friendly and concise
agent = Agent(
    model=ollama_model,
    tools=[calculator, get_weather, calculate_age],  # all three tools registered
    system_prompt=(
        "You are a helpful personal assistant with three tools and a long memory.\n\n"
        "Tools available:\n"
        "  • calculator     – for any maths calculation\n"
        "  • get_weather    – for weather info about a city\n"
        "  • calculate_age  – to find someone's age from their birth year\n\n"
        "Memory:\n"
        "  When the prompt contains a 'Relevant memories' block, treat those\n"
        "  facts as true and use them to personalise your answer.\n"
        "  For example, if memory says the user's name is Alex, greet them by name.\n\n"
        "Always use the appropriate tool when the user's question matches one of the\n"
        "tool capabilities.  Be concise, accurate, and friendly."
    ),
)


# ============================================================
# SECTION 5 – Memory Helper: retrieve and enrich the prompt
# ============================================================
# Before forwarding a message to the agent we search Mem0 for
# facts that are semantically similar to what the user just said.
# Those facts are prepended to the prompt so the agent has full
# context about the person it is talking to.

def build_prompt_with_memory(user_message: str, user_id: str) -> str:
    """
    Search Mem0 / FAISS for facts related to `user_message` and
    prepend them to the message so the agent can reference them.

    Args:
        user_message: Raw text typed by the user.
        user_id:      String identifier that scopes memories per user.

    Returns:
        A string combining retrieved memories with the original message.
    """
    # Mem0 converts user_message to a vector and finds the top-5
    # most similar stored facts using FAISS nearest-neighbour search.
    results = memory.search(
        query=user_message,
        filters={"user_id": user_id},
        limit=5,
    )

    # results is a dict; "results" key holds the list of memory objects.
    memories = results.get("results", [])

    if memories:
        # Format each fact as a bullet point for the agent.
        memory_lines = "\n".join(f"  - {m['memory']}" for m in memories)
        # Build the enriched prompt the agent actually receives.
        return (
            f"Relevant memories about the user:\n{memory_lines}\n\n"
            f"User message: {user_message}"
        )

    # No relevant memories found – pass the message unchanged.
    return user_message


# ============================================================
# SECTION 6 – Memory Helper: persist new user information
# ============================================================
# After the agent replies, we ask Mem0 to extract and store any
# facts from the user's message.  Mem0 uses its internal LLM to
# identify facts (e.g. "name is Jordan") and stores each one as
# a separate vector entry in the FAISS index.

def save_to_memory(user_message: str, user_id: str) -> None:
    """
    Persist information from the user's message into Mem0 / FAISS.

    Args:
        user_message: Raw text typed by the user.
        user_id:      Scopes the saved memories to a specific user.
    """
    # memory.add() accepts an OpenAI-style messages list.
    # user_id is passed as a direct kwarg (NOT inside filters={}).
    memory.add(
        messages=[{"role": "user", "content": user_message}],
        user_id=user_id,
    )


# ============================================================
# SECTION 7 – Interactive Chat Loop
# ============================================================
# The main loop ties everything together:
#
#   (a) Read the user's message from stdin.
#   (b) Handle special debug commands (show/clear memories).
#   (c) Search Mem0 and enrich the prompt with relevant facts.
#   (d) Send the enriched prompt to the agent (tools + memory).
#   (e) Persist the user's message to memory for future turns.
#
# Each turn the agent can:
#   • answer directly from knowledge
#   • call a tool (calculator, weather, age)
#   • reference remembered facts about the user

USER_ID = "challenge4_user"   # change this string to switch between user profiles

print("=" * 60)
print("  Strands Full Agent  (llama3.2:3b + 3 Tools + Mem0/FAISS)")
print("=" * 60)
print("Tools  : calculator | get_weather | calculate_age")
print("Memory : remembers facts across the whole conversation")
print()
print("Try these examples:")
print("  'My name is Alex and I was born in 1992'")
print("  'What is 144 * 37?'")
print("  'What is the weather in Tokyo?'")
print("  'How old am I?'   ← agent uses memory for your birth year")
print("  'What do you know about me?'")
print()
print("Special commands:")
print("  'show memories'  – list everything stored so far")
print("  'clear memories' – wipe the memory store")
print("  'exit'           – quit the agent")
print("=" * 60)
print()

while True:
    # ── (a) Read input ───────────────────────────────────────
    user_input = input("You: ").strip()

    # ── Exit condition ────────────────────────────────────────
    if user_input.lower() in ("exit", "quit", "bye"):
        print("Goodbye! Your memories are saved for next time.")
        break

    # ── Skip blank lines ──────────────────────────────────────
    if not user_input:
        continue

    # ── (b-i) Debug: show all stored memories ────────────────
    if user_input.lower() == "show memories":
        all_memories = memory.get_all(filters={"user_id": USER_ID})
        entries = all_memories.get("results", [])
        if entries:
            print("\n── Stored memories ──────────────────────────")
            for i, m in enumerate(entries, 1):
                print(f"  {i}. {m['memory']}")
            print("─────────────────────────────────────────────\n")
        else:
            print("  (no memories stored yet)\n")
        continue

    # ── (b-ii) Debug: clear all stored memories ───────────────
    if user_input.lower() == "clear memories":
        memory.delete_all(filters={"user_id": USER_ID})
        print("  All memories cleared.\n")
        continue

    # ── (c) Enrich the prompt with relevant memories ──────────
    # Mem0 searches FAISS for facts similar to the user's message
    # and prepends them so the agent knows who it is talking to.
    prompt = build_prompt_with_memory(user_input, USER_ID)

    # ── (d) Send enriched prompt to the agent ─────────────────
    # Strands handles the full reasoning loop:
    #   prompt → model → (optional tool call) → final answer
    # Tool calls and streaming output are managed automatically.
    print("Agent: ", end="", flush=True)
    agent(prompt)
    print()   # blank line for readability

    # ── (e) Persist user message to memory ───────────────────
    # Mem0 extracts and stores any new facts from what the user said.
    save_to_memory(user_input, USER_ID)
