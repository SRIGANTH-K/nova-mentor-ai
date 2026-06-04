# ============================================================
# Challenge 3 - Builders Skill Sprint
# Memory Agent using Strands SDK + Ollama (llama3.2:3b)
#
# New concepts introduced:
#   • Mem0   – a memory layer that stores / retrieves facts
#   • FAISS  – a vector store (used by Mem0 under the hood)
#              for fast similarity-based memory search
#
# How it works end-to-end:
#   1. User sends a message.
#   2. Mem0 searches existing memories for related facts.
#   3. Those facts are injected into the agent's context.
#   4. The agent answers with full awareness of past info.
#   5. After the agent replies, Mem0 saves any new facts
#      mentioned by the user into the FAISS vector store.
# ============================================================


# ============================================================
# SETUP – Install required packages (run once in your terminal)
# ============================================================
# pip install strands-agents strands-agents-tools
# pip install mem0ai
# pip install faiss-cpu          # CPU build; use faiss-gpu if you have CUDA
# pip install sentence-transformers  # local embeddings (no API key needed)
# pip install ollama
# ============================================================


# --- Standard-library imports ---
import os                           # used to silence noisy logs

# --- Suppress verbose logs from underlying libraries ---
# Set these BEFORE importing transformers / faiss so they take effect.
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # stops a HuggingFace warning

# --- Strands imports ---
from strands import Agent                        # core agent loop
from strands.models.ollama import OllamaModel   # local Ollama adapter

# --- Mem0 import ---
# Memory is the main class; it wraps the vector store and exposes
# simple add() / search() / get_all() methods.
from mem0 import Memory


# ============================================================
# SECTION 1 – Mem0 Configuration
# ============================================================
# Mem0 supports many backends (Qdrant, Pinecone, Redis, …).
# Here we use FAISS because it runs 100 % locally with no
# server required.
#
# We also use a local HuggingFace sentence-transformer model
# for embeddings so no OpenAI / Ollama key is needed for memory.

MEM0_CONFIG = {
    # ── Vector store ──────────────────────────────────────────
    "vector_store": {
        "provider": "faiss",          # use FAISS as the storage backend
        "config": {
            "embedding_model_dims": 384,          # must match the embedder below
            "path": "./challenge3_memory_db",     # folder where FAISS saves files
        },
    },
    # ── Embedder ──────────────────────────────────────────────
    # sentence-transformers/all-MiniLM-L6-v2 is a fast, lightweight
    # model (≈ 80 MB) that produces 384-dimensional vectors.
    # It is downloaded automatically the first time you run this.
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },
    # ── LLM used by Mem0 internally ───────────────────────────
    # Mem0 uses an LLM to extract structured facts from raw text
    # before storing them.  We reuse our local Ollama model.
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
            "ollama_base_url": "http://localhost:11434",
            "temperature": 0,         # deterministic extraction is better for memory
            "max_tokens": 2000,
        },
    },
}

# Initialise the Mem0 Memory object with the config above.
# The first call may take a minute while it downloads the embedding model.
print("Initialising memory layer (first run downloads ~80 MB embedding model) …")
memory = Memory.from_config(MEM0_CONFIG)
print("Memory layer ready.\n")


# ============================================================
# SECTION 2 – Ollama Model Configuration
# ============================================================
# Same pattern as Challenges 1 & 2: point Strands at the local
# Ollama instance running llama3.2:3b.
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,
)


# ============================================================
# SECTION 3 – Agent Setup
# ============================================================
# The system prompt is intentionally generic here because we
# dynamically inject retrieved memories into each message
# (see Section 5), rather than hard-coding them up front.
agent = Agent(
    model=ollama_model,
    system_prompt=(
        "You are a helpful personal assistant with a long memory.\n"
        "When the conversation includes a 'Relevant memories' block, "
        "treat those facts as true and use them to personalise your answer.\n"
        "Be concise and friendly."
    ),
)


# ============================================================
# SECTION 4 – Helper: build a memory-enriched prompt
# ============================================================
# Before sending the user's message to the agent we:
#   1. Search Mem0 for memories related to the message.
#   2. Prepend those memories to the prompt so the agent
#      can reference them.

def build_prompt_with_memory(user_message: str, user_id: str) -> str:
    """
    Search Mem0 for facts related to `user_message` and prepend
    them to the message so the agent has full context.

    Args:
        user_message: The raw text typed by the user.
        user_id:      A string identifier for the user (keeps
                      memories scoped per user).

    Returns:
        A string combining any retrieved memories with the
        original user message.
    """
    # Search the FAISS vector store for the top-5 most relevant memories.
    # Mem0 converts `user_message` to a vector and does a similarity search.
    # NOTE: newer mem0ai versions require user_id inside filters={} not as a
    #       top-level kwarg.
    results = memory.search(query=user_message, filters={"user_id": user_id}, limit=5)

    # `results` is a dict with a "results" key containing a list of memory objects.
    memories = results.get("results", [])

    if memories:
        # Format each retrieved memory as a bullet point.
        memory_lines = "\n".join(
            f"  - {m['memory']}" for m in memories
        )
        # Build the enriched prompt that the agent will actually see.
        enriched_prompt = (
            f"Relevant memories about the user:\n{memory_lines}\n\n"
            f"User message: {user_message}"
        )
    else:
        # No relevant memories found – just pass the message as-is.
        enriched_prompt = user_message

    return enriched_prompt


# ============================================================
# SECTION 5 – Helper: save new information to memory
# ============================================================
# After the agent replies, we store the user's message in Mem0.
# Mem0's LLM internally extracts discrete facts (e.g. "name is
# Thamarai") and saves each one as a searchable vector entry.

def save_to_memory(user_message: str, user_id: str) -> None:
    """
    Persist information from the user's message into Mem0 / FAISS.

    Args:
        user_message: The raw text typed by the user.
        user_id:      Scopes the memory to a specific user.
    """
    # memory.add() accepts a list of message dicts (OpenAI-style).
    # Mem0 will parse the content and store extracted facts.
    # NOTE: add() keeps user_id as a direct kwarg (unlike search/get_all/delete_all
    #       which moved to filters={} in newer mem0ai releases).
    memory.add(
        messages=[{"role": "user", "content": user_message}],
        user_id=user_id,
    )


# ============================================================
# SECTION 6 – Interactive Chat Loop
# ============================================================
# Each iteration:
#   (a) Read user input.
#   (b) Search memory and build an enriched prompt.
#   (c) Send prompt to the agent (Strands streams the reply).
#   (d) Save the user's message to memory for future turns.

USER_ID = "challenge3_user"   # change this to support multiple users

print("=" * 60)
print("  Strands Memory Agent  (llama3.2:3b + Mem0 + FAISS)")
print("=" * 60)
print("I remember things you tell me across the whole conversation.")
print("Try: 'My name is Thamarai' then later 'What is my name?'")
print("Type 'show memories' to see everything stored so far.")
print("Type 'clear memories' to wipe the memory store.")
print("Type 'exit' to quit.\n")

while True:
    # ── (a) Read input ───────────────────────────────────────
    user_input = input("You: ").strip()

    # Exit condition
    if user_input.lower() in ("exit", "quit", "bye"):
        print("Goodbye! Your memories are saved for next time.")
        break

    # Skip blank lines
    if not user_input:
        continue

    # ── Debug command: list all stored memories ───────────────
    if user_input.lower() == "show memories":
        # get_all also uses filters in newer mem0ai versions
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

    # ── Debug command: clear all stored memories ─────────────
    if user_input.lower() == "clear memories":
        # delete_all also uses filters in newer mem0ai versions
        memory.delete_all(filters={"user_id": USER_ID})
        print("  All memories cleared.\n")
        continue

    # ── (b) Build memory-enriched prompt ─────────────────────
    prompt = build_prompt_with_memory(user_input, USER_ID)

    # ── (c) Send to agent ────────────────────────────────────
    print("Agent: ", end="", flush=True)
    agent(prompt)           # Strands streams the response to stdout
    print()                 # Blank line for readability

    # ── (d) Persist user message to memory ───────────────────
    save_to_memory(user_input, USER_ID)
