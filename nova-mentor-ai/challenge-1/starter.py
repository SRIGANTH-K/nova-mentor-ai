# ============================================================
# Challenge 1 - Builders Skill Sprint
# Simple AI Agent using Strands SDK + Ollama (llama3.2:3b)
# ============================================================

# --- Imports ---
# Agent is the core class that powers the AI agent loop
from strands import Agent

# OllamaModel tells Strands how to talk to your local Ollama server
from strands.models.ollama import OllamaModel

# --- Step 1: Configure the local model ---
# We point Strands at the Ollama server running on your machine.
# host    : Ollama always listens on port 11434 by default
# model_id: the model you pulled with "ollama pull llama3.2:3b"
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.7,   # 0 = very predictable, 1 = more creative
)

# --- Step 2: Create the agent ---
# The Agent wraps the model and manages the conversation loop.
# system_prompt gives the agent a personality / set of instructions.
agent = Agent(
    model=ollama_model,
    system_prompt=(
        "You are a helpful AI assistant. "
        "Answer questions clearly and concisely."
    ),
)

# --- Step 3: Chat loop ---
# We run a simple REPL so you can keep chatting until you type "exit".
print("=== Strands Agent (llama3.2:3b) ===")
print("Type your message and press Enter. Type 'exit' to quit.\n")

while True:
    # Read user input from the terminal
    user_input = input("You: ").strip()

    # Exit condition
    if user_input.lower() in ("exit", "quit", "bye"):
        print("Goodbye!")
        break

    # Skip empty lines
    if not user_input:
        continue

    # Send the message to the agent and print the response.
    # agent() runs the full reasoning loop: prompt → model → response.
    print("Agent: ", end="", flush=True)
    agent(user_input)   # Strands prints the streamed response automatically
    print()             # Blank line for readability
