# ============================================================
# Challenge 2 - Builders Skill Sprint
# Tools Agent using Strands SDK + Ollama (llama3.2:3b)
#
# Three custom tools are registered with the agent:
#   1. calculator     – evaluates basic math expressions
#   2. get_weather    – returns a mock weather report for any city
#   3. calculate_age  – computes a person's age from their birth year
# ============================================================

# --- Imports ---
from datetime import datetime          # used by the age calculator tool
from strands import Agent, tool        # tool decorator registers functions as agent tools
from strands.models.ollama import OllamaModel


# ============================================================
# SECTION 1 – Tool Definitions
# ============================================================
# The @tool decorator turns a plain Python function into a
# Strands tool.  The agent reads the function name, type hints,
# and docstring to decide *when* and *how* to call each tool.
# Always include a clear docstring – it is the agent's manual.

# ------------------------------------------------------------
# Tool 1: Calculator
# ------------------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic mathematical expression and return the result.

    Use this tool when the user asks you to perform arithmetic such as
    addition, subtraction, multiplication, division, or exponentiation.

    Args:
        expression: A math expression string, e.g. "2 + 3 * 4" or "10 / 2"

    Returns:
        The numeric result as a string, or an error message if the
        expression is invalid or unsafe.
    """
    # eval() can be dangerous with arbitrary input, so we restrict the
    # allowed names to only safe math builtins.
    safe_names = {
        "__builtins__": {},   # block all built-ins
        "abs": abs, "round": round, "pow": pow,
        "min": min, "max": max,
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
# ------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """
    Return the current weather report for a given city.

    Use this tool whenever the user asks about the weather, temperature,
    or forecast for a location.

    Args:
        city: The name of the city to look up, e.g. "London" or "Tokyo"

    Returns:
        A short weather summary string for that city.
    """
    # In a real application you would call a weather API (e.g. OpenWeatherMap).
    # Here we return plausible mock data so the agent can demonstrate tool use
    # without requiring an API key.
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

    # Generic fallback for cities not in the mock database
    return (
        f"Weather in {city.title()}: 20°C, partly cloudy, humidity 65%. "
        "(Mock data – real API not configured)"
    )


# ------------------------------------------------------------
# Tool 3: Age Calculator
# ------------------------------------------------------------
@tool
def calculate_age(birth_year: int) -> str:
    """
    Calculate a person's current age based on their birth year.

    Use this tool when the user wants to know how old someone is,
    or how many years ago a given year was.

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
# SECTION 2 – Model Configuration
# ============================================================
# Same Ollama setup as Challenge 1: local llama3.2:3b instance.
ollama_model = OllamaModel(
    host="http://localhost:11434",
    model_id="llama3.2:3b",
    temperature=0.3,   # lower temperature → more consistent tool choices
)


# ============================================================
# SECTION 3 – Agent Setup
# ============================================================
# Passing the tool functions in the `tools` list makes them
# available to the agent's reasoning loop.  The system prompt
# tells the agent when to prefer tools over answering directly.
agent = Agent(
    model=ollama_model,
    tools=[calculator, get_weather, calculate_age],
    system_prompt=(
        "You are a helpful assistant with access to three tools:\n"
        "  • calculator     – for any math calculation\n"
        "  • get_weather    – for weather information about a city\n"
        "  • calculate_age  – to find someone's age from their birth year\n\n"
        "Always use the appropriate tool when the user's question matches "
        "one of these capabilities. Provide clear, concise answers."
    ),
)


# ============================================================
# SECTION 4 – Interactive Chat Loop
# ============================================================
# Identical structure to Challenge 1: keep chatting until the
# user types 'exit', 'quit', or 'bye'.
print("=" * 55)
print("  Strands Tools Agent  (llama3.2:3b + 3 tools)")
print("=" * 55)
print("Available tools:")
print("  • Calculator   – e.g. 'What is 123 * 456?'")
print("  • Weather      – e.g. 'What is the weather in Tokyo?'")
print("  • Age Calc     – e.g. 'How old is someone born in 1990?'")
print("\nType 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("exit", "quit", "bye"):
        print("Goodbye!")
        break

    if not user_input:
        continue

    print("Agent: ", end="", flush=True)
    agent(user_input)   # Strands handles streaming + tool calls automatically
    print()
