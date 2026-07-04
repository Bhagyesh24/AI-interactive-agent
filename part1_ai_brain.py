import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3"


def get_ai_reply(user_text):
    """Send user text to local Ollama phi3 and return the AI's reply."""
    payload = {
        "model": MODEL,
        "prompt": user_text,
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure Ollama is running."
    except requests.exceptions.Timeout:
        return "Error: Ollama took too long to respond."
    except Exception as e:
        return f"Error: {e}"


def main():
    print("=" * 50)
    print("AI Brain Test - Ollama phi3")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        print("\nAI: ", end="", flush=True)
        reply = get_ai_reply(user_input)
        print(reply)


if __name__ == "__main__":
    main()
