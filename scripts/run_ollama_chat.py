#!/usr/bin/env python3
"""Example: single-model chat via Ollama."""

import sys
from pathlib import Path

# Allow importing from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ollama_client import chat

DEFAULT_MODEL = "llava"


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"Chat with model: {model} (Ctrl+C to exit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            response = chat(model=model, messages=[{"role": "user", "content": user_input}])
            text = response.get("message", {}).get("content", "")
            print(f"\n{model}: {text}\n")
        except KeyboardInterrupt:
            print("\nBye.")
            break


if __name__ == "__main__":
    main()
