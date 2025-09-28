"""Interactive OpenAI coding agent (safe, local CLI).

Usage:
  - Set OPENAI_API_KEY in environment (or .env) and optionally OPENAI_MODEL.
  - Run: python scripts/tools/openai_agent.py
  - Type prompts. Empty line exits. Use '::write <path>' to save the last response to a file.
"""
from __future__ import annotations
import os
import sys
import argparse

try:
    import openai
except Exception:
    print("Install the openai package first: pip install openai")
    raise


def get_model(cli_model: str | None) -> str:
    return cli_model or os.getenv("OPENAI_MODEL") or "gpt-4o-code"


def main():
    parser = argparse.ArgumentParser(description="Interactive OpenAI coding agent")
    parser.add_argument("--model", help="Model to use (overrides OPENAI_MODEL env)")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment. Create a .env or set the variable and retry.")
        sys.exit(2)

    model = get_model(args.model)
    openai.api_key = api_key

    print(f"OpenAI coding agent — model={model}")
    print("Type your prompt. Empty input exits. Use '::write <path>' to save last response to a file.")

    last_response = None
    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            print("Exiting.")
            break

        # handle local commands
        if prompt.startswith("::write "):
            if last_response is None:
                print("No response to write yet.")
                continue
            path = prompt[len("::write "):].strip()
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(last_response)
                print(f"Wrote response to {path}")
            except Exception as e:
                print("Failed to write:", e)
            continue

        # build chat request
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant. Provide clear, concise code examples and explanations."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1200,
            )
        except Exception as e:
            print("API failure:", e)
            continue

        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            content = str(resp)

        last_response = content
        print("\n--- Response ---\n")
        print(content)
        print("\n----------------\n")


if __name__ == "__main__":
    main()
