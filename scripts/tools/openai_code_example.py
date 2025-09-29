import os
import sys

try:
    import openai
except Exception:
    print("The OpenAI Python package is not installed. Install with: pip install openai")
    raise


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in your environment (or create a .env file and load it).")
        sys.exit(2)

    # Allow overriding the model via env; sensible default is a code-capable model.
    model = os.getenv("OPENAI_MODEL", "gpt-4o-code")

    openai.api_key = api_key

    prompt = os.getenv("OPENAI_PROMPT", "Skriv en Python-funksjon som reverserer en streng og inkluder en kort doctest.")

    print(f"Using model: {model}")
    print("Prompt:\n", prompt)

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Du er en hjelpsom kodeassistent."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
        )
    except Exception as e:
        print("API call failed:", e)
        sys.exit(3)

    # The response shape may vary by SDK version; defensive access
    try:
        out = resp["choices"][0]["message"]["content"]
    except Exception:
        out = str(resp)

    print("\nResponse:\n")
    print(out)


if __name__ == "__main__":
    main()
