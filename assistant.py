import os
import sys
from openai import OpenAI, OpenAIError


def init_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Fant ingen API-nøkkel.")
        print("➡ Sett den med:")
        print('   PowerShell:  setx OPENAI_API_KEY "din-nøkkel"')
        print("   (Lukk PowerShell og åpne på nytt etterpå)")
        sys.exit(1)
    return OpenAI(api_key=api_key)


client = init_client()


def ask_codegpt(prompt: str):
    """Send spørsmål/kode til Code GPT og få svar"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # trygg modell
            messages=[
                {"role": "system", "content": "Du er en ekspertkodeassistent."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        return f"⚠️ API-feil: {e}"


if __name__ == "__main__":
    print("🚀 Code GPT terminal-klient startet. Skriv 'exit' for å avslutte.\n")
    while True:
        user_input = input("\n💬 Spørsmål til Code GPT: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Avslutter...")
            break
        answer = ask_codegpt(user_input)
        print("\n🤖 Svar fra Code GPT:\n")
        print(answer)
