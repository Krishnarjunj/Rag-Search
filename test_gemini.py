import os
import time

from dotenv import load_dotenv
from google import genai


PROMPT = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
MODEL_CANDIDATES = (
    os.environ.get("GEMINI_MODEL"),
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
)
MAX_ATTEMPTS_PER_MODEL = 3


def generate_response(client: genai.Client):
    last_error = None

    for model in MODEL_CANDIDATES:
        if not model:
            continue

        for attempt in range(1, MAX_ATTEMPTS_PER_MODEL + 1):
            try:
                return client.models.generate_content(model=model, contents=PROMPT)
            except Exception as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS_PER_MODEL:
                    time.sleep(attempt)

    raise RuntimeError(f"Gemini request failed after trying multiple models: {last_error}")


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    response = generate_response(client)

    print(response.text)

    usage_metadata = response.usage_metadata
    print(f"Prompt tokens: {usage_metadata.prompt_token_count}")
    print(f"Response tokens: {usage_metadata.candidates_token_count}")


if __name__ == "__main__":
    main()
