from openai import OpenAI

from utils import get_openai_api_key


LANGUAGE_OPTIONS = {
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Japanese": "ja",
    "Korean": "ko",
    "Vietnamese": "vi",
    "English": "en",
}


def translate_text(text: str, target_language: str) -> str:
    """
    Translate text to the selected target language.
    Uses a compact OpenAI chat completion for simplicity.
    """
    client = OpenAI(api_key=get_openai_api_key())

    prompt = (
        f"Translate the following text to {target_language}. "
        "Return only the translated text with no extra commentary.\n\n"
        f"Text:\n{text}"
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a precise translation assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    return completion.choices[0].message.content.strip()
