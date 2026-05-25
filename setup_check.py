from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


CHECK = "✓"
CROSS = "✗"


def _status(ok: bool, name: str, detail: str) -> None:
    symbol = CHECK if ok else CROSS
    print(f"{symbol} {name}: {detail}")


def _http_reachable(name: str, url: str, headers: dict, valid_statuses: set[int]) -> bool:
    try:
        response = requests.get(url, headers=headers, timeout=10)
        ok = response.status_code in valid_statuses
        detail = f"HTTP {response.status_code}"
        if response.status_code in {401, 403}:
            detail += " (authentication failed)"
        _status(ok, name, detail)
        return ok
    except Exception as exc:
        _status(False, name, f"unreachable ({exc})")
        return False


def check_deepgram() -> bool:
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        _status(False, "Deepgram key", "DEEPGRAM_API_KEY is missing")
        return False
    _status(True, "Deepgram key", "set")
    return _http_reachable(
        "Deepgram API",
        "https://api.deepgram.com/v1/projects",
        {"Authorization": f"Token {api_key}"},
        {200},
    )


def check_openai() -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _status(False, "OpenAI key", "OPENAI_API_KEY is missing")
        return False
    _status(True, "OpenAI key", "set")
    return _http_reachable(
        "OpenAI API",
        "https://api.openai.com/v1/models",
        {"Authorization": f"Bearer {api_key}"},
        {200},
    )


def check_sarvam() -> bool:
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        _status(False, "Sarvam key", "SARVAM_API_KEY is missing")
        return False
    _status(True, "Sarvam key", "set")
    try:
        response = requests.options(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": api_key},
            timeout=10,
        )
        ok = response.status_code < 500 and response.status_code not in {401, 403}
        detail = f"HTTP {response.status_code}"
        if response.status_code in {401, 403}:
            detail += " (authentication failed)"
        _status(ok, "Sarvam API", detail)
        return ok
    except Exception as exc:
        _status(False, "Sarvam API", f"unreachable ({exc})")
        return False


def check_google() -> bool:
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        _status(False, "Google credentials", "GOOGLE_APPLICATION_CREDENTIALS is missing")
        return False

    path = Path(credentials_path).expanduser()
    if not path.exists():
        _status(False, "Google credentials", f"file not found: {path}")
        return False
    _status(True, "Google credentials", f"found {path}")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
    try:
        from google.cloud import speech

        speech.SpeechClient()
        _status(True, "Google Speech client", "initialized")
        return True
    except Exception as exc:
        _status(False, "Google Speech client", f"failed to initialize ({exc})")
        return False


def main() -> int:
    load_dotenv()
    print("ASR API setup check\n")
    checks = [
        check_deepgram(),
        check_openai(),
        check_sarvam(),
        check_google(),
    ]
    if not all(checks):
        print("\nOne or more checks failed. Fix the items above before running the benchmark.")
        return 1
    print("\nAll API credentials are present and reachable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
