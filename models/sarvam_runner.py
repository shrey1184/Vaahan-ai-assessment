"""
Sarvam AI Saarika-v2.5 — built specifically for Indian languages and accents.
Native Hindi + Hinglish support, optimized for phone-call quality audio.
The most contextually relevant model for this blue-collar hiring use case.
"""

import os, time, requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY")

def transcribe(audio_path: str, metadata: dict) -> dict:
    for attempt in range(3):
        try:
            start = time.time()
            with open(audio_path, "rb") as f:
                response = requests.post(
                    "https://api.sarvam.ai/speech-to-text",
                    headers={"api-subscription-key": API_KEY},
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data={"model": "saarika:v2.5", "language_code": "hi-IN"},
                    timeout=30
                )
            latency = int((time.time() - start) * 1000)
            response.raise_for_status()
            data = response.json()

            return {
                "transcript": data.get("transcript", ""),
                "confidence": None,
                "latency_ms": latency,
                "word_timings": [],
                "detected_language": data.get("language_code"),
                "raw_response": data
            }

        except Exception as e:
            if attempt == 2:
                return {
                    "transcript": None,
                    "confidence": None,
                    "latency_ms": None,
                    "word_timings": [],
                    "detected_language": None,
                    "error": str(e),
                    "raw_response": {}
                }
            time.sleep(2 ** attempt)
