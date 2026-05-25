"""
Deepgram Nova-2 — the team's current production ASR baseline.
Strong on English and Indian English, with a dedicated Hindi model.
API-based so latency reflects real-world phone call conditions.
"""

import os, time, requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("DEEPGRAM_API_KEY")

def transcribe(audio_path: str, metadata: dict) -> dict:
    lang = "en-IN" if metadata.get("language") == "Hinglish" else "hi"
    url = f"https://api.deepgram.com/v1/listen?model=nova-2&language={lang}&punctuate=true&smart_format=true"

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    for attempt in range(3):
        try:
            start = time.time()
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Token {API_KEY}",
                    "Content-Type": "audio/wav"
                },
                data=audio_data,
                timeout=30
            )
            latency = int((time.time() - start) * 1000)
            response.raise_for_status()
            data = response.json()

            alt = data["results"]["channels"][0]["alternatives"][0]
            words = [
                {
                    "word": w.get("word"),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "confidence": w.get("confidence")
                }
                for w in alt.get("words", [])
            ]

            return {
                "transcript": alt.get("transcript", ""),
                "confidence": alt.get("confidence"),
                "latency_ms": latency,
                "word_timings": words,
                "detected_language": None,
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
