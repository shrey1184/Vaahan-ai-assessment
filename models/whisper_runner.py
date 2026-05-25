"""
Whisper (faster-whisper, CPU) — open-source multilingual model.
Running locally removes API cost entirely. Uses CTranslate2 backend
which is 4x faster than original Whisper on CPU.
Model: medium — good balance of accuracy vs speed on CPU.
"""

import time
from faster_whisper import WhisperModel

_model = None

def _get_model():
    global _model
    if _model is None:
        print("Loading Whisper medium model (first time only)...")
        _model = WhisperModel("medium", device="cpu", compute_type="int8")
    return _model

def transcribe(audio_path: str, metadata: dict) -> dict:
    try:
        model = _get_model()
        start = time.time()
        segments, info = model.transcribe(
            audio_path,
            language="hi",
            task="transcribe",
            beam_size=5
        )
        transcript = " ".join(s.text for s in segments)
        latency = int((time.time() - start) * 1000)

        return {
            "transcript": transcript.strip(),
            "confidence": None,
            "latency_ms": latency,
            "word_timings": [],
            "detected_language": info.language,
            "raw_response": {"language_probability": info.language_probability}
        }

    except Exception as e:
        return {
            "transcript": None,
            "confidence": None,
            "latency_ms": None,
            "word_timings": [],
            "detected_language": None,
            "error": str(e),
            "raw_response": {}
        }
