"""Google Cloud Speech-to-Text v1 -- industry-standard API with mature
Hindi support and explicit Indian English accent model. Strong baseline
for comparison since many teams default to Google for Indian speech.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from pydub import AudioSegment


def _duration_to_seconds(duration: Any) -> float | None:
    if duration is None:
        return None
    if hasattr(duration, "total_seconds"):
        return float(duration.total_seconds())
    seconds = getattr(duration, "seconds", 0)
    nanos = getattr(duration, "nanos", 0)
    return float(seconds) + float(nanos) / 1_000_000_000


def _protobuf_to_dict(message: Any) -> dict:
    from google.protobuf.json_format import MessageToDict

    return MessageToDict(message._pb, preserving_proto_field_name=True)


def _with_retries(callable_obj):
    delays = [1, 2, 4]
    last_error: Exception | None = None
    for attempt in range(len(delays) + 1):
        try:
            return callable_obj()
        except Exception as exc:
            last_error = exc
            if attempt >= len(delays):
                break
            time.sleep(delays[attempt])
    raise last_error  # type: ignore[misc]


def transcribe(audio_path: str, metadata: dict) -> dict:
    """
    Args:
        audio_path: absolute path to .wav file
        metadata: dict for this file from metadata.json
                  (has: locality, sentence, condition, language, sample_rate)
    Returns:
        {
            "transcript": str,
            "confidence": float | None,
            "latency_ms": int,
            "word_timings": [
                {"word": str, "start": float, "end": float, "confidence": float|None}
            ],
            "detected_language": str | None,
            "raw_response": dict
        }
    """
    from google.cloud import speech

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    source_path = Path(audio_path)
    client = speech.SpeechClient()

    with tempfile.TemporaryDirectory() as temp_dir:
        converted_path = Path(temp_dir) / f"{source_path.stem}_google_linear16.wav"
        audio = AudioSegment.from_file(source_path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export(converted_path, format="wav")
        audio_bytes = converted_path.read_bytes()

    recognition_audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="hi-IN",
        alternative_language_codes=["en-IN"],
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
        model="latest_long",
    )

    def _call():
        return client.recognize(config=config, audio=recognition_audio)

    start = time.time()
    response = _with_retries(_call)
    latency_ms = int((time.time() - start) * 1000)

    raw_response = _protobuf_to_dict(response)
    alternatives = []
    if response.results:
        alternatives = response.results[0].alternatives
    alternative = alternatives[0] if alternatives else None
    words = getattr(alternative, "words", []) if alternative else []
    word_timings = [
        {
            "word": word.word,
            "start": _duration_to_seconds(word.start_time),
            "end": _duration_to_seconds(word.end_time),
            "confidence": None,
        }
        for word in words
    ]

    return {
        "transcript": alternative.transcript if alternative else "",
        "confidence": alternative.confidence if alternative else None,
        "latency_ms": latency_ms,
        "word_timings": word_timings,
        "detected_language": None,
        "raw_response": raw_response,
    }

