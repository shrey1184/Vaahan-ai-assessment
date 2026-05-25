from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from textwrap import shorten
from typing import Any

from dotenv import load_dotenv
from tabulate import tabulate
from tqdm import tqdm


MODEL_MODULES = {
    "deepgram": "models.deepgram_runner",
    "whisper": "models.whisper_runner",
    "sarvam": "models.sarvam_runner",
}


def _default_metadata_path() -> Path:
    candidates = [
        Path("audio_samples") / "metadata.json",
        Path("audiosample") / "metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _audio_dir_for(metadata_path: Path) -> Path:
    return metadata_path.parent


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        return str(value)


def _failure_result(error: Exception) -> dict:
    return {
        "error": str(error),
        "transcript": None,
        "confidence": None,
        "latency_ms": None,
        "word_timings": [],
        "detected_language": None,
        "raw_response": {},
    }


def _load_metadata(metadata_path: Path) -> list[dict]:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}. "
            "Expected audio_samples/metadata.json or audiosample/metadata.json."
        )
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        data = json.load(metadata_file)
    recordings = data.get("recordings")
    if not isinstance(recordings, list):
        raise ValueError("metadata.json must contain a top-level 'recordings' list")
    return recordings


def _selected_models(model_arg: str) -> list[str]:
    if model_arg == "all":
        return list(MODEL_MODULES)
    return [model_arg]


def _result_path(output: str | None) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    if output:
        path = Path(output)
        return path if path.is_absolute() else results_dir / path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / f"results_{timestamp}.json"


def _print_transcript_table(results: list[dict], models: list[str]) -> None:
    rows = []
    for index, result in enumerate(results, start=1):
        row = [
            index,
            result["file"],
            result.get("condition"),
        ]
        for model in models:
            model_result = result["models"].get(model, {})
            transcript = model_result.get("transcript")
            if model_result.get("error"):
                transcript = f"ERROR: {model_result['error']}"
            row.append(shorten(str(transcript or ""), width=22, placeholder="..."))
        rows.append(row)
    headers = ["#", "File", "Condition", *[model.title() for model in models]]
    print("\nTranscripts")
    print(tabulate(rows, headers=headers, tablefmt="github"))


def _print_stats(results: list[dict], models: list[str]) -> None:
    rows = []
    total = len(results)
    for model in models:
        model_results = [result["models"].get(model, {}) for result in results]
        failures = sum(1 for item in model_results if item.get("error"))
        latencies = [
            item.get("latency_ms")
            for item in model_results
            if isinstance(item.get("latency_ms"), (int, float))
        ]
        confidences = [
            item.get("confidence")
            for item in model_results
            if isinstance(item.get("confidence"), (int, float))
        ]
        avg_latency = f"{int(statistics.mean(latencies))}ms" if latencies else "N/A"
        avg_confidence = f"{statistics.mean(confidences):.2f}" if confidences else "N/A"
        rows.append([model.title(), avg_latency, f"{failures}/{total}", avg_confidence])
    print("\nPer-model stats")
    print(tabulate(rows, headers=["Model", "Avg latency", "Failures", "Avg confidence"], tablefmt="github"))


def run_benchmark(args: argparse.Namespace) -> Path:
    load_dotenv()
    metadata_path = Path(args.metadata) if args.metadata else _default_metadata_path()
    metadata_path = metadata_path.expanduser().resolve()
    audio_dir = _audio_dir_for(metadata_path)
    recordings = _load_metadata(metadata_path)
    if args.dry_run:
        recordings = recordings[:3]

    models = _selected_models(args.model)
    imported_models = {
        model: importlib.import_module(module_name)
        for model, module_name in MODEL_MODULES.items()
        if model in models
    }

    output = {
        "run_info": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "models": models,
            "total_files": len(recordings),
            "dry_run": args.dry_run,
        },
        "results": [],
    }

    progress = tqdm(recordings, total=len(recordings), unit="file")
    for index, recording in enumerate(progress, start=1):
        filename = recording.get("file")
        progress.set_description(f"Processing file {index}/{len(recordings)}")
        audio_path = (audio_dir / filename).resolve()
        result = {
            "file": filename,
            "locality": recording.get("locality"),
            "ground_truth": recording.get("sentence"),
            "condition": recording.get("condition"),
            "language": recording.get("language"),
            "sample_rate": recording.get("sample_rate"),
            "models": {},
        }

        for model in models:
            try:
                model_result = imported_models[model].transcribe(str(audio_path), recording)
                result["models"][model] = _json_safe(model_result)
            except Exception as exc:
                result["models"][model] = _failure_result(exc)
            time.sleep(0.5)

        output["results"].append(result)

    result_path = _result_path(args.output)
    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump(output, result_file, ensure_ascii=False, indent=2, default=str)

    _print_transcript_table(output["results"], models)
    _print_stats(output["results"], models)
    print(f"\nSaved results to {result_path}")
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run API ASR benchmark over Bangalore locality audio samples.")
    parser.add_argument(
        "--model",
        choices=[*MODEL_MODULES.keys(), "all"],
        default="all",
        help="Model to run. Defaults to all.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run only the first 3 recordings.")
    parser.add_argument("--output", help="Custom output filename. Relative paths are placed under results/.")
    parser.add_argument(
        "--metadata",
        help="Optional path to metadata.json. Defaults to audio_samples/metadata.json, then audiosample/metadata.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())

