from __future__ import annotations

import json
import math
import os
import re
import statistics
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from textwrap import shorten
from typing import Any

from jiwer import cer, wer
from rapidfuzz import fuzz
from tabulate import tabulate


CONDITIONS = ["clean", "noisy", "phone", "whisper", "fast"]
MODEL_COLORS = {
    "deepgram": "tab:blue",
    "whisper": "tab:orange",
    "sarvam": "tab:green",
    "google": "tab:red",
}


def normalize(text: Any) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text).lower().strip())
    cleaned = []
    for char in text:
        if unicodedata.category(char).startswith("P"):
            cleaned.append(" ")
        else:
            cleaned.append(char)
    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


def entity_accuracy(locality: str, transcript: str) -> dict:
    locality_norm = normalize(locality)
    transcript_norm = normalize(transcript)
    exact = bool(locality_norm and locality_norm in transcript_norm)
    fuzzy_score = fuzz.partial_ratio(locality_norm, transcript_norm) if locality_norm else 0.0
    fuzzy_match = fuzzy_score >= 80
    return {
        "exact_match": exact,
        "fuzzy_match": fuzzy_match,
        "fuzzy_score": float(fuzzy_score),
    }


def has_devanagari(text: Any) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", str(text or "")))


def has_latin(text: Any) -> bool:
    return bool(re.search(r"[a-zA-Z]", str(text or "")))


def latest_results_file(results_dir: Path = Path("results")) -> Path:
    candidates = sorted(results_dir.glob("results_*.json"))
    if not candidates:
        raise FileNotFoundError("No results/results_*.json files found")
    return candidates[-1]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def pct_string(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.0f}%"


def metric_string(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def latency_string(value: float | None) -> str:
    return "N/A" if value is None else f"{int(round(value))}ms"


def safe_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def compute_metrics(data: dict) -> dict:
    models = data.get("run_info", {}).get("models") or sorted(
        {
            model
            for item in data.get("results", [])
            for model in (item.get("models") or {}).keys()
        }
    )
    per_file = []
    by_model: dict[str, list[dict]] = defaultdict(list)
    by_condition: dict[str, dict[str, list[dict]]] = {
        condition: {model: [] for model in models} for condition in CONDITIONS
    }
    locality_scores: dict[str, list[float]] = defaultdict(list)
    code_switch_failures = []

    for item in data.get("results", []):
        ground_truth = item.get("ground_truth") or item.get("sentence") or ""
        ground_truth_norm = normalize(ground_truth)
        condition = item.get("condition") or "unknown"
        locality = item.get("locality") or ""
        file_record = {
            "file": item.get("file"),
            "locality": locality,
            "condition": condition,
            "ground_truth": ground_truth,
            "ground_truth_normalized": ground_truth_norm,
            "models": {},
        }

        for model in models:
            model_result = (item.get("models") or {}).get(model, {})
            transcript = model_result.get("transcript")
            failed = transcript is None or bool(model_result.get("error"))
            transcript_norm = normalize(transcript)
            latency_ms = safe_float(model_result.get("latency_ms"))

            if failed:
                metrics = {
                    "failed": True,
                    "error": model_result.get("error") or "missing transcript",
                    "transcript": transcript,
                    "transcript_normalized": transcript_norm,
                    "wer": None,
                    "cer": None,
                    "entity_exact": False,
                    "entity_fuzzy": False,
                    "entity_fuzzy_score": 0.0,
                    "latency_ms": latency_ms,
                }
            else:
                entity = entity_accuracy(locality, transcript_norm)
                metrics = {
                    "failed": False,
                    "error": None,
                    "transcript": transcript,
                    "transcript_normalized": transcript_norm,
                    "wer": float(wer(ground_truth_norm, transcript_norm)),
                    "cer": float(cer(ground_truth_norm, transcript_norm)),
                    "entity_exact": entity["exact_match"],
                    "entity_fuzzy": entity["fuzzy_match"],
                    "entity_fuzzy_score": entity["fuzzy_score"],
                    "latency_ms": latency_ms,
                }
                locality_scores[locality].append(metrics["wer"])

                reference_mixed = has_devanagari(ground_truth) and has_latin(ground_truth)
                transcript_one_script = has_devanagari(transcript) != has_latin(transcript)
                if reference_mixed and transcript_one_script:
                    code_switch_failures.append(
                        {
                            "model": model,
                            "file": item.get("file"),
                            "condition": condition,
                            "ground_truth": ground_truth,
                            "transcript": transcript,
                        }
                    )

            file_record["models"][model] = metrics
            by_model[model].append(metrics)
            if condition in by_condition:
                by_condition[condition][model].append(metrics)

        per_file.append(file_record)

    aggregate = aggregate_model_metrics(by_model)
    condition_breakdown = aggregate_condition_metrics(by_condition, models)
    worst_files = worst_performing_files(per_file, models)
    locality_difficulty = [
        {
            "locality": locality,
            "avg_wer": mean_or_none(scores),
            "count": len(scores),
        }
        for locality, scores in locality_scores.items()
    ]
    locality_difficulty.sort(key=lambda item: item["avg_wer"] or -1, reverse=True)
    qualitative = qualitative_examples(per_file, models)
    recommendation = build_recommendation(aggregate)

    return {
        "run_info": data.get("run_info", {}),
        "models": models,
        "per_file": per_file,
        "aggregate": aggregate,
        "condition_breakdown": condition_breakdown,
        "worst_files": worst_files,
        "locality_difficulty": locality_difficulty,
        "code_switch_failures": code_switch_failures,
        "qualitative_examples": qualitative,
        "recommendation": recommendation,
    }


def aggregate_model_metrics(by_model: dict[str, list[dict]]) -> dict:
    aggregate = {}
    for model, items in by_model.items():
        valid = [item for item in items if not item.get("failed")]
        latencies = [item["latency_ms"] for item in items if item.get("latency_ms") is not None]
        aggregate[model] = {
            "avg_wer": mean_or_none([item["wer"] for item in valid if item.get("wer") is not None]),
            "avg_cer": mean_or_none([item["cer"] for item in valid if item.get("cer") is not None]),
            "entity_exact": mean_or_none([1.0 if item.get("entity_exact") else 0.0 for item in valid]),
            "entity_fuzzy": mean_or_none([1.0 if item.get("entity_fuzzy") else 0.0 for item in valid]),
            "avg_latency_ms": mean_or_none(latencies),
            "median_latency": statistics.median(latencies) if latencies else None,
            "p95_latency": percentile(latencies, 95),
            "failure_rate": (len(items) - len(valid)) / len(items) if items else None,
            "failures": len(items) - len(valid),
            "total": len(items),
        }
    return aggregate


def aggregate_condition_metrics(by_condition: dict[str, dict[str, list[dict]]], models: list[str]) -> dict:
    breakdown = {}
    for condition in CONDITIONS:
        breakdown[condition] = {}
        for model in models:
            items = [item for item in by_condition.get(condition, {}).get(model, []) if not item.get("failed")]
            breakdown[condition][model] = {
                "avg_wer": mean_or_none([item["wer"] for item in items if item.get("wer") is not None]),
                "entity_fuzzy": mean_or_none([1.0 if item.get("entity_fuzzy") else 0.0 for item in items]),
                "count": len(items),
            }
    return breakdown


def worst_performing_files(per_file: list[dict], models: list[str]) -> dict:
    worst = {}
    for model in models:
        rows = []
        for item in per_file:
            metrics = item["models"].get(model, {})
            if metrics.get("wer") is None:
                continue
            rows.append(
                {
                    "model": model,
                    "file": item["file"],
                    "condition": item["condition"],
                    "locality": item["locality"],
                    "wer": metrics["wer"],
                    "transcript": metrics["transcript"],
                    "ground_truth": item["ground_truth"],
                }
            )
        worst[model] = sorted(rows, key=lambda row: row["wer"], reverse=True)[:5]
    return worst


def qualitative_examples(per_file: list[dict], models: list[str]) -> dict:
    examples = {}
    for model in models:
        missed = None
        partial = None
        for item in sorted(
            per_file,
            key=lambda file_item: file_item["models"].get(model, {}).get("wer") or -1,
            reverse=True,
        ):
            metrics = item["models"].get(model, {})
            if metrics.get("failed"):
                continue
            fuzzy = metrics.get("entity_fuzzy_score", 0)
            example = {
                "ground_truth": item["ground_truth"],
                "transcript": metrics.get("transcript"),
                "locality_hit": bool(metrics.get("entity_fuzzy")),
                "fuzzy_score": fuzzy,
                "wer": metrics.get("wer"),
                "file": item["file"],
                "locality": item["locality"],
            }
            if missed is None and fuzzy < 40:
                missed = example
            if partial is None and 40 <= fuzzy < 80:
                partial = example
            if missed and partial:
                break
        examples[model] = {"missed": missed, "partial": partial}
    return examples


def build_recommendation(aggregate: dict) -> str:
    valid_entity = {
        model: metrics["entity_fuzzy"]
        for model, metrics in aggregate.items()
        if metrics.get("entity_fuzzy") is not None
    }
    valid_latency = {
        model: metrics["avg_latency_ms"]
        for model, metrics in aggregate.items()
        if metrics.get("avg_latency_ms") is not None
    }
    valid_wer = {
        model: metrics["avg_wer"]
        for model, metrics in aggregate.items()
        if metrics.get("avg_wer") is not None
    }
    if not valid_entity or not valid_latency or not valid_wer:
        return "RECOMMEND: collect successful transcripts for all models before choosing production ASR."

    best_entity = max(valid_entity, key=valid_entity.get)
    best_latency = min(valid_latency, key=valid_latency.get)
    best_wer = min(valid_wer, key=valid_wer.get)

    if best_entity == best_wer:
        return f"RECOMMEND: {best_entity} -- best accuracy AND entity recognition"
    if valid_latency[best_latency] < 1000 and valid_entity[best_entity] > 0.75:
        return f"RECOMMEND: {best_entity} for accuracy, {best_latency} for real-time use cases"
    return "RECOMMEND: Sarvam for production (India-specific), Whisper as offline fallback"


def save_charts(analysis: dict, charts_dir: Path = Path("charts")) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(charts_dir / ".mplconfig"))
    import matplotlib.pyplot as plt

    models = analysis["models"]
    aggregate = analysis["aggregate"]
    conditions = CONDITIONS

    plot_wer_by_model(plt, charts_dir, models, aggregate)
    plot_entity_accuracy(plt, charts_dir, models, aggregate)
    plot_wer_by_condition(plt, charts_dir, models, analysis["condition_breakdown"], conditions)
    plot_latency(plt, charts_dir, models, aggregate)
    plot_locality_difficulty(plt, charts_dir, analysis["locality_difficulty"])


def plot_wer_by_model(plt, charts_dir: Path, models: list[str], aggregate: dict) -> None:
    values = [aggregate[model].get("avg_wer") or 0 for model in models]
    colors = [MODEL_COLORS.get(model, "gray") for model in models]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([model.title() for model in models], values, color=colors)
    ax.set_title("Average WER by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Average WER")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(charts_dir / "wer_by_model.png", dpi=150)
    plt.close(fig)


def plot_entity_accuracy(plt, charts_dir: Path, models: list[str], aggregate: dict) -> None:
    x = list(range(len(models)))
    width = 0.35
    exact = [(aggregate[model].get("entity_exact") or 0) * 100 for model in models]
    fuzzy = [(aggregate[model].get("entity_fuzzy") or 0) * 100 for model in models]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([pos - width / 2 for pos in x], exact, width, label="Exact match")
    ax.bar([pos + width / 2 for pos in x], fuzzy, width, label="Fuzzy match")
    ax.set_title("Entity (Locality) Accuracy by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([model.title() for model in models])
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "entity_accuracy_by_model.png", dpi=150)
    plt.close(fig)


def plot_wer_by_condition(plt, charts_dir: Path, models: list[str], breakdown: dict, conditions: list[str]) -> None:
    x = list(range(len(conditions)))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(11, 6))
    for index, model in enumerate(models):
        offsets = [pos - 0.4 + width / 2 + index * width for pos in x]
        values = [breakdown[condition][model].get("avg_wer") or 0 for condition in conditions]
        ax.bar(offsets, values, width, label=model.title(), color=MODEL_COLORS.get(model))
    ax.set_title("WER by Audio Condition")
    ax.set_xlabel("Audio Condition")
    ax.set_ylabel("Average WER")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "wer_by_condition.png", dpi=150)
    plt.close(fig)


def plot_latency(plt, charts_dir: Path, models: list[str], aggregate: dict) -> None:
    values = [aggregate[model].get("avg_latency_ms") or 0 for model in models]
    p95_values = [aggregate[model].get("p95_latency") or value for model, value in zip(models, values)]
    errors = [max(p95 - value, 0) for p95, value in zip(p95_values, values)]
    colors = [MODEL_COLORS.get(model, "gray") for model in models]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([model.title() for model in models], values, yerr=errors, capsize=6, color=colors)
    ax.set_title("API Latency by Model (ms)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Average latency (ms)")
    ax.text(0.5, -0.18, "Whisper runs locally -- latency = compute time, not network", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(charts_dir / "latency_by_model.png", dpi=150)
    plt.close(fig)


def plot_locality_difficulty(plt, charts_dir: Path, locality_difficulty: list[dict]) -> None:
    sorted_items = list(reversed(locality_difficulty))
    localities = [item["locality"] for item in sorted_items]
    values = [item.get("avg_wer") or 0 for item in sorted_items]
    hardest = {item["locality"] for item in locality_difficulty[:5]}
    colors = ["tab:red" if locality in hardest else "tab:gray" for locality in localities]
    height = max(6, len(localities) * 0.35)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(localities, values, color=colors)
    ax.set_title("Locality Name Difficulty (avg WER across models)")
    ax.set_xlabel("Average WER")
    ax.set_ylabel("Locality")
    fig.tight_layout()
    fig.savefig(charts_dir / "locality_difficulty.png", dpi=150)
    plt.close(fig)


def print_summary(analysis: dict) -> None:
    models = analysis["models"]
    aggregate = analysis["aggregate"]

    print("\n========================================")
    print("ASR BENCHMARK SUMMARY")
    print("========================================")
    print("\nOVERALL METRICS")
    rows = []
    for model in models:
        metrics = aggregate[model]
        rows.append(
            [
                model.title(),
                metric_string(metrics.get("avg_wer")),
                metric_string(metrics.get("avg_cer")),
                pct_string(metrics.get("entity_exact")),
                pct_string(metrics.get("entity_fuzzy")),
                latency_string(metrics.get("avg_latency_ms")),
            ]
        )
    print(tabulate(rows, headers=["Model", "Avg WER", "Avg CER", "Entity Exact", "Entity Fuzzy", "Avg Latency"], tablefmt="github"))

    print("\nLATENCY / FAILURES")
    rows = []
    for model in models:
        metrics = aggregate[model]
        rows.append(
            [
                model.title(),
                latency_string(metrics.get("avg_latency_ms")),
                latency_string(metrics.get("median_latency")),
                latency_string(metrics.get("p95_latency")),
                pct_string(metrics.get("failure_rate")),
            ]
        )
    print(tabulate(rows, headers=["Model", "Avg", "Median", "P95", "Failure Rate"], tablefmt="github"))

    print("\nCONDITION BREAKDOWN (avg WER)")
    rows = []
    for condition in CONDITIONS:
        rows.append(
            [
                condition,
                *[
                    metric_string(analysis["condition_breakdown"][condition][model].get("avg_wer"))
                    for model in models
                ],
            ]
        )
    print(tabulate(rows, headers=["Condition", *[model.title() for model in models]], tablefmt="github"))

    print("\nCONDITION BREAKDOWN (entity fuzzy accuracy)")
    rows = []
    for condition in CONDITIONS:
        rows.append(
            [
                condition,
                *[
                    pct_string(analysis["condition_breakdown"][condition][model].get("entity_fuzzy"))
                    for model in models
                ],
            ]
        )
    print(tabulate(rows, headers=["Condition", *[model.title() for model in models]], tablefmt="github"))

    print("\nWORST PERFORMING FILES")
    rows = []
    for model in models:
        for item in analysis["worst_files"].get(model, []):
            rows.append(
                [
                    model,
                    item["file"],
                    item["condition"],
                    item["locality"],
                    metric_string(item["wer"]),
                    shorten(str(item["transcript"]), width=36, placeholder="..."),
                    shorten(str(item["ground_truth"]), width=36, placeholder="..."),
                ]
            )
    print(tabulate(rows, headers=["Model", "File", "Condition", "Locality", "WER", "Transcript", "Ground truth"], tablefmt="github"))

    print("\nTOP 5 HARDEST LOCALITIES")
    for index, item in enumerate(analysis["locality_difficulty"][:5], start=1):
        print(f"{index}. {item['locality']}  (avg WER: {metric_string(item.get('avg_wer'))})")

    print("\nCODE-SWITCH FAILURES")
    code_switch = analysis["code_switch_failures"]
    if code_switch:
        rows = [
            [
                item["model"],
                item["file"],
                item["condition"],
                shorten(item["ground_truth"], width=42, placeholder="..."),
                shorten(item["transcript"], width=42, placeholder="..."),
            ]
            for item in code_switch
        ]
        print(tabulate(rows, headers=["Model", "File", "Condition", "Ground truth", "Transcript"], tablefmt="github"))
    else:
        print("No script-level code-switch failures detected by the heuristic.")

    print("\nQUALITATIVE EXAMPLES")
    for model in models:
        print(f"\n{model.title()}")
        examples = analysis["qualitative_examples"].get(model, {})
        for label, example in [("Locality missed", examples.get("missed")), ("Partially right", examples.get("partial"))]:
            print(f"{label}:")
            if not example:
                print("  No matching example found.")
                continue
            print(f"  Ground truth : \"{example['ground_truth']}\"")
            print(f"  Transcript   : \"{example['transcript']}\"")
            print(f"  Locality hit : {'YES' if example['locality_hit'] else 'NO'} (fuzzy: {example['fuzzy_score']:.0f})")
            print(f"  WER          : {metric_string(example['wer'])}")

    print("\nRECOMMENDATION")
    print(analysis["recommendation"])


def save_analysis_json(analysis: dict, results_dir: Path = Path("results")) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"analysis_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(analysis, output_file, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    results_path = latest_results_file()
    print(f"Loading latest results: {results_path}")
    with results_path.open("r", encoding="utf-8") as results_file:
        data = json.load(results_file)

    analysis = compute_metrics(data)
    analysis["source_results_file"] = str(results_path)
    save_charts(analysis)
    analysis_path = save_analysis_json(analysis)
    print_summary(analysis)
    print(f"\nSaved charts to charts/")
    print(f"Saved analysis to {analysis_path}")


if __name__ == "__main__":
    main()
