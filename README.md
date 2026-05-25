# Phase 2 ASR API Benchmark

Benchmark four API-based ASR models on pre-recorded Bangalore locality audio samples:
Deepgram Nova-2, OpenAI Whisper API, Sarvam AI Saarika-v2, and Google Cloud Speech-to-Text v1.

## Prerequisites

- Python 3.10+
- `ffmpeg` installed and available on `PATH` for `pydub` audio conversion.
- Google Cloud service account JSON with Speech-to-Text access enabled.
- API keys for Deepgram, OpenAI, and Sarvam AI.

Install `ffmpeg`:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg
```

Google setup:

1. Create or select a Google Cloud project.
2. Enable the Cloud Speech-to-Text API.
3. Create a service account with permission to call Speech-to-Text.
4. Download the service account JSON file.
5. Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env` to that JSON file path.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```bash
DEEPGRAM_API_KEY=your_deepgram_key
OPENAI_API_KEY=your_openai_key
SARVAM_API_KEY=your_sarvam_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

Validate credentials and basic API reachability:

```bash
python setup_check.py
```

## Run

Dry run on the first 3 files:

```bash
python run_benchmark.py --dry-run
```

Run all models on all files:

```bash
python run_benchmark.py --model all
```

Run one model:

```bash
python run_benchmark.py --model deepgram
python run_benchmark.py --model whisper
python run_benchmark.py --model sarvam
python run_benchmark.py --model google
```

Results are written to `results/results_YYYYMMDD_HHMMSS.json` unless `--output` is provided.

## Expected Input Layout

Preferred:

```text
audio_samples/
├── 01_koramangala_noisy.wav
├── ...
└── metadata.json
```

This repo currently uses `audiosample/`; `run_benchmark.py` automatically checks both:

```text
audiosample/
├── 01_kr_puram_fast.wav
├── ...
└── metadata.json
```

`metadata.json` must contain:

```json
{
  "recordings": [
    {
      "file": "01_koramangala_noisy.wav",
      "locality": "Koramangala",
      "sentence": "Haan, main Koramangala mein rehta hoon",
      "condition": "noisy",
      "language": "Hindi",
      "sample_rate": 22050
    }
  ]
}
```

## Notes

- All API calls retry after 1s, 2s, and 4s before failing that model/file.
- The benchmark continues after a model failure and records the error in the result JSON.
- Google audio is converted with `pydub` to mono 16 kHz LINEAR16 WAV before recognition.
- Whisper files are checked against the 25 MB API upload limit before submission.
- A 0.5s pause is inserted between API calls to reduce rate-limit pressure.
# Vaahan-ai-assessment
