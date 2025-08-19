# mIsper – MLX Whisper Video Transcriber (macOS)

A modern Python GUI app for macOS (Qt/PySide6) that transcribes video/audio files using OpenAI Whisper running natively on Apple Silicon via MLX (GPU-accelerated). Supports model selection (tiny/base/small/medium/large-v3/turbo/distil) and outputs .txt or .srt.

## Features
- Multiple file selection (video: mp4/mov/m4v/mkv/avi; audio: wav/mp3/m4a/flac/ogg/aac)
- Model picker with popular MLX-converted Whisper checkpoints
- Output formats: plain text (.txt) or subtitles (.srt)
- Progress, logs, and graceful stop
- Extracts audio from videos via FFmpeg automatically
- Live in-app transcription streaming (chunked processing with FFmpeg)
- Word-level timestamps (enable in-app option)
- Optional speaker recognition (experimental) with segment-level speaker labels in .txt/.srt

## Requirements
- macOS on Apple Silicon (MLX uses Apple GPUs/NPUs)
- Python 3.10+
- FFmpeg (for video files)

## Quickstart
1) Optional: create a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies.

```bash
pip install -r requirements.txt
```

3) Ensure FFmpeg is installed (for video input).

```bash
# Using Homebrew
brew install ffmpeg
```

4) Run the app (Qt-based GUI).

```bash
python3 qt_app.py
```

On first use of a given model, weights are downloaded from Hugging Face and cached under `~/.cache/huggingface`.

### Optional: Speaker Recognition (Diarization)
Speaker labels are supported via a lightweight pipeline using Resemblyzer + SpectralClusterer. These are optional dependencies.

Install extras (CPU-based):

```bash
pip install Resemblyzer spectralcluster librosa
```

Notes:
- Diarization is experimental and may be sensitive to audio quality and overlapping speech.
- If extras are not installed, the app will skip diarization gracefully.

### Streaming & Timestamps
- Streaming shows partial transcripts live while processing by chunking audio with FFmpeg.
- Word-level timestamps can be enabled; `mlx_whisper.transcribe(..., word_timestamps=True)` is used internally.
- .srt export uses segment timestamps; when speaker labels are available, they are prefixed in the subtitle text.

## Models (Hugging Face)
These repos are known-good MLX Whisper checkpoints (auto-downloaded by `mlx-whisper`):
- `mlx-community/whisper-tiny`
- `mlx-community/whisper-base-mlx`
- `mlx-community/whisper-small-mlx`
- `mlx-community/whisper-small.en-mlx`
- `mlx-community/whisper-medium-mlx`
- `mlx-community/whisper-medium.en-mlx`
- `mlx-community/whisper-large-v3-mlx`
- `mlx-community/whisper-large-v3-turbo`
- `mlx-community/distil-whisper-large-v3`
- Quantized options: `mlx-community/whisper-base-mlx-8bit`, `mlx-community/whisper-small-mlx-4bit`

You can add more by editing the `MODELS` list in `qt_app.py`.

## References
- PyPI – mlx-whisper: https://pypi.org/project/mlx-whisper/
- MLX Whisper examples: https://github.com/ml-explore/mlx-examples/tree/main/whisper
- MLX Community Whisper models: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc
- Example write-up: https://simonwillison.net/2024/Aug/13/mlx-whisper/

## Notes
- Large models need more RAM/VRAM; try a smaller or quantized model if you run into memory issues.
- For .srt, we use segment timestamps from `mlx_whisper.transcribe`. If segments are unavailable, we emit a single segment as a fallback.
- The first run of a new model may take longer due to weight downloads and compilation.
- Streaming mode requires FFmpeg (both `ffmpeg` and `ffprobe`).
- Diarization requires optional packages (see above) and runs on CPU; it adds extra processing time.
