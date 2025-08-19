import os
import sys
import subprocess
import threading
import queue
import tempfile
import shutil
import math
import time
from datetime import timedelta
from typing import List, Tuple

# Ensure the model is cached locally and enable explicit download step
from huggingface_hub import snapshot_download

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QComboBox,
    QTextEdit,
    QProgressBar,
    QListWidget,
    QCheckBox,
)

# MLX Whisper
try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None

# Optional speaker diarization deps (lightweight pipeline)
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from spectralcluster import SpectralClusterer
    import librosa
    import numpy as np
except Exception:
    VoiceEncoder = None
    preprocess_wav = None
    SpectralClusterer = None
    librosa = None
    np = None

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

# Curated MLX Whisper models on Hugging Face
MODELS: List[Tuple[str, str]] = [
    ("Tiny (fast, ~39M)", "mlx-community/whisper-tiny"),
    ("Base (~92M)", "mlx-community/whisper-base-mlx"),
    ("Small (~244M)", "mlx-community/whisper-small-mlx"),
    ("Small.en (English-only)", "mlx-community/whisper-small.en-mlx"),
    ("Medium (~769M)", "mlx-community/whisper-medium-mlx"),
    ("Medium.en (English-only)", "mlx-community/whisper-medium.en-mlx"),
    ("Large v3 (high accuracy)", "mlx-community/whisper-large-v3-mlx"),
    ("Large v3 Turbo (faster)", "mlx-community/whisper-large-v3-turbo"),
    ("Distil Large v3 (smaller)", "mlx-community/distil-whisper-large-v3"),
    ("Base 8-bit (low VRAM)", "mlx-community/whisper-base-mlx-8bit"),
    ("Small 4-bit (low VRAM)", "mlx-community/whisper-small-mlx-4bit"),
]

OUTPUT_FORMATS = [".txt", ".srt"]


def which_ffmpeg() -> str:
    return shutil.which("ffmpeg") or ""


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_VIDEO_EXTS


def is_audio_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_AUDIO_EXTS


def format_srt_time(seconds: float) -> str:
    td = timedelta(seconds=float(max(0.0, seconds)))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_txt(output_path: str, text: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def write_srt(output_path: str, segments: List[dict]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seg.get("start", 0.0)
            end = seg.get("end", start)
            text = seg.get("text", "").strip()
            spk = seg.get("speaker")
            if spk:
                text = f"{spk}: {text}"
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
            f.write(text + "\n\n")


def ffprobe_duration(path: str) -> float:
    """Return media duration in seconds using ffprobe, or -1 on failure."""
    try:
        cmd = [
            which_ffmpeg().replace("ffmpeg", "ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return float(out.stdout.decode().strip())
    except Exception:
        return -1.0


def extract_chunk_ffmpeg(in_path: str, out_path: str, start: float, duration: float) -> bool:
    """Extract a chunk [start, start+duration] to out_path using ffmpeg."""
    try:
        cmd = [
            which_ffmpeg(),
            "-y",
            "-ss",
            str(max(0.0, start)),
            "-t",
            str(max(0.01, duration)),
            "-i",
            in_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            out_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


def diarize_with_resemblyzer(audio_path: str) -> List[dict]:
    """
    Simple diarization using Resemblyzer + SpectralClusterer.
    Returns a list of dicts: {"start": float, "end": float, "speaker": "SPK{n}"}
    """
    if VoiceEncoder is None or preprocess_wav is None or SpectralClusterer is None or librosa is None:
        return []
    try:
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Resemblyzer expects preprocessed waveform
        wav16 = preprocess_wav(wav, source_sr=sr)
        encoder = VoiceEncoder("cpu")
        _, cont_embeds, wav_splits = encoder.embed_utterance(wav16, return_partials=True)
        # Cluster embeddings into an estimated number of speakers (auto)
        clusterer = SpectralClusterer(
            min_clusters=2,
            max_clusters=8,
            p_percentile=0.9,
            gaussian_blur_sigma=1,
            thresholding_soft_multiplier=0.01,
        )
        labels = clusterer.predict(cont_embeds)
        # Convert partial windows to speaker segments
        segs: List[dict] = []
        last_label = None
        seg_start = None
        for (s, e), lab in zip(wav_splits, labels):
            if last_label is None:
                last_label = lab
                seg_start = s
            elif lab != last_label:
                segs.append({"start": float(seg_start), "end": float(e), "speaker": f"SPK{int(last_label)+1}"})
                last_label = lab
                seg_start = s
        if seg_start is not None and last_label is not None:
            segs.append({"start": float(seg_start), "end": float(wav_splits[-1][1]), "speaker": f"SPK{int(last_label)+1}"})
        return segs
    except Exception:
        return []


def assign_speakers_to_segments(segments: List[dict], diar_segs: List[dict]) -> None:
    """Mutate segments in-place to add 'speaker' based on diarization segments (by midpoint overlap)."""
    if not segments or not diar_segs:
        return
    for seg in segments:
        mid = 0.5 * (float(seg.get("start", 0.0)) + float(seg.get("end", 0.0)))
        for d in diar_segs:
            if float(d["start"]) <= mid <= float(d["end"]):
                seg["speaker"] = d.get("speaker")
                break


class ChunkStreamer:
    """
    Background streamer that emits words gradually to the UI via log queue messages
    while the next chunk is loading/transcribing. Calling set_words() interrupts any
    ongoing stream and starts streaming the new list of words.
    """
    def __init__(self, log_q: "queue.Queue[str]", word_delay: float = 0.06):
        self.log_q = log_q
        self.word_delay = max(0.01, float(word_delay))
        self._words: list[str] = []
        self._lock = threading.Lock()
        self._new_words_evt = threading.Event()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ChunkStreamer", daemon=True)
        self._thread.start()

    def set_words(self, words: list[str]):
        # Replace current stream content
        clean = [w.strip() for w in words if w and str(w).strip()]
        with self._lock:
            self._words = clean
            # Signal replacement
            self._new_words_evt.set()

    def stop(self):
        self._stop_evt.set()
        self._new_words_evt.set()
        try:
            self._thread.join(timeout=1.5)
        except Exception:
            pass

    def _run(self):
        while not self._stop_evt.is_set():
            # Wait for words to stream
            self._new_words_evt.wait(timeout=0.1)
            if self._stop_evt.is_set():
                break
            # Copy snapshot of words and immediately clear event so a new set will interrupt
            with self._lock:
                words = list(self._words)
                self._new_words_evt.clear()
            # Stream words; if new words arrive, restart with them
            i = 0
            while i < len(words) and not self._stop_evt.is_set():
                # If another set_words happened, restart outer loop
                if self._new_words_evt.is_set():
                    break
                try:
                    self.log_q.put("__TRANSCRIPT__ " + words[i])
                except Exception:
                    pass
                time.sleep(self.word_delay)
                i += 1


class TranscribeWorker:
    def __init__(self, selected_files: List[str], model_repo: str, out_fmt: str, output_dir: str, temp_dir: str, log_q: "queue.Queue[str]", *, stream: bool = True, chunk_seconds: int = 30, word_level: bool = True, diarize: bool = False):
        self.selected_files = selected_files
        self.model_repo = model_repo
        self.out_fmt = out_fmt
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.log_q = log_q
        self.stop_requested = False
        self.local_model_path: str | None = None
        self.stream = stream
        self.chunk_seconds = max(5, int(chunk_seconds))
        self.word_level = word_level
        self.diarize = diarize
        self.streamer: ChunkStreamer | None = None

    def log(self, msg: str):
        self.log_q.put(msg)

    def run(self):
        total = len(self.selected_files)

        # Ensure the selected model is present in the local HF cache before starting
        # Large models (e.g., Large v3) can be several GB and take minutes on first download.
        try:
            self.log(f"Ensuring model is cached (this may take a while on first use): {self.model_repo}")
            self.local_model_path = snapshot_download(repo_id=self.model_repo)
            self.log(f"Model ready: {self.model_repo}")
        except Exception as e:
            self.log(f"Failed to fetch model from Hugging Face: {e}")
            return
        # Start streamer if needed
        if self.stream:
            self.streamer = ChunkStreamer(self.log_q, word_delay=0.06)

        for idx, in_path in enumerate(self.selected_files, start=1):
            if self.stop_requested:
                self.log("\nStopped by user.")
                break
            self.log(f"\n[File {idx}/{total}] {os.path.basename(in_path)}")

            # Determine output directory
            out_dir = self.output_dir if self.output_dir else (os.path.dirname(in_path) or os.getcwd())

            # Prepare input for whisper (audio path)
            audio_path = in_path
            cleanup_audio = False
            if is_video_file(in_path):
                base = os.path.splitext(os.path.basename(in_path))[0]
                audio_path = os.path.join(self.temp_dir, f"{base}.wav")
                cmd = [
                    which_ffmpeg(), "-y", "-i", in_path,
                    "-ac", "1", "-ar", "16000", "-vn",
                    audio_path,
                ]
                self.log("Extracting audio with ffmpeg…")
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    cleanup_audio = True
                except subprocess.CalledProcessError as e:
                    self.log("ffmpeg failed: " + e.stderr.decode(errors="ignore")[:800])
                    continue

            # Transcribe (streaming by chunks or single pass)
            aggregated_segments: List[dict] = []
            aggregated_text_parts: List[str] = []
            if self.stream:
                dur = ffprobe_duration(audio_path)
                if dur <= 0:
                    dur = 0.0
                num_chunks = max(1, math.ceil(dur / self.chunk_seconds)) if dur > 0 else None
                self.log(f"Transcribing in chunks (~{self.chunk_seconds}s)…")
                chunk_idx = 0
                while True:
                    if self.stop_requested:
                        break
                    # If duration known, stop at end; else break when ffmpeg fails
                    if num_chunks is not None and chunk_idx >= num_chunks:
                        break
                    start_t = chunk_idx * self.chunk_seconds
                    chunk_wav = os.path.join(self.temp_dir, f"chunk_{threading.get_ident()}_{chunk_idx}.wav")
                    ok = extract_chunk_ffmpeg(audio_path, chunk_wav, start_t, self.chunk_seconds)
                    if not ok:
                        # likely ran past end
                        break
                    try:
                        result_chunk = mlx_whisper.transcribe(
                            chunk_wav,
                            path_or_hf_repo=self.local_model_path or self.model_repo,
                            word_timestamps=self.word_level,
                            verbose=False,
                        )
                    except Exception as e:
                        self.log(f"Transcription error on chunk {chunk_idx}: {e}")
                        try:
                            os.remove(chunk_wav)
                        except Exception:
                            pass
                        break
                    # Adjust timestamps by start offset
                    segs = result_chunk.get("segments", []) or []
                    for s in segs:
                        s["start"] = float(s.get("start", 0.0)) + start_t
                        s["end"] = float(s.get("end", s.get("start", 0.0))) + start_t
                    aggregated_segments.extend(segs)
                    new_text = result_chunk.get("text", "").strip()
                    if new_text:
                        aggregated_text_parts.append(new_text)
                        # Prepare words to fake-stream during next chunk
                        words_for_stream: list[str] = []
                        if self.word_level:
                            # Use word-level timestamps if present
                            for seg in segs:
                                for w in seg.get("words", []) or []:
                                    token = w.get("word") or w.get("text") or ""
                                    if token:
                                        words_for_stream.append(str(token))
                        if not words_for_stream:
                            # Fallback: whitespace tokenization of the chunk text
                            words_for_stream = new_text.split()
                        if self.stream and self.streamer is not None:
                            # Stream these words while the NEXT chunk loads
                            self.streamer.set_words(words_for_stream)
                    try:
                        os.remove(chunk_wav)
                    except Exception:
                        pass
                    chunk_idx += 1
                result = {"text": " ".join(aggregated_text_parts).strip(), "segments": aggregated_segments}
            else:
                try:
                    self.log(f"Loading model: {self.model_repo}")
                    result = mlx_whisper.transcribe(
                        audio_path,
                        path_or_hf_repo=self.local_model_path or self.model_repo,
                        word_timestamps=self.word_level,
                        verbose=True,
                    )
                except Exception as e:
                    self.log(f"Transcription error: {e}")
                    if cleanup_audio:
                        try:
                            os.remove(audio_path)
                        except Exception:
                            pass
                    continue

            # Save output
            base = os.path.splitext(os.path.basename(in_path))[0]
            segments = result.get("segments", [])
            # Optional diarization
            if self.diarize:
                self.log("Running speaker recognition (experimental)…")
                diar_segs = diarize_with_resemblyzer(audio_path)
                if diar_segs:
                    assign_speakers_to_segments(segments, diar_segs)
                else:
                    self.log("Diarization skipped or failed (missing deps or runtime error).")
            if self.out_fmt == ".txt":
                out_path = os.path.join(out_dir, base + ".txt")
                text = result.get("text", "").strip()
                if self.diarize and segments:
                    # Produce per-segment labeled text
                    labeled = []
                    for seg in segments:
                        spk = seg.get("speaker")
                        t = seg.get("text", "").strip()
                        if not t:
                            continue
                        if spk:
                            labeled.append(f"{spk}: {t}")
                        else:
                            labeled.append(t)
                    text = "\n".join(labeled)
                write_txt(out_path, text)
                self.log(f"Saved transcript: {out_path}")
            elif self.out_fmt == ".srt":
                out_path = os.path.join(out_dir, base + ".srt")
                if not segments:
                    segments = [{"start": 0.0, "end": max(1.0, len(result.get('text', ''))/10.0), "text": result.get("text", "")}]
                write_srt(out_path, segments)
                self.log(f"Saved subtitles: {out_path}")

            if cleanup_audio:
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

            # Progress marker
            self.log(f"__PROGRESS__ {idx}/{total}")

        # Stop streamer after processing
        if self.stream and self.streamer is not None:
            try:
                # Let streamer finish current words briefly
                time.sleep(0.2)
                self.streamer.stop()
            except Exception:
                pass

        if not self.stop_requested:
            self.log("\nAll done.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WhisperMac – MLX Whisper Video Transcriber")
        self.resize(980, 620)

        self.selected_files: List[str] = []
        self.output_dir: str = ""
        self.temp_dir: str = tempfile.mkdtemp(prefix="whispermac_")
        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.worker: TranscribeWorker | None = None

        central = QWidget(self)
        self.setCentralWidget(central)

        root = QHBoxLayout(central)

        # Left controls
        left = QVBoxLayout()
        root.addLayout(left, 0)

        title = QLabel("Whisper Video Transcriber")
        title.setStyleSheet("font-size:18px; font-weight:600;")
        left.addWidget(title)

        subtitle = QLabel("Apple MLX + Whisper on Mac")
        subtitle.setStyleSheet("color: gray;")
        left.addWidget(subtitle)

        btn_pick = QPushButton("Choose Video/Audio Files…")
        btn_pick.clicked.connect(self.choose_files)
        left.addWidget(btn_pick)

        self.files_list = QListWidget()
        self.files_list.setMinimumWidth(320)
        self.files_list.setMaximumHeight(160)
        left.addWidget(self.files_list)

        left.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        for name, _repo in MODELS:
            self.model_combo.addItem(name)
        self.model_combo.setCurrentIndex(2 if self.model_combo.count() > 2 else 0)
        left.addWidget(self.model_combo)

        left.addWidget(QLabel("Output format:"))
        self.format_combo = QComboBox()
        for fmt in OUTPUT_FORMATS:
            self.format_combo.addItem(fmt)
        self.format_combo.setCurrentIndex(0)
        left.addWidget(self.format_combo)

        # Options
        self.chk_word_ts = QCheckBox("Word-level timestamps")
        self.chk_word_ts.setChecked(True)
        left.addWidget(self.chk_word_ts)

        self.chk_stream = QCheckBox("Stream transcript during processing")
        self.chk_stream.setChecked(True)
        left.addWidget(self.chk_stream)

        self.chk_diarize = QCheckBox("Speaker recognition (experimental)")
        self.chk_diarize.setChecked(False)
        left.addWidget(self.chk_diarize)

        btn_outdir = QPushButton("Choose Output Folder…")
        btn_outdir.clicked.connect(self.choose_output_dir)
        left.addWidget(btn_outdir)

        self.outdir_lbl = QLabel("(Default: alongside each input)")
        self.outdir_lbl.setWordWrap(True)
        left.addWidget(self.outdir_lbl)

        self.btn_start = QPushButton("Start Transcription")
        self.btn_start.setStyleSheet("background:#1f6aa5;color:white;padding:6px 10px;border-radius:6px;")
        self.btn_start.clicked.connect(self.start_transcription)
        left.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.request_stop)
        left.addWidget(self.btn_stop)

        left.addStretch(1)

        # Right logs
        right = QVBoxLayout()
        root.addLayout(right, 1)

        self.transcript_text = QTextEdit()
        self.transcript_text.setReadOnly(True)
        self.transcript_text.setPlaceholderText("Live transcript will appear here…")
        right.addWidget(self.transcript_text)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right.addWidget(self.log_text)

        # Bottom progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        right.addWidget(self.progress)

        # Timer to poll logs
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_log_queue)
        self.timer.start(100)

    def closeEvent(self, event):
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        finally:
            return super().closeEvent(event)

    def log(self, msg: str):
        self.log_text.moveCursor(QTextCursor.End)
        self.log_text.insertPlainText(msg + "\n")
        self.log_text.moveCursor(QTextCursor.End)

    def poll_log_queue(self):
        updated_progress = None
        while True:
            try:
                msg = self.log_q.get_nowait()
            except queue.Empty:
                break
            if msg.startswith("__PROGRESS__"):
                try:
                    _tag, frac = msg.split(" ", 1)
                    num, den = frac.strip().split("/")
                    val = int(int(num) * 100 / max(1, int(den)))
                    updated_progress = val
                except Exception:
                    pass
            elif msg.startswith("__TRANSCRIPT__"):
                # Append live transcript
                try:
                    _, content = msg.split(" ", 1)
                except ValueError:
                    content = msg
                self.transcript_text.moveCursor(QTextCursor.End)
                self.transcript_text.insertPlainText(content.strip() + " ")
                self.transcript_text.moveCursor(QTextCursor.End)
            else:
                self.log(msg)
        if updated_progress is not None:
            self.progress.setValue(updated_progress)
        # Reset buttons when thread ends
        if self.worker_thread and not self.worker_thread.is_alive():
            self.set_running(False)

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video or audio files",
            "",
            "Video/Audio Files (*.mp4 *.mov *.m4v *.mkv *.avi *.wav *.mp3 *.m4a *.flac *.ogg *.aac);;All Files (*.*)",
        )
        if not paths:
            return
        self.selected_files = list(paths)
        self.files_list.clear()
        for p in self.selected_files:
            self.files_list.addItem(p)
        self.log(f"Selected {len(self.selected_files)} file(s).")

    def choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self.output_dir = d
            self.outdir_lbl.setText(d)
        else:
            self.output_dir = ""
            self.outdir_lbl.setText("(Default: alongside each input)")

    def request_stop(self):
        if self.worker:
            self.worker.stop_requested = True
            self.log("Stop requested. Finishing current file…")

    def set_running(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def start_transcription(self):
        if not self.selected_files:
            self.log("Please select at least one file.")
            return
        if mlx_whisper is None:
            self.log("mlx-whisper is not installed. Run: pip install mlx-whisper")
            return
        if any(is_video_file(p) for p in self.selected_files) and not which_ffmpeg():
            self.log("FFmpeg is required for video files. Install via Homebrew: brew install ffmpeg")
            return
        if self.chk_stream.isChecked() and not which_ffmpeg():
            self.log("FFmpeg is required for streaming (chunking). Install via Homebrew: brew install ffmpeg")
            return

        model_label = self.model_combo.currentText()
        model_repo = dict(MODELS)[model_label]
        out_fmt = self.format_combo.currentText()

        self.set_running(True)
        self.progress.setValue(0)
        self.transcript_text.clear()

        self.worker = TranscribeWorker(
            selected_files=self.selected_files,
            model_repo=model_repo,
            out_fmt=out_fmt,
            output_dir=self.output_dir,
            temp_dir=self.temp_dir,
            log_q=self.log_q,
            stream=self.chk_stream.isChecked(),
            word_level=self.chk_word_ts.isChecked(),
            diarize=self.chk_diarize.isChecked(),
        )
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker_thread.start()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
