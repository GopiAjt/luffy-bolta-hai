import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from app.config import UPLOADS_DIR, normalize_video_profile
from app.utils.audio_processor import get_audio_duration

logger = logging.getLogger(__name__)

DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_VOICE_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "Voice_Sample.wav"
DEFAULT_VOICE_SAMPLE_TRANSCRIPT = (
    "So chapter 1182 dropped and we gotta talk about Ragnar. "
    "Imu's all like traitor about Nidhogg, "
    "right? But then Ragnar appears, literally the hammer Ratatoskr"
)
DEFAULT_VOICE_INSTRUCT = """
Young male anime-style narrator with consistent identity and stable pitch range.

Maintain controlled energetic delivery with forward momentum.
Avoid monotone speech and avoid exaggerated emotional swings.

Emotional shaping depends on section mode:
- HOOK: high urgency, fast, attention-grabbing
- BUILDUP: steady, informative, curious
- CLIMAX: powerful emphasis, controlled intensity, impactful pauses
- RESOLVE: calm, stable conclusion

Emphasis rules:
- Limit emphasis to key words only
- Use pitch variation, not volume spikes
- Never flatten entire sentences

Always maintain clarity and natural rhythm.
"""

_tts_model = None
_tts_model_id = None


def _split_paragraphs_for_tts(text: str) -> List[str]:
    """Split narration into paragraph blocks before TTS chunking."""
    text = (text or "").strip()
    if not text:
        return []
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text) if paragraph.strip()]


def _split_paragraph_for_tts(text: str, max_chars: int) -> List[str]:
    """Split one paragraph into sentence-aware chunks for Qwen TTS."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?।])\s+", text)
    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            words = sentence.split()
            word_chunk = []
            for word in words:
                candidate = " ".join([*word_chunk, word])
                if len(candidate) > max_chars and word_chunk:
                    chunks.append(" ".join(word_chunk))
                    word_chunk = [word]
                else:
                    word_chunk.append(word)
            if word_chunk:
                chunks.append(" ".join(word_chunk))
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current.strip())
    return chunks


def _split_text_for_tts(text: str, max_chars: int) -> List[str]:
    """Split narration into paragraph-first, sentence-aware chunks for Qwen TTS."""
    chunks: List[str] = []
    for paragraph in _split_paragraphs_for_tts(text):
        chunks.extend(_split_paragraph_for_tts(paragraph, max_chars))
    return chunks


def _concat_wav_files(input_paths: List[Path], output_path: Path) -> None:
    concat_path = output_path.with_suffix(".concat.txt")
    try:
        lines = []
        for path in input_paths:
            escaped = str(path).replace("'", "'\\''")
            lines.append(f"file '{escaped}'\n")
        concat_path.write_text("".join(lines), encoding="utf-8")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c:a",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "1",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        concat_path.unlink(missing_ok=True)


def _master_voiceover(raw_path: Path, output_path: Path) -> None:
    """Apply light voiceover mastering with FFmpeg, falling back to raw audio."""
    if os.getenv("QWEN_TTS_MASTERING", "1").strip().lower() in {"0", "false", "no"}:
        shutil.move(str(raw_path), str(output_path))
        logger.info("Voiceover mastering disabled; using raw generated audio")
        return

    try:
        raw_duration = get_audio_duration(str(raw_path))
    except Exception:
        raw_duration = 0.0

    fade_out_start = max(0.0, raw_duration - 0.08)
    audio_filter = (
        "loudnorm=I=-16:TP=-1.5:LRA=11,"
        "acompressor=threshold=-18dB:ratio=2.2:attack=20:release=250,"
        "alimiter=limit=0.95,"
        "afade=t=in:st=0:d=0.04,"
        f"afade=t=out:st={fade_out_start:.3f}:d=0.08"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_path),
        "-af",
        audio_filter,
        "-ar",
        "44100",
        "-ac",
        "1",
        str(output_path),
    ]

    try:
        logger.info("Mastering voiceover with FFmpeg")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        raw_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Voiceover mastering failed, using raw audio: %s", exc)
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(raw_path), str(output_path))


@contextmanager
def _progress_logger(label: str, interval_seconds: int = 10):
    """Log a heartbeat while a long Qwen TTS step is running."""
    stop_event = threading.Event()
    started_at = time.monotonic()

    def log_progress():
        while not stop_event.wait(interval_seconds):
            elapsed = time.monotonic() - started_at
            logger.info("%s still running... elapsed %.1fs", label, elapsed)

    logger.info("%s started", label)
    thread = threading.Thread(target=log_progress, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        elapsed = time.monotonic() - started_at
        logger.info("%s finished in %.1fs", label, elapsed)


def _torch_dtype(torch):
    dtype_name = os.getenv("QWEN_TTS_DTYPE", "bfloat16").strip().lower()
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    if dtype_name in {"float16", "fp16"}:
        return torch.float16
    return torch.bfloat16


def _load_qwen_tts_model(model_id: str = DEFAULT_TTS_MODEL):
    """
    Lazily load Qwen3-TTS.

    The official examples target CUDA + FlashAttention. This project defaults to
    CPU because the user's laptop has no GPU, so we avoid FlashAttention and keep
    dtype configurable via QWEN_TTS_DTYPE.
    """
    global _tts_model, _tts_model_id

    if _tts_model is not None and _tts_model_id == model_id:
        return _tts_model

    numba_cache_dir = Path(os.getenv("QWEN_TTS_NUMBA_CACHE_DIR", "/tmp/qwen_tts_numba_cache"))
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))

    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError(
            "Qwen TTS dependencies are missing. Install them with "
            "`pip install -U qwen-tts soundfile`."
        ) from exc

    device = os.getenv("QWEN_TTS_DEVICE", "cpu")
    dtype = _torch_dtype(torch)

    logger.info(
        "Loading Qwen TTS model %s on %s with dtype=%s. This can take a while on CPU.",
        model_id,
        device,
        dtype,
    )
    with _progress_logger("Qwen TTS model load"):
        _tts_model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            attn_implementation=os.getenv("QWEN_TTS_ATTN", "eager"),
        )
    _tts_model_id = model_id
    return _tts_model


def _voice_sample_path() -> Optional[Path]:
    configured_path = os.getenv("QWEN_TTS_VOICE_SAMPLE_PATH")
    sample_path = Path(configured_path).expanduser() if configured_path else DEFAULT_VOICE_SAMPLE_PATH
    if not sample_path.exists():
        return None
    return sample_path


def _voice_sample_transcript() -> str:
    return os.getenv("QWEN_TTS_VOICE_SAMPLE_TRANSCRIPT", DEFAULT_VOICE_SAMPLE_TRANSCRIPT).strip()


def generate_voiceover(
    text: str,
    output_dir: Path = UPLOADS_DIR,
    language: str = "English",
    instruct: Optional[str] = None,
    model_id: str = DEFAULT_TTS_MODEL,
    video_profile: str = "short_vertical",
) -> dict:
    """
    Generate a voiceover WAV with Qwen3-TTS voice cloning.

    Returns a dict shaped like the upload endpoint response:
    {"id": filename, "duration": seconds, "path": full_path}
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is required to generate a voiceover")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4()}.wav"
    raw_output_path = output_path.with_suffix(".raw.wav")
    voice_sample_path = _voice_sample_path()
    if voice_sample_path is None:
        raise RuntimeError(
            f"Voice sample not found. Add it at {DEFAULT_VOICE_SAMPLE_PATH} "
            "or set QWEN_TTS_VOICE_SAMPLE_PATH."
        )
    voice_sample_transcript = _voice_sample_transcript()
    language = (language or "English").strip()
    video_profile = normalize_video_profile(video_profile)

    model = _load_qwen_tts_model(model_id)

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "The `soundfile` package is required to save generated audio. "
            "Install it with `pip install soundfile`."
        ) from exc

    max_chars = int(os.getenv("QWEN_TTS_MAX_CHARS", "2400"))
    chunks = _split_text_for_tts(text, max_chars if video_profile == "long_youtube" else len(text))
    if not chunks:
        raise ValueError("Text is required to generate a voiceover")

    logger.info(
        "Generating Qwen voiceover (%d chars, %d chunk(s), language=%s, profile=%s)",
        len(text),
        len(chunks),
        language,
        video_profile,
    )
    logger.info("Using voice clone sample: %s", voice_sample_path)

    chunk_paths = []
    sample_rate = None
    try:
        for idx, chunk in enumerate(chunks, 1):
            chunk_path = output_path.with_suffix(f".part{idx}.raw.wav")
            logger.info("Generating TTS chunk %d/%d (%d chars)", idx, len(chunks), len(chunk))
            with _progress_logger(f"Qwen voiceover generation chunk {idx}/{len(chunks)}"):
                wavs, sample_rate = model.generate_voice_clone(
                    text=chunk,
                    language=language,
                    ref_audio=str(voice_sample_path),
                    ref_text=voice_sample_transcript,
                    x_vector_only_mode=False,
                )

            if not wavs:
                raise RuntimeError(f"Qwen TTS returned no audio for chunk {idx}")
            logger.info("Writing generated chunk WAV to %s at %s Hz", chunk_path, sample_rate)
            sf.write(str(chunk_path), wavs[0], sample_rate)
            chunk_paths.append(chunk_path)

        if len(chunk_paths) == 1:
            shutil.move(str(chunk_paths[0]), str(raw_output_path))
        else:
            logger.info("Concatenating %d generated voiceover chunks", len(chunk_paths))
            _concat_wav_files(chunk_paths, raw_output_path)
    finally:
        for chunk_path in chunk_paths:
            chunk_path.unlink(missing_ok=True)

    _master_voiceover(raw_output_path, output_path)
    duration = get_audio_duration(str(output_path))
    logger.info("Generated voiceover %s (duration %.2fs)", output_path.name, duration)

    return {
        "id": output_path.name,
        "duration": duration,
        "path": str(output_path),
    }
