import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from app.config import UPLOADS_DIR
from app.utils.audio_processor import get_audio_duration

logger = logging.getLogger(__name__)

DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_VOICE_INSTRUCT = (
    "Young energetic male anime narrator, bright and adventurous, expressive "
    "anime fan energy, clear English delivery, medium-fast pace, excited but not shouting"
)

_tts_model = None
_tts_model_id = None


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
    _tts_model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        attn_implementation=os.getenv("QWEN_TTS_ATTN", "eager"),
    )
    _tts_model_id = model_id
    return _tts_model


def generate_voiceover(
    text: str,
    output_dir: Path = UPLOADS_DIR,
    language: str = "English",
    instruct: Optional[str] = None,
    model_id: str = DEFAULT_TTS_MODEL,
) -> dict:
    """
    Generate a voiceover WAV with Qwen3-TTS VoiceDesign.

    Returns a dict shaped like the upload endpoint response:
    {"id": filename, "duration": seconds, "path": full_path}
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Text is required to generate a voiceover")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{uuid.uuid4()}.wav"
    voice_instruct = (instruct or DEFAULT_VOICE_INSTRUCT).strip()
    language = (language or "English").strip()

    model = _load_qwen_tts_model(model_id)

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "The `soundfile` package is required to save generated audio. "
            "Install it with `pip install soundfile`."
        ) from exc

    logger.info("Generating Qwen voiceover (%d chars, language=%s)", len(text), language)
    wavs, sample_rate = model.generate_voice_design(
        text=text,
        language=language,
        instruct=voice_instruct,
    )

    if not wavs:
        raise RuntimeError("Qwen TTS returned no audio")

    sf.write(str(output_path), wavs[0], sample_rate)
    duration = get_audio_duration(str(output_path))

    return {
        "id": output_path.name,
        "duration": duration,
        "path": str(output_path),
    }
