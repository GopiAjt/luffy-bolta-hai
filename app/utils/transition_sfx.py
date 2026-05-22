"""Transition sound effects mixed into the voiceover track."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SFX_DIR = Path(
    os.getenv(
        "TRANSITION_SFX_DIR",
        Path(__file__).resolve().parent.parent / "data" / "sfx",
    )
)
ENABLE_TRANSITION_SFX = os.getenv("ENABLE_TRANSITION_SFX", "true").lower() not in {
    "0",
    "false",
    "no",
}
TRANSITION_SFX_VOLUME = float(os.getenv("TRANSITION_SFX_VOLUME", "0.22"))

# Map slideshow transition names to sfx filenames (under SFX_DIR)
TRANSITION_SFX_FILES: Dict[str, str] = {
    "fade": "soft_whoosh.wav",
    "fade_eased": "soft_whoosh.wav",
    "crossfade": "soft_whoosh.wav",
    "zoom_dissolve": "impact_hit.wav",
    "radial_wipe": "whoosh.wav",
    "iris_wipe": "whoosh.wav",
    "whip_pan_right": "whoosh.wav",
    "whip_pan_left": "whoosh.wav",
    "motion_slide_right": "slide.wav",
    "motion_slide_left": "slide.wav",
    "slide_left": "slide.wav",
    "slide_right": "slide.wav",
    "slide_up": "slide.wav",
    "slide_down": "slide.wav",
    "cube_rotation_left": "whoosh.wav",
    "cube_rotation_right": "whoosh.wav",
    "page_curl_tl": "paper.wav",
    "page_curl_tr": "paper.wav",
    "water_ripple": "soft_whoosh.wav",
    "default": "soft_whoosh.wav",
}


def parse_time_to_seconds(value: str) -> float:
    """Parse H:MM:SS.cs to seconds."""
    parts = (value or "0:0:0").strip().split(":")
    if len(parts) != 3:
        return 0.0
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_parts = parts[2].split(".")
    seconds = int(sec_parts[0])
    centis = int(sec_parts[1]) if len(sec_parts) > 1 else 0
    return hours * 3600 + minutes * 60 + seconds + centis / 100.0


def resolve_transition_sfx(transition_type: str) -> Optional[Path]:
    if not ENABLE_TRANSITION_SFX:
        return None
    key = (transition_type or "default").lower()
    if key.startswith("whip_pan"):
        key = key
    filename = TRANSITION_SFX_FILES.get(key) or TRANSITION_SFX_FILES["default"]
    path = SFX_DIR / filename
    return path if path.exists() else None


def _generate_placeholder_wav(path: Path, duration_ms: int = 180, freq: int = 220) -> None:
    """Create a short whoosh-like noise burst with ffmpeg when assets are missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_s = duration_ms / 1000.0
    fade_out_st = max(0.03, duration_s - 0.06)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        (
            f"anoisesrc=d={duration_s:.3f}:color=pink:seed={freq},"
            f"volume=0.28,afade=t=in:st=0:d=0.03,afade=t=out:st={fade_out_st:.3f}:d=0.05"
        ),
        "-ac",
        "1",
        "-ar",
        "44100",
        str(path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        logger.info("Generated placeholder SFX: %s", path.name)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not generate placeholder SFX %s: %s", path, exc)


def ensure_sfx_library() -> None:
    """Ensure default sfx files exist (generates placeholders if missing)."""
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    for filename in set(TRANSITION_SFX_FILES.values()):
        target = SFX_DIR / filename
        if not target.exists():
            _generate_placeholder_wav(target)


def build_transition_events_from_slides(slides: List[Dict]) -> List[Dict]:
    """Build SFX cue list from slide boundaries (slide 2+ start times)."""
    events = []
    for idx, slide in enumerate(slides):
        if idx == 0:
            continue
        start = slide.get("start_time")
        if not start:
            continue
        transition_type = slide.get("transition_in") or slide.get("transition_type") or "default"
        events.append(
            {
                "time": parse_time_to_seconds(start),
                "transition": transition_type,
            }
        )
    return events


def load_transition_events(slides_json_path: str) -> List[Dict]:
    """Load .transitions.json sibling file or infer from slides."""
    slides_path = Path(slides_json_path)
    transitions_path = slides_path.with_name(
        slides_path.name.replace(".image_slides.json", ".transitions.json")
    )
    if transitions_path.exists():
        try:
            return json.loads(transitions_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Invalid transitions file: %s", transitions_path)

    if slides_path.exists():
        slides = json.loads(slides_path.read_text(encoding="utf-8"))
        return build_transition_events_from_slides(slides)
    return []


def save_transition_events(slides_json_path: str, events: List[Dict]) -> str:
    slides_path = Path(slides_json_path)
    out_path = slides_path.with_name(
        slides_path.name.replace(".image_slides.json", ".transitions.json")
    )
    out_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    return str(out_path)


def mix_transition_sfx(
    voice_audio_path: str,
    events: List[Dict],
    output_path: str,
    duration: Optional[float] = None,
) -> str:
    """
    Mix transition SFX into the narration WAV/MP3. Returns path to mixed audio.
    """
    if not ENABLE_TRANSITION_SFX or not events:
        return voice_audio_path

    ensure_sfx_library()
    voice_path = Path(voice_audio_path)
    if not voice_path.exists():
        return voice_audio_path

    inputs = ["-i", str(voice_path)]
    filter_parts = ["[0:a]asetpts=PTS-STARTPTS[voice]"]
    mix_labels = ["[voice]"]
    input_idx = 1
    used = 0

    for event in events:
        transition = event.get("transition") or "default"
        sfx_path = resolve_transition_sfx(transition)
        if not sfx_path:
            continue
        start = max(0.0, float(event.get("time", 0)))
        delay_ms = int(start * 1000)
        inputs.extend(["-i", str(sfx_path)])
        label = f"[sfx{input_idx}]"
        filter_parts.append(
            f"[{input_idx}:a]adelay={delay_ms}|{delay_ms},volume={TRANSITION_SFX_VOLUME:.3f}{label}"
        )
        mix_labels.append(label)
        input_idx += 1
        used += 1

    if not used:
        return voice_audio_path

    filter_parts.append(
        "".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0,alimiter=limit=0.98[aout]"
    )
    filter_complex = ";".join(filter_parts)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[aout]",
        "-c:a",
        "pcm_s16le",
    ]
    if duration:
        cmd.extend(["-t", f"{duration:.3f}"])
    cmd.append(str(out))

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        logger.info("Mixed %s transition SFX into %s", used, out.name)
        return str(out)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Transition SFX mix failed, using dry voice: %s", exc)
        return voice_audio_path
