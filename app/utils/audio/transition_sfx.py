"""Transition sound effects mixed into the voiceover track."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SFX_DIR = Path(
    os.getenv(
        "TRANSITION_SFX_DIR",
        Path(__file__).resolve().parents[2] / "data" / "sfx",
    )
)
ENABLE_TRANSITION_SFX = os.getenv("ENABLE_TRANSITION_SFX", "true").lower() not in {
    "0",
    "false",
    "no",
}
TRANSITION_SFX_VOLUME = float(os.getenv("TRANSITION_SFX_VOLUME", "1.25"))
# Legacy weight kept for env compatibility. The active mix now limits the SFX bed
# before adding it to voice, so narration is not pulled down by a final limiter.
TRANSITION_SFX_MIX_WEIGHT = float(os.getenv("TRANSITION_SFX_MIX_WEIGHT", "2.0"))
TRANSITION_SFX_BED_LIMIT = float(os.getenv("TRANSITION_SFX_BED_LIMIT", "0.42"))
TRANSITION_SFX_BED_GAIN = float(os.getenv("TRANSITION_SFX_BED_GAIN", "0.82"))
# Human perception usually wants transition sounds to lead the visual by a frame or two.
TRANSITION_SFX_SYNC_OFFSET = float(os.getenv("TRANSITION_SFX_SYNC_OFFSET", "-0.04"))
# Trim long MP3 assets so a single cue does not play for 3–6 seconds
TRANSITION_SFX_MAX_DURATION = float(os.getenv("TRANSITION_SFX_MAX_DURATION", "1.15"))
# Avoid adding a sound on every visual cut in long videos.
TRANSITION_SFX_MIN_GAP_SECONDS = float(os.getenv("TRANSITION_SFX_MIN_GAP_SECONDS", "4.75"))
TRANSITION_SFX_IMPACT_COOLDOWN_SECONDS = float(
    os.getenv("TRANSITION_SFX_IMPACT_COOLDOWN_SECONDS", "14.0")
)

# Per-file gain tweaks (stem without extension). Values multiply TRANSITION_SFX_VOLUME.
SFX_VOLUME_SCALE: Dict[str, float] = {
    "impact_hit": 0.95,
    "sub_boom": 0.92,
    "riser": 1.0,
    "stinger": 1.15,
    "heartbeat": 1.0,
    "electric_charge": 1.1,
    "sword_slash": 1.1,
    "sparkle": 0.70,
    "whoosh": 1.1,
    "soft_whoosh": 1.0,
    "reverse_whoosh": 1.0,
    "slide": 1.0,
}

# Exact transition name -> sfx stem (app/data/sfx/{stem}.mp3)
TRANSITION_SFX_FILES: Dict[str, str] = {
    "none": "",
    "default": "soft_whoosh",
    # Gentle blends
    "fade": "soft_whoosh",
    "fade_eased": "soft_whoosh",
    "crossfade": "soft_whoosh",
    # Impact / scale
    "zoom_dissolve": "impact_hit",
    "radial_wipe": "sub_boom",
    # Circular / rotational
    "iris_wipe": "riser",
    "cube_rotation_left": "reverse_whoosh",
    "cube_rotation_right": "reverse_whoosh",
    "cube_rotation_up": "reverse_whoosh",
    "cube_rotation_down": "reverse_whoosh",
    # Directional motion
    "whip_pan_right": "whoosh",
    "whip_pan_left": "whoosh",
    "motion_slide_right": "slide",
    "motion_slide_left": "slide",
    "slide_left": "slide",
    "slide_right": "slide",
    "slide_up": "slide",
    "slide_down": "slide",
    # Stylized
    "page_curl_tl": "stinger",
    "page_curl_tr": "stinger",
    "page_curl_bl": "stinger",
    "page_curl_br": "stinger",
    "water_ripple": "sparkle",
}

# Longest-prefix wins when exact key is missing
TRANSITION_SFX_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("whip_pan", "whoosh"),
    ("motion_slide", "slide"),
    ("slide_", "slide"),
    ("cube_rotation", "reverse_whoosh"),
    ("page_curl", "stinger"),
)

GENTLE_TRANSITION_SFX_VARIANTS: Tuple[str, ...] = (
    "soft_whoosh",
    "reverse_whoosh",
    "slide",
    "sparkle",
)

IMPACT_TRANSITION_SFX_VARIANTS: Tuple[str, ...] = (
    "impact_hit",
    "sub_boom",
    "riser",
    "reverse_whoosh",
)

# Optional per-visual-style overrides (prefix match on transition name -> sfx stem)
VISUAL_STYLE_SFX_OVERRIDES: Dict[str, Dict[str, str]] = {
    "manga_hype": {
        "whip_pan": "electric_charge",
        "zoom_dissolve": "impact_hit",
    },
    "action": {
        "whip_pan": "electric_charge",
        "motion_slide": "sword_slash",
        "slide_": "sword_slash",
    },
    "emotional": {
        "fade": "heartbeat",
        "fade_eased": "heartbeat",
        "crossfade": "heartbeat",
    },
    "dark_lore": {
        "radial_wipe": "sub_boom",
        "iris_wipe": "reverse_whoosh",
    },
}

SUPPORTED_SFX_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")


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


def _resolve_sfx_file(stem: str) -> Optional[Path]:
    """Resolve sfx stem to an existing file under SFX_DIR (mp3 preferred)."""
    if not stem:
        return None
    for ext in SUPPORTED_SFX_EXTENSIONS:
        path = SFX_DIR / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def _style_override_stem(transition_type: str, visual_style: Optional[str]) -> Optional[str]:
    style = (visual_style or "").strip().lower()
    overrides = VISUAL_STYLE_SFX_OVERRIDES.get(style)
    if not overrides:
        return None
    key = (transition_type or "").lower()
    if key in overrides:
        return overrides[key]
    for prefix, stem in overrides.items():
        if key.startswith(prefix):
            return stem
    return None


def _sfx_stem_for_transition(
    transition_type: str,
    visual_style: Optional[str] = None,
) -> Optional[str]:
    key = (transition_type or "default").lower().strip()
    if key in {"", "none"}:
        return None
    styled = _style_override_stem(key, visual_style)
    if styled:
        return styled
    if key in TRANSITION_SFX_FILES:
        stem = TRANSITION_SFX_FILES[key]
        return stem or None
    for prefix, stem in TRANSITION_SFX_PREFIXES:
        if key.startswith(prefix):
            return stem
    return TRANSITION_SFX_FILES.get("default", "soft_whoosh")


def _transition_sfx_candidates(
    transition_type: str,
    visual_style: Optional[str] = None,
) -> Tuple[str, ...]:
    """Return preferred SFX stems for a transition, with variety fallbacks."""
    key = (transition_type or "default").lower().strip()
    styled = _style_override_stem(key, visual_style)
    base = styled or _sfx_stem_for_transition(key, visual_style=None)
    if not base:
        return tuple()

    if key in {"fade", "fade_eased", "crossfade", "default"}:
        return tuple(dict.fromkeys((base, *GENTLE_TRANSITION_SFX_VARIANTS)))
    if key == "zoom_dissolve":
        return tuple(dict.fromkeys((base, *IMPACT_TRANSITION_SFX_VARIANTS)))
    if base in {"soft_whoosh", "whoosh", "reverse_whoosh", "slide", "sparkle"}:
        return tuple(dict.fromkeys((base, *GENTLE_TRANSITION_SFX_VARIANTS)))
    return (base,)


def _pick_varied_sfx_stem(
    candidates: Tuple[str, ...],
    index: int,
    previous_stem: Optional[str],
    last_impact_time: Optional[float],
    event_time: float,
) -> Optional[str]:
    if not candidates:
        return None

    if "impact_hit" in candidates:
        impact_ready = (
            last_impact_time is None
            or event_time - last_impact_time >= TRANSITION_SFX_IMPACT_COOLDOWN_SECONDS
        )
        if impact_ready and previous_stem != "impact_hit" and _resolve_sfx_file("impact_hit"):
            return "impact_hit"

    ordered = list(candidates)
    if len(ordered) > 1:
        offset = index % len(ordered)
        ordered = ordered[offset:] + ordered[:offset]

    for stem in ordered:
        if stem == previous_stem and len(candidates) > 1:
            continue
        if stem == "impact_hit" and last_impact_time is not None:
            if event_time - last_impact_time < TRANSITION_SFX_IMPACT_COOLDOWN_SECONDS:
                continue
        if _resolve_sfx_file(stem):
            return stem

    for stem in candidates:
        if _resolve_sfx_file(stem):
            return stem
    return None


def plan_transition_sfx_cues(
    events: List[Dict],
    visual_style: Optional[str] = None,
) -> List[Dict]:
    """Plan sparse, varied SFX cues from transition events."""
    planned: List[Dict] = []
    previous_stem: Optional[str] = None
    last_cue_time: Optional[float] = None
    last_impact_time: Optional[float] = None

    for index, event in enumerate(events):
        transition = event.get("transition") or "default"
        event_time = float(event.get("time", 0))
        if last_cue_time is not None and event_time - last_cue_time < TRANSITION_SFX_MIN_GAP_SECONDS:
            logger.debug(
                "Skipping transition SFX at %.2fs (%s): too close to previous cue",
                event_time,
                transition,
            )
            continue

        candidates = _transition_sfx_candidates(transition, visual_style=visual_style)
        stem = _pick_varied_sfx_stem(
            candidates,
            index,
            previous_stem,
            last_impact_time,
            event_time,
        )
        sfx_path = _resolve_sfx_file(stem or "")
        if not sfx_path:
            logger.debug("No SFX file for transition %r", transition)
            continue

        planned.append(
            {
                "time": event_time,
                "transition": transition,
                "sfx_path": sfx_path,
                "stem": sfx_path.stem.lower(),
            }
        )
        previous_stem = sfx_path.stem.lower()
        last_cue_time = event_time
        if previous_stem == "impact_hit":
            last_impact_time = event_time

    return planned


def resolve_transition_sfx(
    transition_type: str,
    visual_style: Optional[str] = None,
) -> Optional[Path]:
    if not ENABLE_TRANSITION_SFX:
        return None
    stem = _sfx_stem_for_transition(transition_type, visual_style=visual_style)
    if not stem:
        return None
    path = _resolve_sfx_file(stem)
    if not path:
        logger.debug("No SFX file for transition %r (expected %s.*)", transition_type, stem)
    return path


def sfx_volume_for_path(sfx_path: Path) -> float:
    stem = sfx_path.stem.lower()
    scale = SFX_VOLUME_SCALE.get(stem, 1.0)
    return TRANSITION_SFX_VOLUME * scale


def ensure_sfx_library() -> None:
    """Log missing mapped SFX; does not generate placeholders when using real MP3 assets."""
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    required_stems = set(TRANSITION_SFX_FILES.values()) | {s for _, s in TRANSITION_SFX_PREFIXES}
    required_stems.discard("")
    missing = [stem for stem in sorted(required_stems) if not _resolve_sfx_file(stem)]
    if missing:
        logger.warning(
            "Missing transition SFX in %s: %s (expected %s)",
            SFX_DIR,
            ", ".join(missing),
            ", ".join(f"{s}.mp3" for s in missing),
        )
    else:
        found = sorted(p.name for p in SFX_DIR.iterdir() if p.suffix.lower() in SUPPORTED_SFX_EXTENSIONS)
        logger.debug("Transition SFX library OK (%s files)", len(found))


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
        if (transition_type or "").lower() == "none":
            continue
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


def _sfx_clip_filter(input_idx: int, delay_ms: int, volume: float, label: str) -> str:
    """Trim, fade, delay, and level a single SFX cue."""
    max_dur = TRANSITION_SFX_MAX_DURATION
    fade_out_st = max(0.02, max_dur - 0.08)
    return (
        f"[{input_idx}:a]aresample=44100,atrim=0:{max_dur:.3f},asetpts=PTS-STARTPTS,"
        f"afade=t=in:st=0:d=0.01,afade=t=out:st={fade_out_st:.3f}:d=0.07,"
        f"adelay={delay_ms}|{delay_ms},volume={volume:.3f}{label}"
    )


def mix_transition_sfx(
    voice_audio_path: str,
    events: List[Dict],
    output_path: str,
    duration: Optional[float] = None,
    visual_style: Optional[str] = None,
) -> str:
    """
    Mix transition SFX into the narration track.

    Builds a separate SFX bed, limits that bed, then adds it to the voice with
    normalize=0. The final mix intentionally has no limiter, because a final
    limiter reacts to SFX peaks by pulling down the narration.
    """
    if not ENABLE_TRANSITION_SFX or not events:
        return voice_audio_path

    ensure_sfx_library()
    voice_path = Path(voice_audio_path)
    if not voice_path.exists():
        return voice_audio_path

    inputs = ["-i", str(voice_path)]
    filter_parts = [
        "[0:a]aresample=44100,asetpts=PTS-STARTPTS[voice]"
    ]
    input_idx = 1
    used = 0
    sfx_labels = []
    planned_cues = plan_transition_sfx_cues(events, visual_style=visual_style)

    for event in planned_cues:
        transition = event.get("transition") or "default"
        sfx_path = event["sfx_path"]
        start = max(0.0, float(event.get("time", 0)) + TRANSITION_SFX_SYNC_OFFSET)
        delay_ms = int(start * 1000)
        volume = sfx_volume_for_path(sfx_path)
        inputs.extend(["-i", str(sfx_path)])
        sfx_label = f"[sfx{input_idx}]"
        filter_parts.append(
            _sfx_clip_filter(input_idx, delay_ms, volume, sfx_label)
        )
        sfx_labels.append(sfx_label)
        input_idx += 1
        used += 1
        logger.info(
            "SFX cue @ %.2fs: %s -> %s (vol=%.2f)",
            start,
            transition,
            sfx_path.name,
            volume,
        )

    if not used:
        logger.warning("No transition SFX resolved for %s events", len(events))
        return voice_audio_path

    if len(sfx_labels) == 1:
        filter_parts.append(
            f"{sfx_labels[0]}alimiter=limit={TRANSITION_SFX_BED_LIMIT:.3f},"
            f"volume={TRANSITION_SFX_BED_GAIN:.3f}[sfxbed]"
        )
    else:
        filter_parts.append(
            f"{''.join(sfx_labels)}amix=inputs={len(sfx_labels)}:duration=longest:"
            f"dropout_transition=0:normalize=0,"
            f"alimiter=limit={TRANSITION_SFX_BED_LIMIT:.3f},"
            f"volume={TRANSITION_SFX_BED_GAIN:.3f}[sfxbed]"
        )
    filter_parts.append(
        "[voice][sfxbed]amix=inputs=2:duration=first:dropout_transition=0:"
        "normalize=0:weights=1 1[aout]"
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
        "-ar",
        "44100",
        "-ac",
        "1",
    ]
    if duration:
        cmd.extend(["-t", f"{duration:.3f}"])
    cmd.append(str(out))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
        if result.stderr:
            logger.debug("SFX mix ffmpeg: %s", result.stderr.strip())
        logger.info(
            "Mixed %s transition SFX cues into %s (voice level preserved)",
            used,
            out.name,
        )
        return str(out)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Transition SFX mix failed, using dry voice: %s",
            (exc.stderr or str(exc)).strip(),
        )
        return voice_audio_path
    except FileNotFoundError as exc:
        logger.warning("Transition SFX mix failed, using dry voice: %s", exc)
        return voice_audio_path
