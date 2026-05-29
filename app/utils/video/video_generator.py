import os
import subprocess
import logging
import sys
import traceback
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import tempfile
import json
from collections import deque


from app.utils.text.subtitle_generator import SubtitleGenerator
from app.config import BACKGROUND_MUSIC_VOLUME
from app.utils.video.visual_effects import build_global_ffmpeg_filter
from app.utils.expressions.expression_assets import resolve_expression_image
from app.utils.expressions.expression_effects import (
    format_expression_filter_step,
    format_expression_overlay,
    resolve_expression_effect,
)
from app.utils.expressions.expression_overlay_cv import render_expression_overlays_opencv
from app.config import NARRATOR_CHARACTER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/moviepy_debug.log')
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Max overlays per FFmpeg pass (avoids huge filter_complex graphs)
EXPRESSION_OVERLAY_CHUNK_SIZE = max(
    4, int(os.getenv("EXPRESSION_OVERLAY_CHUNK_SIZE", "12"))
)
# opencv = single-pass (fast); ffmpeg = chunked filter_complex (legacy)
EXPRESSION_RENDERER = os.getenv("EXPRESSION_RENDERER", "opencv").strip().lower()
EXPRESSION_MERGE_GAP_SECONDS = float(os.getenv("EXPRESSION_MERGE_GAP_SECONDS", "0.15"))


def _x264_speed_settings(quality_mode: str) -> Tuple[str, str]:
    """Return slideshow-friendly x264 preset/crf values."""
    quality_mode = (quality_mode or 'standard').lower()
    if quality_mode == 'pro':
        return os.getenv("VIDEO_X264_PRESET_PRO", "fast"), os.getenv("VIDEO_X264_CRF_PRO", "22")
    return os.getenv("VIDEO_X264_PRESET", "veryfast"), os.getenv("VIDEO_X264_CRF", "23")


def _video_threads() -> str:
    """Return FFmpeg encoder thread count; 0 lets x264 auto-select."""
    return os.getenv("VIDEO_FFMPEG_THREADS", os.getenv("FFMPEG_THREADS", "0"))


def _video_encoder() -> str:
    """Return configured final-video encoder."""
    return os.getenv("VIDEO_ENCODER", "libx264").strip()


def _escape_ffmpeg_subtitles_path(path: str) -> str:
    """Escape a path for use inside the FFmpeg subtitles filter."""
    resolved = str(Path(path).resolve())
    for char in ("\\", ":", "'", "[", "]", ",", ";"):
        resolved = resolved.replace(char, f"\\{char}")
    return resolved


def _video_encoder_args(quality_mode: str) -> List[str]:
    """Build encoder args for CPU x264 or opt-in hardware encoders."""
    encoder = _video_encoder()
    preset, crf = _x264_speed_settings(quality_mode)
    if encoder == "libx264":
        return [
            "-c:v", encoder,
            "-preset", preset,
            "-crf", crf,
            "-threads", _video_threads(),
        ]
    if encoder == "h264_qsv":
        return [
            "-c:v", encoder,
            "-preset", os.getenv("QSV_PRESET", "veryfast"),
            "-global_quality", os.getenv("QSV_GLOBAL_QUALITY", "23"),
        ]
    return ["-c:v", encoder]


class VideoGenerator:
    def __init__(self, ffmpeg_path: str = 'ffmpeg', ffprobe_path: str = 'ffprobe'):
        """
        Initialize the VideoGenerator with paths to ffmpeg and ffprobe executables.

        Args:
            ffmpeg_path: Path to ffmpeg executable
            ffprobe_path: Path to ffprobe executable
        """
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path

    def _temp_output_path(self, output_path: str) -> str:
        path = Path(output_path)
        return str(path.with_name(f".{path.name}.tmp{path.suffix}"))

    def _run_ffmpeg(self, cmd: List[str]) -> None:
        """Run FFmpeg while streaming progress into the app logs."""
        logger.info("Starting FFmpeg process")
        stderr_tail = deque(maxlen=80)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stderr is not None
        for line in process.stderr:
            clean = line.rstrip()
            if not clean:
                continue
            stderr_tail.append(clean)
            if "frame=" in clean or "time=" in clean or "speed=" in clean:
                logger.info("FFmpeg progress: %s", clean)
            else:
                logger.debug("FFmpeg: %s", clean)

        return_code = process.wait()
        if return_code != 0:
            stderr = "\n".join(stderr_tail)
            raise subprocess.CalledProcessError(return_code, cmd, stderr=stderr)

    def _validate_media_file(self, output_path: str) -> None:
        result = subprocess.run(
            [
                self.ffprobe,
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Output video is not readable: {result.stderr.strip()}")

    def _mix_narration_with_background_music(
        self,
        narration_path: str,
        background_music_path: str,
        duration: float,
        output_path: str,
    ) -> str:
        """
        Duck background music under narration in a small audio-only FFmpeg graph.
        Kept separate from the video filter_complex to avoid failures when many
        expression overlay inputs are present.
        """
        music_volume = max(0.0, min(float(BACKGROUND_MUSIC_VOLUME), 1.0))
        bgm_fade_start = max(0.0, duration - 0.6)
        # Duck BGM under voice; keep voice at full weight (normalize=0). Short release
        # avoids music swelling between words and at the end of the video.
        # asplit narration: sidechaincompress consumes its inputs; [nar] stays for amix
        filter_audio = (
            f"[0:a]aresample=44100,asplit=2[nar][narsc];"
            f"[1:a]atrim=duration={duration:.3f},asetpts=N/SR/TB,aresample=44100,"
            f"volume={music_volume:.3f},"
            f"afade=t=out:st={bgm_fade_start:.3f}:d=0.55[bgm];"
            f"[bgm][narsc]sidechaincompress=threshold=0.02:ratio=5:attack=12:release=90:mix=1[ducked];"
            f"[nar][ducked]amix=inputs=2:duration=first:dropout_transition=0:"
            f"normalize=0:weights=1 0.32,alimiter=limit=0.95[aout]"
        )
        cmd = [
            self.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(narration_path),
            "-stream_loop",
            "-1",
            "-i",
            str(background_music_path),
            "-filter_complex",
            filter_audio,
            "-map",
            "[aout]",
            "-t",
            f"{duration:.3f}",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        self._run_ffmpeg(cmd)
        return output_path

    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file in seconds.

        Args:
            audio_path: Path to the audio file

        Returns:
            Duration in seconds as float
        """
        try:
            cmd = [
                self.ffprobe,
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(audio_path)
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"Error getting audio duration: {e}")
            raise RuntimeError(f"Could not get audio duration: {e}")

    def generate_video(
        self,
        audio_path: str,
        subtitle_path: str,
        output_path: str,
        background_video_path: Optional[str] = None,
        resolution: str = '1080x1920',  
        color: str = 'green',
        fps: int = 30,
        subtitle_effect: str = 'fade',  
        font_size: int = 72,          
        font: str = 'Roboto',          
        convert_to_mp4: bool = True,
        quality_mode: str = 'standard',
        background_music_path: Optional[str] = None,
        visual_style: Optional[str] = None,
    ) -> str:
        """
        Generate a video with audio and animated subtitles using MoviePy.

        Args:
            audio_path: Path to the audio file
            subtitle_path: Path to the ASS subtitle file (or JSON with phrase/word timing)
            output_path: Path where the output video will be saved
            resolution: Video resolution (default: '1080x1920')
            color: Background color (default: 'green')
            fps: Frames per second (default: 30)
            subtitle_effect: Effect for subtitles ('fade' or 'pop')
            font_size: Base font size for subtitles
            font: Font name for subtitles

        Returns:
            Path to the generated video file
        """
        # (REMOVED: All MoviePy-based video generation code. generate_video now uses ffmpeg for video and subtitle processing, similar to generate_video_with_expressions. All MoviePy imports and error handling removed.)

        logger.info("=" * 80)
        logger.info("STARTING VIDEO GENERATION")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Subtitles: {subtitle_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Background: {background_video_path or 'Color: ' + color}")
        logger.info(f"Resolution: {resolution}, FPS: {fps}, Effect: {subtitle_effect}")
        logger.info("=" * 80)

        try:
            # Parse resolution
            try:
                res_x, res_y = map(int, resolution.lower().split('x'))
                logger.debug(f"Parsed resolution: {res_x}x{res_y}")
            except Exception as e:
                logger.warning(f"Invalid resolution '{resolution}', defaulting to 1080x1920. Error: {e}")
                res_x, res_y = 1080, 1920

            quality_mode = (quality_mode or 'standard').lower()
            pro_mode = quality_mode == 'pro'
            duration = self.get_audio_duration(audio_path)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            escaped_subtitle_path = _escape_ffmpeg_subtitles_path(subtitle_path)

            global_fx = build_global_ffmpeg_filter(visual_style)
            if global_fx:
                logger.info("Applying visual style filters: %s", global_fx)

            if background_video_path and os.path.exists(background_video_path):
                base_video_input = ['-i', str(background_video_path)]
                audio_input_idx = 1
                video_filter = "[0:v]setpts=PTS-STARTPTS,tpad=stop=-1:stop_mode=clone"
                if global_fx:
                    video_filter += f",{global_fx}"
                video_filter += f",subtitles=filename='{escaped_subtitle_path}'[vout]"
            else:
                base_video_input = [
                    '-f', 'lavfi',
                    '-i', f'color=c={color}:s={res_x}x{res_y}:d={duration}:r={fps}'
                ]
                audio_input_idx = 1
                video_filter = "[0:v]"
                if global_fx:
                    video_filter += f",{global_fx}"
                video_filter += f",subtitles=filename='{escaped_subtitle_path}'[vout]"

            narration_for_video = audio_path
            temp_mixed_audio = None
            if background_music_path and os.path.exists(background_music_path):
                temp_mixed_audio = self._temp_output_path(
                    str(Path(audio_path).with_suffix(".bgm_mixed.wav"))
                )
                narration_for_video = self._mix_narration_with_background_music(
                    audio_path,
                    background_music_path,
                    duration,
                    temp_mixed_audio,
                )
                logger.info("Pre-mixed narration with background music: %s", narration_for_video)

            filter_complex = video_filter

            tmp_output_path = self._temp_output_path(output_path)
            if os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)

            cmd = [
                self.ffmpeg, '-y',
                '-hide_banner',
                '-nostats',
                '-progress', 'pipe:2',
                *base_video_input,
                '-i', narration_for_video,
                '-filter_complex', filter_complex,
                '-map', '[vout]', '-map', f'{audio_input_idx}:a',
                '-t', f'{duration:.3f}',
                *_video_encoder_args(quality_mode),
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-movflags', '+faststart',
                '-shortest',
                str(tmp_output_path)
            ]
            logger.info("Running FFmpeg video generation without expressions")
            self._run_ffmpeg(cmd)
            if not os.path.exists(tmp_output_path) or os.path.getsize(tmp_output_path) == 0:
                raise RuntimeError(f"Output video file was not created: {tmp_output_path}")
            self._validate_media_file(tmp_output_path)
            os.replace(tmp_output_path, output_path)
            if temp_mixed_audio and os.path.exists(temp_mixed_audio):
                try:
                    os.remove(temp_mixed_audio)
                except OSError:
                    pass
            logger.info(f"Successfully generated video at {output_path}")
            return output_path

            # Load audio
            logger.info(f"Loading audio from {audio_path}")
            try:
                audio_clip = AudioFileClip(audio_path)
                duration = audio_clip.duration
                logger.info(f"Audio loaded successfully. Duration: {duration:.2f}s")
            except Exception as e:
                logger.error(f"Failed to load audio: {str(e)}")
                raise

            # Create background
            if background_video_path and os.path.exists(background_video_path):
                logger.info(f"Using video background: {background_video_path}")
                try:
                    bg_clip = VideoFileClip(background_video_path)
                    bg_clip = bg_clip.set_duration(duration)
                    logger.debug(f"Background video loaded. Original duration: {bg_clip.duration:.2f}s, Set to: {duration:.2f}s")
                except Exception as e:
                    logger.error(f"Error loading background video: {str(e)}")
                    raise
            else:
                logger.info(f"Creating solid color background: {color}")
                try:
                    bg_clip = ColorClip(size=(res_x, res_y), color=color, duration=duration)
                    bg_clip = bg_clip.set_fps(fps)
                    logger.debug(f"Created {color} background. Size: {res_x}x{res_y}, Duration: {duration:.2f}s")
                except Exception as e:
                    logger.error(f"Error creating background: {str(e)}")
                    raise

            # Generate subtitles using MoviePy
            logger.info("Initializing subtitle generator...")
            try:
                # Validate subtitle effect
                valid_effects = [
                    'fade', 'pop', 'color', 'underline', 
                    'font_size', 'shadow', 'background', 'glow', 'wave'
                ]
                
                if subtitle_effect not in valid_effects:
                    logger.warning(f"Unknown subtitle effect '{subtitle_effect}'. Defaulting to 'fade'.")
                    subtitle_effect = 'fade'
                
                sg = SubtitleGenerator("", audio_path)
                logger.info("Loading subtitle phrases...")
                
                try:
                    with open(subtitle_path, 'r', encoding='utf-8') as f:
                        phrases = json.load(f)
                    logger.info(f"Loaded {len(phrases)} subtitle phrases")
                except json.JSONDecodeError as je:
                    logger.error(f"Failed to parse subtitle JSON: {str(je)}")
                    raise ValueError(f"Invalid subtitle file format: {str(je)}")
                
                # Generate animated subtitles
                logger.info(f"Generating animated subtitles with effect: {subtitle_effect}")
                try:
                    subtitle_clip = sg.generate_moviepy_video(
                        phrases,
                        "",  
                        resolution=resolution,
                        effect=subtitle_effect,
                        font_size=font_size,
                        font=font,
                        fps=fps
                    )
                    logger.info("Successfully generated subtitle clip")
                except Exception as e:
                    logger.error(f"Error in MoviePy subtitle generation: {str(e)}")
                    logger.debug(traceback.format_exc())
                    # Fallback to default effect if there's an error with the selected effect
                    if subtitle_effect != 'fade':
                        logger.info("Falling back to 'fade' effect")
                        subtitle_clip = sg.generate_moviepy_video(
                            phrases,
                            "",
                            resolution=resolution,
                            effect='fade',
                            font_size=font_size,
                            font=font,
                            fps=fps
                        )
                    else:
                        raise
            except Exception as e:
                logger.error(f"Error generating subtitles: {str(e)}\n{traceback.format_exc()}")
                raise

            # Composite everything
            logger.info("Compositing final video...")
            try:
                final_clip = CompositeVideoClip([bg_clip, subtitle_clip], size=(res_x, res_y))
                final_clip = final_clip.set_audio(audio_clip)
                logger.info("Final clip composition complete")
            except Exception as e:
                logger.error(f"Error during composition: {str(e)}")
                raise

            # Write to file
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Writing video to {output_path}...")
            
            try:
                final_clip.write_videofile(
                    output_path,
                    fps=fps,
                    codec="libx264",
                    audio_codec="aac",
                    preset="medium",
                    threads=4,
                    verbose=True,  # Enable verbose output for debugging
                    logger='bar' if logger.level <= logging.INFO else None,
                )
                logger.info(f"Successfully generated video at {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Error writing video file: {str(e)}")
                raise

        except Exception as e:
            error_msg = f"Error generating video with MoviePy: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            logger.error(f"Python version: {sys.version}")
            logger.error(f"MoviePy version: {moviepy.__version__ if 'moviepy' in globals() else 'Not available'}")
            raise RuntimeError(f"Video generation failed: {error_msg}")
        finally:
            logger.info("=" * 80)
            logger.info("VIDEO GENERATION COMPLETED")
            logger.info("=" * 80)

    def _convert_mkv_to_mp4(self, input_path: str, output_path: str) -> bool:
        """
        Convert an MKV file to MP4 format.

        Args:
            input_path: Path to the input MKV file
            output_path: Path where the output MP4 file will be saved

        Returns:
            bool: True if conversion was successful, False otherwise
        """
        try:
            logger.info(f"Converting {input_path} to MP4 format...")

            # Build ffmpeg command for MKV to MP4 conversion
            cmd = [
                self.ffmpeg,
                '-y',  # Overwrite output file if it exists
                '-i', input_path,  # Input file
                '-c:v', 'libx264',  # Video codec
                '-preset', 'medium',  # Encoding speed to compression ratio
                # Constant Rate Factor (lower = better quality, 23 is default)
                '-crf', '23',
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-c:a', 'libmp3lame',  # Audio codec
                '-b:a', '192k',  # Audio bitrate
                '-movflags', '+faststart',  # Enable streaming
                output_path
            ]

            logger.debug(f"Running command: {' '.join(cmd)}")

            # Run the command
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Verify output file was created
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                logger.error(f"Failed to create output file: {output_path}")
                return False

            logger.info(f"Successfully converted to MP4: {output_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error during MP4 conversion: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error during MP4 conversion: {str(e)}")
            return False

    def _render_expression_overlay_pass(
        self,
        base_video_path: str,
        overlay_specs: List[Dict[str, Any]],
        duration: float,
        quality_mode: str,
        output_path: str,
        visual_style: Optional[str] = None,
    ) -> str:
        """Apply a batch of expression overlays; reuses one -i per unique PNG."""
        if not overlay_specs:
            return base_video_path

        image_input_index: Dict[str, int] = {}
        overlay_inputs: List[str] = []
        filter_steps = ["[0:v]setpts=PTS-STARTPTS,tpad=stop=-1:stop_mode=clone[bg_video]"]
        overlay_filters: List[str] = []
        last_label = "[bg_video]"
        input_idx = 1

        for overlay_step, spec in enumerate(overlay_specs, 1):
            abs_path = spec["img_path"]
            if abs_path not in image_input_index:
                image_input_index[abs_path] = input_idx
                overlay_inputs.extend(["-loop", "1", "-i", abs_path])
                input_idx += 1
            img_label = f"[{image_input_index[abs_path]}:v]"
            expr_label = f"[expr{overlay_step}]"
            label = spec["label"]
            start = spec["start"]
            end = spec["end"]
            fade_duration = spec["fade_duration"]
            interval_duration = spec["interval_duration"]

            filter_steps.append(
                format_expression_filter_step(
                    img_label,
                    expr_label,
                    label,
                    fade_duration,
                    interval_duration,
                    start,
                    visual_style=visual_style,
                )
            )
            overlay_filter, last_label = format_expression_overlay(
                last_label,
                expr_label,
                label,
                fade_duration,
                interval_duration,
                start,
                end,
                overlay_step,
                visual_style=visual_style,
            )
            overlay_filters.append(overlay_filter)

        filter_steps += [f for f in overlay_filters if f]
        filter_steps.append(f"{last_label}format=yuv420p[vout]")
        filter_complex_clean = ";".join(filter_steps).replace("\n", "")

        use_filter_script = len(filter_complex_clean) > 48000 or len(filter_steps) > 35
        fscript_path = None
        if use_filter_script:
            with tempfile.NamedTemporaryFile(
                "w+", suffix=".ffmpeg", delete=False
            ) as fscript:
                fscript.write(filter_complex_clean)
                fscript_path = fscript.name
            filter_complex_arg = ["-filter_complex_script", fscript_path]
            logger.info(
                "Expression chunk filter script (%s steps, %s unique images)",
                len(filter_steps),
                len(image_input_index),
            )
        else:
            filter_complex_arg = ["-filter_complex", filter_complex_clean]
            logger.info(
                "Expression chunk filter (%s steps, %s unique images)",
                len(filter_steps),
                len(image_input_index),
            )

        tmp_output_path = self._temp_output_path(output_path)
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)

        cmd = [
            self.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(base_video_path),
            *overlay_inputs,
            *filter_complex_arg,
            "-map",
            "[vout]",
            "-t",
            f"{duration:.3f}",
            *_video_encoder_args(quality_mode),
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(tmp_output_path),
        ]
        self._run_ffmpeg(cmd)
        if fscript_path:
            try:
                os.remove(fscript_path)
            except OSError:
                pass
        os.replace(tmp_output_path, output_path)
        return output_path

    def _merge_consecutive_expressions(self, expressions):
        """
        Merge consecutive or overlapping expressions with the same label.
        Returns a new list of merged expressions.
        """
        if not expressions:
            return []
        merged = [expressions[0].copy()]
        for expr in expressions[1:]:
            prev = merged[-1]
            if (
                expr['expression'].lower() == prev['expression'].lower() and
                self._parse_time(expr['start']) <= self._parse_time(
                    prev['end']) + 0.05  # allow small gap/overlap
            ):
                # Extend the previous interval
                prev['end'] = expr['end']
                prev['text'] += ' ' + expr.get('text', '')
            else:
                merged.append(expr.copy())
        return merged

    def _merge_intervals(self, intervals):
        """
        Merge overlapping or adjacent intervals.
        Args:
            intervals: List of (start, end) tuples
        Returns:
            List of merged (start, end) tuples
        """
        if not intervals:
            return []
        # Sort by start time
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            prev = merged[-1]
            # If current overlaps or touches previous, merge
            if current[0] <= prev[1] + 0.01:  # allow tiny gap
                merged[-1] = (prev[0], max(prev[1], current[1]))
            else:
                merged.append(current)
        return merged

    def _merge_expression_intervals(
        self,
        intervals: List[Tuple[float, float]],
        gap_tolerance: float = EXPRESSION_MERGE_GAP_SECONDS,
    ) -> List[Dict[str, Any]]:
        """
        Merge subtitle-sized expression spans into longer holds.

        The expression mapper can emit many tiny lines with the same expression and
        small gaps between them. Treat those as one visual cue so the same PNG does
        not repeatedly re-enter for a continuous emotion.
        """
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda item: item[0])
        merged: List[Dict[str, Any]] = []
        for start, end in sorted_intervals:
            if end <= start:
                continue
            if not merged or start > merged[-1]["end"] + gap_tolerance:
                merged.append({"start": start, "end": end, "source_count": 1})
                continue

            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["source_count"] += 1

        return merged

    def generate_video_with_expressions(
        self,
        audio_path: str,
        subtitle_path: str,
        expressions_path: str,
        output_path: str,
        background_video_path: Optional[str] = None,
        resolution: str = '1080x1920',
        color: str = 'green',
        fps: int = 30,
        expr_img_dir: str = None,
        convert_to_mp4: bool = True,
        quality_mode: str = 'standard',
        background_music_path: Optional[str] = None,
        visual_style: Optional[str] = None,
    ) -> str:
        """
        Generate a video with audio, burned-in subtitles, and facial expression overlays.

        Args:
            audio_path: Path to the audio file
            subtitle_path: Path to the ASS subtitle file
            expressions_path: Path to the expressions JSON file
            output_path: Path where the output video will be saved
            resolution: Video resolution (default: '1080x1920')
            color: Background color (default: 'green')
            fps: Frames per second (default: 30)
            expr_img_dir: Directory containing expression images (default: app/static/expressions)

        Returns:
            Path to the generated video file
        """
        import json
        if expr_img_dir is None:
            expr_img_dir = os.path.join(os.path.dirname(
                __file__), '../static/expressions')
        try:
            quality_mode = (quality_mode or 'standard').lower()
            pro_mode = quality_mode == 'pro'
            logger.info(
                f"generate_video_with_expressions called with background_video_path: {background_video_path}")
            logger.info(f"Video quality mode: {quality_mode}")
            duration = self.get_audio_duration(audio_path)
            os.makedirs(os.path.dirname(
                os.path.abspath(output_path)), exist_ok=True)

            # Load expressions
            with open(expressions_path, 'r', encoding='utf-8') as f:
                expressions = json.load(f)
                
            logger.info(f"Loaded {len(expressions)} expressions from {expressions_path}")
            logger.debug(f"Expressions data: {json.dumps(expressions, indent=2, ensure_ascii=False)}")
            
            # Group intervals by character + expression (supports multi-character scripts)
            overlay_groups: Dict[str, Dict] = {}
            for expr in expressions:
                label = expr['expression'].lower()
                character = (expr.get('character') or NARRATOR_CHARACTER).lower()
                group_key = f"{character}::{label}"
                if group_key not in overlay_groups:
                    overlay_groups[group_key] = {
                        "character": character,
                        "label": label,
                        "intervals": [],
                    }
                start_time = self._parse_time(expr['start'])
                end_time = self._parse_time(expr['end'])
                overlay_groups[group_key]["intervals"].append((start_time, end_time))

            for group in overlay_groups.values():
                group["intervals"] = self._merge_expression_intervals(group["intervals"])

            logger.info(
                "Processed expression overlays: %s",
                [f"{g['character']}/{g['label']}" for g in overlay_groups.values()],
            )

            overlay_specs: List[Dict[str, Any]] = []
            logger.info(f"Expression images directory: {expr_img_dir}")
            available_images = os.listdir(expr_img_dir)
            logger.info(f"Available expression images: {available_images}")

            for group in overlay_groups.values():
                character = group["character"]
                label = group["label"]
                intervals = group["intervals"]
                context_samples = [
                    (expr.get("text") or "").strip()
                    for expr in expressions
                    if (expr.get("character") or NARRATOR_CHARACTER).lower() == character
                    and (expr.get("expression") or "neutral").lower() == label
                    and (expr.get("text") or "").strip()
                ]
                context_text = " ".join(context_samples[:4])
                img_path = resolve_expression_image(
                    character,
                    label,
                    fallback_dir=Path(expr_img_dir) if expr_img_dir else None,
                    context_text=context_text,
                )
                if not img_path:
                    logger.error(
                        "No expression PNG for '%s' in %s (expected {emotion}.png)",
                        label,
                        expr_img_dir,
                    )
                    continue
                logger.info(
                    "Using expression overlay %s -> %s",
                    label,
                    os.path.basename(img_path),
                )

                abs_img_path = os.path.abspath(img_path)
                for interval in intervals:
                    start = interval["start"]
                    end = interval["end"]
                    if end <= start:
                        logger.warning(
                            f"Skipping invalid interval for label '{label}': start={start}, end={end}"
                        )
                        continue
                    interval_duration = end - start
                    fade_duration = max(0.05, min(0.28, interval_duration / 3))
                    overlay_specs.append(
                        {
                            "img_path": abs_img_path,
                            "label": label,
                            "character": character,
                            "start": start,
                            "end": end,
                            "fade_duration": fade_duration,
                            "interval_duration": interval_duration,
                            "effect": resolve_expression_effect(label, visual_style),
                            "source_count": interval["source_count"],
                            "continuous_hold": interval["source_count"] > 1,
                        }
                    )
                    logger.info(
                        "Expression cue %s/%s effect=%s %.2f-%.2fs source_count=%s",
                        character,
                        label,
                        overlay_specs[-1]["effect"],
                        start,
                        end,
                        interval["source_count"],
                    )

            working_video = str(background_video_path) if background_video_path else None
            chunk_temp_paths: List[str] = []
            if overlay_specs and working_video:
                if EXPRESSION_RENDERER == "opencv":
                    expr_out = self._temp_output_path(
                        str(Path(output_path).with_suffix(".expr_opencv.mp4"))
                    )
                    chunk_temp_paths.append(expr_out)
                    logger.info(
                        "Rendering %s expression overlays in one OpenCV pass",
                        len(overlay_specs),
                    )
                    working_video = render_expression_overlays_opencv(
                        working_video,
                        overlay_specs,
                        expr_out,
                        duration,
                        fps=fps,
                        visual_style=visual_style,
                    )
                else:
                    chunks = [
                        overlay_specs[i : i + EXPRESSION_OVERLAY_CHUNK_SIZE]
                        for i in range(0, len(overlay_specs), EXPRESSION_OVERLAY_CHUNK_SIZE)
                    ]
                    logger.info(
                        "Rendering %s expression overlays in %s FFmpeg chunk(s) (max %s per pass)",
                        len(overlay_specs),
                        len(chunks),
                        EXPRESSION_OVERLAY_CHUNK_SIZE,
                    )
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_out = self._temp_output_path(
                            str(
                                Path(output_path).with_suffix(
                                    f".expr_chunk{chunk_idx}.mp4"
                                )
                            )
                        )
                        chunk_temp_paths.append(chunk_out)
                        working_video = self._render_expression_overlay_pass(
                            working_video,
                            chunk,
                            duration,
                            quality_mode,
                            chunk_out,
                            visual_style=visual_style,
                        )
                        if chunk_idx > 0 and chunk_temp_paths[chunk_idx - 1] != working_video:
                            try:
                                os.remove(chunk_temp_paths[chunk_idx - 1])
                            except OSError:
                                pass
            elif overlay_specs and not working_video:
                logger.warning("Expression overlays requested but no background video path")

            narration_for_video = audio_path
            temp_mixed_audio = None
            if background_music_path and os.path.exists(background_music_path):
                temp_mixed_audio = self._temp_output_path(
                    str(Path(audio_path).with_suffix(".bgm_mixed.wav"))
                )
                narration_for_video = self._mix_narration_with_background_music(
                    audio_path,
                    background_music_path,
                    duration,
                    temp_mixed_audio,
                )
                logger.info("Pre-mixed narration with background music: %s", narration_for_video)

            escaped_subtitle_path = _escape_ffmpeg_subtitles_path(subtitle_path)
            global_fx = build_global_ffmpeg_filter(visual_style)
            video_filter = "[0:v]setpts=PTS-STARTPTS,tpad=stop=-1:stop_mode=clone"
            if global_fx:
                logger.info("Applying visual style filters: %s", global_fx)
                video_filter += f",{global_fx}"
            video_filter += f",subtitles=filename='{escaped_subtitle_path}'[vout]"

            tmp_output_path = self._temp_output_path(output_path)
            if os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)

            if working_video and os.path.exists(working_video):
                base_video_input = ["-i", str(working_video)]
            else:
                res_x, res_y = map(int, resolution.lower().split("x"))
                base_video_input = [
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c={color}:s={res_x}x{res_y}:d={duration}:r={fps}",
                ]
                video_filter = f"[0:v]"
                if global_fx:
                    video_filter += f",{global_fx}"
                video_filter += f",subtitles=filename='{escaped_subtitle_path}'[vout]"

            cmd = [
                self.ffmpeg,
                "-y",
                "-hide_banner",
                "-nostats",
                "-progress",
                "pipe:2",
                *base_video_input,
                "-i",
                narration_for_video,
                "-filter_complex",
                video_filter,
                "-map",
                "[vout]",
                "-map",
                "1:a",
                "-t",
                f"{duration:.3f}",
                *_video_encoder_args(quality_mode),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-shortest",
                str(tmp_output_path),
            ]
            logger.info("Final encode: subtitles + audio on %s", working_video or "color background")
            self._run_ffmpeg(cmd)

            for chunk_path in chunk_temp_paths:
                if chunk_path != working_video and os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass
            if temp_mixed_audio and os.path.exists(temp_mixed_audio):
                try:
                    os.remove(temp_mixed_audio)
                except OSError:
                    pass

            # Verify the output file was created
            if not os.path.exists(tmp_output_path):
                logger.error(f"Output video file was not created: {tmp_output_path}")
                raise FileNotFoundError(f"Output video file was not created: {tmp_output_path}")
            
            output_size = os.path.getsize(tmp_output_path)
            if output_size == 0:
                logger.error(f"Output video file is empty: {tmp_output_path}")
                raise RuntimeError(f"Output video file is empty: {tmp_output_path}")
            self._validate_media_file(tmp_output_path)
            os.replace(tmp_output_path, output_path)
                
            # Convert to MP4 if needed
            if convert_to_mp4 and output_path.lower().endswith('.mkv'):
                mp4_path = output_path.rsplit('.', 1)[0] + '.mp4'
                if self._convert_mkv_to_mp4(output_path, mp4_path):
                    try:
                        os.remove(output_path)
                        logger.info(f"Removed temporary MKV file: {output_path}")
                    except OSError as e:
                        logger.warning(f"Failed to remove temporary MKV file: {e}")
                    output_path = mp4_path
                else:
                    logger.warning("Failed to convert MKV to MP4, keeping MKV file")
            
            logger.info(f"Successfully generated video with expressions: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Video generation failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Error generating video with expressions: {str(e)}")
            raise

    def _parse_time(self, t):
        # Accepts H:MM:SS.ss or S.ss
        if isinstance(t, (int, float)):
            return float(t)
        if ':' in t:
            h, m, s = t.split(':')
            return float(h)*3600 + float(m)*60 + float(s)
        return float(t)
