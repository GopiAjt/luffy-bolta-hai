import os
import subprocess
import logging
from typing import Optional
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


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
        resolution: str = '1080x1920',  # Changed from '1920x1080' to '1080x1920' for 9:16
        color: str = 'green',
        fps: int = 30,
        convert_to_mp4: bool = True
    ) -> str:
        """
        Generate a video with audio and burned-in subtitles on a colored background.

        Args:
            audio_path: Path to the audio file
            subtitle_path: Path to the ASS subtitle file
            output_path: Path where the output video will be saved
            resolution: Video resolution (default: '1920x1080')
            color: Background color (default: 'green')
            fps: Frames per second (default: 30)

        Returns:
            Path to the generated video file
        """
        try:
            # Get audio duration
            duration = self.get_audio_duration(audio_path)
            os.makedirs(os.path.dirname(
                os.path.abspath(output_path)), exist_ok=True)

            # Build ffmpeg command for burning in subtitles and explicit stream mapping
            if background_video_path:
                cmd = [
                    self.ffmpeg,
                    '-y',
                    '-i', str(background_video_path), # Input 0: background video
                    '-i', str(audio_path), # Input 1: audio
                    '-vf', f"subtitles='{subtitle_path}'",  # Burn in subtitles
                    '-map', '0:v',  # Explicitly map video from background video
                    '-map', '1:a',  # Explicitly map audio from audio file
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'libmp3lame' if not str(
                        audio_path).lower().endswith('.mp3') else 'copy',
                    '-b:a', '192k',
                    '-shortest',  # Stop encoding when the shortest input ends
                    str(output_path)
                ]
            else:
                cmd = [
                    self.ffmpeg,
                    '-y',
                    '-f', 'lavfi',
                    '-i', f'color=c={color}:s={resolution}:d={duration}:r={fps}',
                    '-i', str(audio_path),
                    '-vf', f"subtitles='{subtitle_path}'",  # Burn in subtitles
                    '-map', '0:v',  # Explicitly map video from color
                    '-map', '1:a',  # Explicitly map audio from audio file
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'libmp3lame' if not str(
                        audio_path).lower().endswith('.mp3') else 'copy',
                    '-b:a', '192k',
                    '-shortest',  # Stop encoding when the shortest input ends
                    str(output_path)
                ]

            logger.info(f"Generating video with command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"Successfully generated video at {output_path}")

            if convert_to_mp4 and output_path.lower().endswith('.mkv'):
                mp4_path = output_path.rsplit('.', 1)[0] + '.mp4'
                if self._convert_mkv_to_mp4(output_path, mp4_path):
                    try:
                        os.remove(output_path)
                        logger.info(
                            f"Removed temporary MKV file: {output_path}")
                    except OSError as e:
                        logger.warning(
                            f"Failed to remove temporary MKV file: {e}")
                    output_path = mp4_path
                else:
                    logger.warning(
                        "Failed to convert MKV to MP4, keeping MKV file")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Video generation failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            raise

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
        convert_to_mp4: bool = True
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
            logger.info(f"generate_video_with_expressions called with background_video_path: {background_video_path}")
            duration = self.get_audio_duration(audio_path)
            os.makedirs(os.path.dirname(
                os.path.abspath(output_path)), exist_ok=True)

            # Load expressions
            with open(expressions_path, 'r', encoding='utf-8') as f:
                expressions = json.load(f)
            # Group intervals by label
            label_intervals = {}
            for expr in expressions:
                label = expr['expression'].lower()
                if label not in label_intervals:
                    label_intervals[label] = []
                label_intervals[label].append(
                    (self._parse_time(expr['start']), self._parse_time(expr['end'])))
            # Merge intervals for each label
            for label in label_intervals:
                label_intervals[label] = self._merge_intervals(
                    label_intervals[label])

            # Prepare ffmpeg inputs and overlay filters (plain overlays, no fade)
            overlay_inputs = []
            overlay_filters = []
            overlay_step = 1

            # Determine the base video input and initial filter chain
            if background_video_path:
                # Input 0 is the background video
                base_video_input = ['-i', str(background_video_path)]
                # Initial filter to set PTS for the background video
                filter_steps = [f"[0:v]setpts=PTS-STARTPTS[bg_video]"]
                last_label = '[bg_video]'
                input_idx = 1 # First overlay image will be input 1
            else:
                # Input 0 is the color background generated by lavfi
                base_video_input = ['-f', 'lavfi', '-i', f'color=c={color}:s={resolution}:d={duration}:r={fps}']
                filter_steps = [f"color=c={color}:s={resolution}:d={duration}:r={fps}[bg]"]
                last_label = '[bg]'
                input_idx = 1 # First overlay image will be input 1

            # For each label and interval, add a separate image input and overlay (no fade/transition)
            for label, intervals in label_intervals.items():
                img_path = os.path.join(expr_img_dir, f"{label}.png")
                if not os.path.exists(img_path):
                    logger.warning(
                        f"Image for expression '{label}' not found: {img_path}")
                    continue
                for fade_idx, (start, end) in enumerate(intervals):
                    if end <= start:
                        logger.warning(
                            f"Skipping invalid interval for label '{label}': start={start}, end={end}")
                        continue
                    overlay_inputs.append('-i')
                    overlay_inputs.append(img_path)
                    img_label = f"[{input_idx}:v]"
                    # No fade/transition, just overlay with enable
                    enable_expr = f"between(t,{start},{end})"
                    overlay_filters.append(
                        f"{last_label}{img_label} overlay=x=(W-w)/2:y=H-h-200:enable='{enable_expr}'[bg{overlay_step}]"
                    )
                    last_label = f"[bg{overlay_step}]"
                    overlay_step += 1
                    input_idx += 1

            # Compose filter_complex without empty filter chains
            filter_steps += [f for f in overlay_filters if f]
            filter_steps.append(
                f"{last_label}subtitles='{subtitle_path}'[vout]")
            filter_complex = ';'.join(filter_steps)
            # Clean filtergraph: remove newlines only (do not escape double quotes)
            filter_complex_clean = filter_complex.replace('\n', '')

            # Write filter_complex to a temporary file as-is (preserve newlines and formatting)
            with tempfile.NamedTemporaryFile('w+', suffix='.ffmpeg', delete=False) as fscript:
                fscript.write(filter_complex)
                fscript_path = fscript.name

            # Build ffmpeg command using -filter_complex <string> (pass as string, not file)
            # audio_input_idx will be the index of the audio input stream
            # If background_video_path is used, it's input 0, then overlay images, then audio
            # If color background is used, it's input 0 (lavfi color), then overlay images, then audio
            audio_input_idx = input_idx # input_idx is already incremented to the next available input index

            cmd = [
                self.ffmpeg, '-y',
                *base_video_input, # Use the determined base video input
                *overlay_inputs,
                '-i', audio_path,
                '-filter_complex', filter_complex_clean,  # Pass cleaned filtergraph as string
                '-map', '[vout]', '-map', f'{audio_input_idx}:a',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p',
                '-c:a', 'libmp3lame' if not str(
                    audio_path).lower().endswith('.mp3') else 'copy',
                '-b:a', '192k', '-shortest', str(output_path)
            ]
            logger.info(
                f"Generating video with expressions: {' '.join(map(str, cmd))}")
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Clean up temp filter script
            try:
                os.remove(fscript_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp filter script: {e}")
            if convert_to_mp4 and output_path.lower().endswith('.mkv'):
                mp4_path = output_path.rsplit('.', 1)[0] + '.mp4'
                if self._convert_mkv_to_mp4(output_path, mp4_path):
                    try:
                        os.remove(output_path)
                        logger.info(
                            f"Removed temporary MKV file: {output_path}")
                    except OSError as e:
                        logger.warning(
                            f"Failed to remove temporary MKV file: {e}")
                    output_path = mp4_path
                else:
                    logger.warning(
                        "Failed to convert MKV to MP4, keeping MKV file")
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
