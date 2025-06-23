import os
import subprocess
import logging
from typing import Optional
from pathlib import Path

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
        resolution: str = '1920x1080',
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
