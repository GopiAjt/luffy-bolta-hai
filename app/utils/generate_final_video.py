import os
import logging
import argparse
import traceback
from pathlib import Path
import sys

# Ensure the project root is in the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.utils.generate_slideshow import main as generate_slideshow_video
from app.utils.video_generator import VideoGenerator
from app.utils.audio_processor import get_audio_duration
from config.config import (
    UPLOADS_DIR,
    IMAGE_SLIDES_DIR,
    EXPRESSIONS_DIR,
    VIDEO_RESOLUTION,
    COMPILED_VIDEO_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_final_video(
    audio_id: str,
    slides_json_path: str,
    subtitle_path: str,
    images_dir: str = str(IMAGE_SLIDES_DIR),
    expressions_path: str = None,
    output_filename: str = None,
):
    """
    Generates the final video by first creating a slideshow and then adding audio, subtitles, and expressions.

    Args:
        audio_id (str): The ID of the audio file in the UPLOADS_DIR.
        slides_json_path (str): Path to the JSON file containing slide information.
        subtitle_path (str): Path to the .ass subtitle file.
        images_dir (str, optional): Path to the directory containing the slide images.
        expressions_path (str, optional): Path to the expressions JSON file. Defaults to None.
        output_filename (str, optional): The name of the final output file. Defaults to None.

    Returns:
        str: The path to the final generated video.
    """
    slideshow_output_path = None  # Initialize to None
    try:
        # 1. Get audio duration to sync the slideshow
        audio_path = UPLOADS_DIR / f"{audio_id}"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        total_audio_duration = get_audio_duration(str(audio_path))
        logger.info(f"Total audio duration: {total_audio_duration:.2f}s")

        # 2. Generate the slideshow video (as a temporary file)
        slideshow_output_path = COMPILED_VIDEO_DIR / f"slideshow_temp_{Path(audio_id).stem}.mp4"
        logger.info(f"Generating slideshow video at: {slideshow_output_path}")
        generate_slideshow_video(
            json_path=slides_json_path,
            image_dir=images_dir,
            output_path=str(slideshow_output_path),
            total_duration=total_audio_duration,
            resolution=VIDEO_RESOLUTION,
        )
        logger.info("Slideshow video generated successfully.")

        # 3. Initialize the VideoGenerator
        video_generator = VideoGenerator()

        # 4. Define the final output path
        if not output_filename:
            base_name = Path(audio_id).stem
            output_filename = f"final_video_{base_name}.mp4"

        final_output_path = COMPILED_VIDEO_DIR / output_filename

        # 5. Generate the final video with audio, subtitles, and expressions
        logger.info("Generating final video with audio, subtitles, and expressions...")
        if expressions_path and Path(expressions_path).exists():
            logger.info("Expressions file found, generating video with expressions.")
            final_video_path = video_generator.generate_video_with_expressions(
                audio_path=str(audio_path),
                subtitle_path=subtitle_path,
                expressions_path=expressions_path,
                output_path=str(final_output_path),
                background_video_path=str(slideshow_output_path),
                expr_img_dir=str(EXPRESSIONS_DIR),
            )
        else:
            logger.info("No expressions file found, generating video without expressions.")
            final_video_path = video_generator.generate_video(
                audio_path=str(audio_path),
                subtitle_path=subtitle_path,
                output_path=str(final_output_path),
                background_video_path=str(slideshow_output_path),
            )

        logger.info(f"Final video generated successfully at: {final_video_path}")
        return final_video_path

    except Exception as e:
        logger.error(f"An error occurred during final video generation: {e}")
        traceback.print_exc()
        raise
    finally:
        # 6. Clean up the temporary slideshow video
        if slideshow_output_path and slideshow_output_path.exists():
            try:
                os.remove(slideshow_output_path)
                logger.info(f"Cleaned up temporary slideshow file: {slideshow_output_path}")
            except OSError as e:
                logger.error(f"Error removing temporary slideshow file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a complete video with slideshow, audio, subtitles, and expressions.")
    parser.add_argument("--audio_id", required=True, help="Filename of the audio file in the uploads directory (e.g., 'my_audio.wav').")
    parser.add_argument("--slides_json", required=True, help="Path to the slides JSON file.")
    parser.add_argument("--subtitle_file", required=True, help="Path to the .ass subtitle file.")
    parser.add_argument("--images_dir", default=str(IMAGE_SLIDES_DIR), help=f"Directory containing slide images. Defaults to {IMAGE_SLIDES_DIR}")
    parser.add_argument("--expressions_file", default=None, help="Path to the expressions JSON file (optional).")
    parser.add_argument("--output_filename", default=None, help="Name for the final output video file (optional).")

    args = parser.parse_args()

    try:
        # Construct full paths for files that might be relative
        slides_json_full_path = Path(args.slides_json)
        if not slides_json_full_path.is_absolute():
            slides_json_full_path = UPLOADS_DIR / slides_json_full_path.name

        subtitle_full_path = Path(args.subtitle_file)
        if not subtitle_full_path.is_absolute():
            subtitle_full_path = UPLOADS_DIR / subtitle_full_path.name

        expressions_full_path = None
        if args.expressions_file:
            expressions_full_path = Path(args.expressions_file)
            if not expressions_full_path.is_absolute():
                expressions_full_path = UPLOADS_DIR / expressions_full_path.name

        final_path = generate_final_video(
            audio_id=args.audio_id,
            slides_json_path=str(slides_json_full_path),
            images_dir=args.images_dir,
            subtitle_path=str(subtitle_full_path),
            expressions_path=str(expressions_full_path) if expressions_full_path else None,
            output_filename=args.output_filename,
        )
        print(f"Successfully created final video: {final_path}")
    except Exception as e:
        print(f"Failed to generate video. Error: {e}")
