import os
import logging
import argparse
import traceback
import json
from pathlib import Path
import sys

# Ensure the project root is in the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from app.utils.slides.generate_slideshow import main as generate_slideshow_video
from app.utils.video.video_generator import VideoGenerator
from app.utils.audio.audio_processor import get_audio_duration
from app.config import (
    BACKGROUND_MUSIC_DIR,
    ENABLE_BACKGROUND_MUSIC,
    UPLOADS_DIR,
    EXPRESSIONS_DIR,
    COMPILED_VIDEO_DIR,
    get_video_profile_config,
    normalize_video_profile,
    normalize_visual_style,
)
from app.utils.slides.image_slides_upload import slides_images_dir
from app.utils.audio.transition_sfx import load_transition_events, mix_transition_sfx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUPPORTED_MUSIC_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def pick_background_music(music_path: str = None) -> str:
    if music_path:
        path = Path(music_path)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Background music file not found: {music_path}")

    if not ENABLE_BACKGROUND_MUSIC:
        return None

    if not BACKGROUND_MUSIC_DIR.exists():
        return None

    music_files = sorted(
        path for path in BACKGROUND_MUSIC_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_MUSIC_EXTENSIONS
    )
    if not music_files:
        return None
    return str(music_files[0])


def generate_final_video(
    audio_id: str,
    slides_json_path: str,
    subtitle_path: str,
    images_dir: str = None,
    expressions_path: str = None,
    output_filename: str = None,
    generate_slides: bool = True,  # New parameter to control slides generation
    blur_amount: int = 5,  # Controls transition blur (0 = no blur, higher values = softer crossfades)
    quality_mode: str = "pro",
    background_music_path: str = None,
    video_profile: str = "short_vertical",
    visual_style: str = "clean_pro",
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
        quality_mode = (quality_mode or "standard").lower()
        pro_mode = quality_mode == "pro"
        video_profile = normalize_video_profile(video_profile)
        visual_style = normalize_visual_style(visual_style)
        profile_config = get_video_profile_config(video_profile)
        video_resolution = profile_config["video_resolution"]
        if not images_dir:
            images_dir = str(slides_images_dir(audio_id))
        logger.info("Using slide images directory: %s", images_dir)
        logger.info("Using video quality mode: %s", quality_mode)
        logger.info("Using video profile: %s (%sx%s)", video_profile, video_resolution[0], video_resolution[1])
        logger.info("Using visual style: %s", visual_style)
        selected_music_path = pick_background_music(background_music_path)
        logger.info("Background music: %s", selected_music_path or "disabled")

        # 1. Get audio duration to sync the slideshow
        audio_path = UPLOADS_DIR / f"{audio_id}"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        total_audio_duration = get_audio_duration(str(audio_path))
        logger.info(f"Total audio duration: {total_audio_duration:.2f}s")

        # 2. Generate the slideshow video (as a temporary file)
        slideshow_output_path = COMPILED_VIDEO_DIR / f"slideshow_temp_{Path(audio_id).stem}_{video_profile}.mp4"
        
        # Only generate slides if needed or if file doesn't exist
        if generate_slides or not slideshow_output_path.exists():
            logger.info(f"Generating slideshow video at: {slideshow_output_path}")
            generate_slideshow_video(
                json_path=slides_json_path,
                image_dir=images_dir,
                output_path=str(slideshow_output_path),
                total_duration=total_audio_duration,
                resolution=video_resolution,
                blur_amount=3 if pro_mode else blur_amount,
                transition_type='pro' if pro_mode else 'auto',
                transition_duration=0.38 if pro_mode else 0.5,
                quality_mode=quality_mode,
                visual_style=visual_style,
            )
            logger.info("Slideshow video generated successfully.")
        else:
            logger.info("Using existing slideshow video")

        narration_path = str(audio_path)
        transition_events = load_transition_events(slides_json_path)
        if transition_events:
            mixed_path = COMPILED_VIDEO_DIR / f"narration_sfx_{Path(audio_id).stem}.wav"
            narration_path = mix_transition_sfx(
                narration_path,
                transition_events,
                str(mixed_path),
                duration=total_audio_duration,
                visual_style=visual_style,
            )
            if narration_path != str(audio_path):
                audio_path = Path(narration_path)

        # 3. Initialize the VideoGenerator
        video_generator = VideoGenerator()

        # 4. Define the final output path
        if not output_filename:
            base_name = Path(audio_id).stem
            output_filename = f"final_video_{base_name}_{video_profile}.mp4"

        final_output_path = COMPILED_VIDEO_DIR / output_filename

        # 5. Generate the final video with audio, subtitles, and expressions
        logger.info("Generating final video with audio, subtitles, and expressions...")
        if expressions_path and Path(expressions_path).exists():
            logger.info(f"Expressions file found at: {expressions_path}")
            logger.info(f"Expressions directory: {EXPRESSIONS_DIR}")
            logger.info(f"Background video path: {slideshow_output_path}")
            logger.info(f"Output path: {final_output_path}")
            
            # Log expressions file content for debugging
            try:
                with open(expressions_path, 'r', encoding='utf-8') as f:
                    expressions_data = json.load(f)
                    logger.info(f"Loaded {len(expressions_data)} expressions from {expressions_path}")
                    logger.debug(f"Expressions data: {json.dumps(expressions_data, indent=2, ensure_ascii=False)}")
            except Exception as e:
                logger.error(f"Error reading expressions file: {e}", exc_info=True)
            
            # Log available expression images
            try:
                if EXPRESSIONS_DIR.exists():
                    available_images = list(EXPRESSIONS_DIR.glob('*.png'))
                    logger.info(f"Found {len(available_images)} expression images in {EXPRESSIONS_DIR}")
                    logger.debug(f"Available expression images: {[img.name for img in available_images]}")
                else:
                    logger.error(f"Expressions directory not found: {EXPRESSIONS_DIR}")
            except Exception as e:
                logger.error(f"Error listing expression images: {e}", exc_info=True)
            
            # Generate video with expressions
            logger.info("Starting video generation with expressions...")
            final_video_path = video_generator.generate_video_with_expressions(
                audio_path=str(audio_path),
                subtitle_path=subtitle_path,
                expressions_path=expressions_path,
                output_path=str(final_output_path),
                background_video_path=str(slideshow_output_path),
                expr_img_dir=str(EXPRESSIONS_DIR),
                quality_mode=quality_mode,
                background_music_path=selected_music_path,
                resolution=f"{video_resolution[0]}x{video_resolution[1]}",
                visual_style=visual_style,
            )
        else:
            logger.info("No expressions file found, generating video without expressions.")
            final_video_path = video_generator.generate_video(
                audio_path=str(audio_path),
                subtitle_path=subtitle_path,
                output_path=str(final_output_path),
                background_video_path=str(slideshow_output_path),
                quality_mode=quality_mode,
                background_music_path=selected_music_path,
                resolution=f"{video_resolution[0]}x{video_resolution[1]}",
                visual_style=visual_style,
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
    parser.add_argument(
        "--images_dir",
        default=None,
        help="Directory containing slide images (default: output/image_slides/<audio_stem>/)",
    )
    parser.add_argument("--expressions_file", default=None, help="Path to the expressions JSON file (optional).")
    parser.add_argument("--output_filename", default=None, help="Name for the final output video file (optional).")
    parser.add_argument("--blur", type=int, default=5, help="Amount of blur to apply to slides (0 = no blur, 5 = default, higher values = more blur, typical range: 0-10).")
    parser.add_argument("--video_profile", default="short_vertical", choices=["short_vertical", "long_youtube"], help="Video profile to render.")

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
            blur_amount=args.blur,
            video_profile=args.video_profile,
        )
        print(f"Successfully created final video: {final_path}")
    except Exception as e:
        print(f"Failed to generate video. Error: {e}")
