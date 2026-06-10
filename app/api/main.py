from flask import Flask, request, jsonify, send_from_directory, send_file, url_for, Response
from flask_cors import CORS
import os
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
import uuid
from app.utils.text.script_generator import generate_script
from app.utils.audio.audio_processor import save_audio_file, convert_to_wav, get_audio_duration, create_uploads_dir
from app.utils.text.subtitle_generator import SubtitleGenerator
from app.utils.slides.image_slides import generate_image_slides
from app.utils.slides.image_slides_upload import (
    apply_vivre_asset_to_slide,
    auto_resolve_slide_assets,
    audio_stem,
    build_slides_response,
    load_slides,
    save_slide_upload,
    slides_images_dir,
    slides_json_path_for_audio,
    slides_upload_status,
)
from app.utils.video.generate_final_video import generate_final_video
from app.utils.audio.tts_generator import generate_voiceover
from app.utils.output_cleanup import cleanup_output, get_output_usage
from app.utils.manga.manga_pdf_processor import (
    create_pdf_slides,
    fetch_ohara_context,
    load_manga_pdf_manifest,
    load_manga_session,
    process_manga_pdf,
    score_text_quality,
    update_manga_session,
)
import re
import json
import glob
from app.utils.expressions.expression_assets import (
    ensure_vivre_card_index,
    suggest_vivre_assets,
    vivre_asset_path_from_relative,
    vivre_card_status,
)
from app.config import (
    UPLOADS_DIR,
    IMAGE_SLIDES_DIR,
    EXPRESSIONS_DIR,
    VIVRE_CARD_ASSETS_DIR,
    VIDEO_BACKGROUND_COLOR,
    MAX_PDF_SIZE,
    HOST,
    PORT,
    DEBUG,
    get_video_profile_config,
    normalize_video_profile,
    normalize_visual_style,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
# Configure Flask app to serve static files
app = Flask(__name__, 
            static_folder='../../app/static',  # Path to static files
            static_url_path='/static')  # URL prefix for static files
CORS(app)


def _index_vivre_cards_on_startup():
    try:
        count = ensure_vivre_card_index()
        if count:
            logger.info("Vivre Card assets ready: %s PNGs indexed under %s", count, VIVRE_CARD_ASSETS_DIR)
    except Exception as exc:
        logger.warning("Vivre Card index skipped: %s", exc)


_index_vivre_cards_on_startup()


def _build_pdf_script_context(manifest: dict) -> dict:
    context_text = manifest.get('extracted_text', '')
    text_quality = manifest.get('text_quality') or {}
    if not text_quality and context_text.strip():
        text_quality = score_text_quality(context_text, 70)

    chapter_number = manifest.get('chapter_number')
    ohara_context = manifest.get('ohara_context') or None
    context_sources = list(manifest.get('context_sources') or [])
    warnings = list(manifest.get('warnings') or [])

    if chapter_number and not ohara_context:
        ohara_context, ohara_warning = fetch_ohara_context(chapter_number)
        if ohara_context:
            context_sources.append({
                'source': ohara_context.get('source'),
                'title': ohara_context.get('title'),
                'url': ohara_context.get('url'),
                'chapter_number': ohara_context.get('chapter_number'),
                'fetched_at': ohara_context.get('fetched_at'),
                'quality': ohara_context.get('quality'),
            })
        elif ohara_warning:
            warnings.append(ohara_warning)

    has_usable_pdf_text = bool(context_text.strip()) and bool(text_quality.get('usable'))
    has_usable_ohara_context = bool(ohara_context and ohara_context.get('text'))

    return {
        'context_text': context_text if has_usable_pdf_text else '',
        'chapter_number': chapter_number,
        'ohara_context': ohara_context,
        'context_sources': context_sources,
        'text_quality': text_quality,
        'warnings': warnings,
        'has_usable_context': has_usable_pdf_text or has_usable_ohara_context,
    }


def _generate_pdf_script_from_manifest(
    manifest: dict,
    topic: Optional[str] = None,
    video_profile: str = 'short_vertical',
) -> dict:
    context = _build_pdf_script_context(manifest)
    if not context['has_usable_context']:
        raise ValueError(
            'Could not find reliable text for this PDF. OCR quality is too low and no matching '
            'Library of Ohara context was found. Provide a manual topic/summary or try a cleaner PDF.'
        )

    result = generate_script(
        topic_override=topic,
        language='english',
        context_text=context['context_text'],
        chapter_number=context['chapter_number'],
        ohara_context=context['ohara_context'].get('text', '') if context['ohara_context'] else '',
        context_sources=context['context_sources'],
        video_profile=video_profile,
    )
    return {
        'title': result.get('title', ''),
        'script': result.get('script', ''),
        'description': result.get('description', ''),
        'hashtags': result.get('hashtags', ''),
        'resolved_topic': result.get('resolved_topic', ''),
        'quality_warnings': result.get('quality_warnings', []),
        'chapter_number': context['chapter_number'],
        'text_quality': context['text_quality'],
        'context_sources': context['context_sources'],
        'warnings': context['warnings'],
        'video_profile': normalize_video_profile(video_profile),
    }


def _generate_subtitle_assets(
    audio_id: str,
    script: Optional[str],
    subtitle_style: str = 'pro',
    video_profile: str = 'short_vertical',
    visual_style: Optional[str] = None,
) -> dict:
    from app.utils.video.visual_effects import subtitle_style_for_visual_style

    if visual_style:
        subtitle_style = subtitle_style_for_visual_style(visual_style, subtitle_style)
    audio_path = os.path.join(UPLOADS_DIR, audio_id)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'Audio file not found: {audio_id}')

    subtitle_generator = SubtitleGenerator(audio_path, script, style=subtitle_style)
    timestamps = subtitle_generator.generate_timestamps()
    if not timestamps:
        raise RuntimeError('Failed to generate subtitles: No timestamps generated')

    phrases = subtitle_generator.group_words_into_phrases(timestamps)
    if not phrases:
        raise RuntimeError('Failed to generate subtitles: No phrases generated')

    ass_path = os.path.join(UPLOADS_DIR, f"{os.path.splitext(audio_id)[0]}.ass")
    profile = get_video_profile_config(video_profile)
    subtitle_generator.generate_ass_file(
        phrases,
        ass_path,
        resolution=profile['subtitle_resolution'],
    )
    if not os.path.exists(ass_path) or os.path.getsize(ass_path) < 100:
        raise RuntimeError('Failed to generate subtitles: ASS file not generated correctly')

    expressions = None
    expr_json_path = None
    try:
        from app.utils.expressions.expression_mapper import gemini_expression_mapping_from_ass, parse_ass_file
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        expr_json_path = ass_path.rsplit('.', 1)[0] + '.expressions.json'
        if gemini_api_key:
            expressions = gemini_expression_mapping_from_ass(ass_path, gemini_api_key)
        else:
            expressions = parse_ass_file(ass_path)
        with open(expr_json_path, 'w', encoding='utf-8') as f:
            json.dump(expressions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Expression mapping failed: {e}")
        expr_json_path = None
        expressions = None

    return {
        'ass_path': ass_path,
        'ass_file': os.path.basename(ass_path),
        'expressions_path': expr_json_path,
        'expressions_file': os.path.basename(expr_json_path) if expr_json_path else None,
        'expressions': expressions,
        'phrases': phrases,
        'video_profile': normalize_video_profile(video_profile),
    }

# Serve static files


@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('../static/js', filename)


@app.route('/')
def index():
    return send_from_directory('../static', 'index.html')


@app.route('/api/v1/audio/<path:filename>')
def serve_audio(filename):
    """
    Serve uploaded audio files.
    
    Args:
        filename: Name of the audio file to serve
        
    Returns:
        Audio file with appropriate headers for streaming
    """
    try:
        # Security check to prevent directory traversal
        if '..' in filename or filename.startswith('/'):
            return jsonify({'error': 'Invalid file path'}), 400
            
        # Get the full path to the file
        file_path = UPLOADS_DIR / filename
        
        # Check if file exists
        if not file_path.exists():
            logger.error(f"Audio file not found: {filename}")
            return jsonify({'error': 'File not found'}), 404
            
        # Determine MIME type based on file extension
        mime_type = 'audio/wav'  # default
        if filename.lower().endswith('.mp3'):
            mime_type = 'audio/mpeg'
            
        # Set appropriate headers for audio streaming
        response = send_file(
            file_path,
            mimetype=mime_type,
            as_attachment=False,
            download_name=filename
        )
        
        # Enable range requests for seeking in the audio player
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'no-cache'
        
        return response
        
    except Exception as e:
        logger.error(f"Error serving audio file {filename}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/v1/latest-audio-file', methods=['GET'])
def get_latest_audio_file():
    """Return the newest audio file in the upload/output directory."""
    try:
        uploads_dir = Path(UPLOADS_DIR)
        audio_files = [
            path
            for path in uploads_dir.glob("*")
            if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".m4a"}
        ]
        if not audio_files:
            return jsonify({'error': 'No audio file found'}), 404

        latest_file = max(audio_files, key=lambda path: path.stat().st_mtime)
        try:
            duration = get_audio_duration(str(latest_file))
        except Exception:
            duration = None

        return jsonify({
            'id': latest_file.name,
            'path': str(latest_file.absolute()),
            'duration': duration,
        })
    except Exception as e:
        logger.error(f"Error finding latest audio file: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to find latest audio file'}), 500


@app.route('/api/v1/generate-script', methods=['POST'])
def generate_script_endpoint():
    """
    Generate script and subtitles using Gemini AI.

    Request:
    {
        "script": "Your script text here",
        "output_name": "optional_output_name"
    }
    """
    try:
        data = request.json or {}
        topic = data.get('script') or data.get('topic')
        language = data.get('language', 'english')
        video_profile = normalize_video_profile(data.get('video_profile'))

        # Generate script along with description and hashtags
        result = generate_script(
            topic_override=topic,
            language=language,
            video_profile=video_profile,
        )

        return jsonify({
            'status': 'success',
            'message': 'Script generated successfully',
            'output': {
                'title': result.get('title', ''),
                'script': result.get('script', ''),
                'description': result.get('description', ''),
                'hashtags': result.get('hashtags', ''),
                'resolved_topic': result.get('resolved_topic', ''),
                'quality_warnings': result.get('quality_warnings', []),
                'video_profile': video_profile,
            }
        })
    except Exception as e:
        logger.error(f"Error in generate_script_endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'output': None
        }), 500


@app.route('/api/v1/generate-slideshow', methods=['POST'])
def generate_slideshow():
    """
    Generate a slideshow based on the provided audio ID.
    
    Request JSON:
    {
        "audio_id": "audio_file_id"
    }
    
    Returns:
        JSON: { "slideshow_video_id": str, "message": str }
    """
    try:
        data = request.get_json()
        if not data or 'audio_id' not in data:
            return jsonify({'error': 'Missing audio_id parameter'}), 400
            
        audio_id = data['audio_id']
        video_profile = normalize_video_profile(data.get('video_profile'))
        profile_config = get_video_profile_config(video_profile)
        audio_path = UPLOADS_DIR / audio_id
        
        if not audio_path.exists():
            return jsonify({'error': 'Audio file not found'}), 404
            
        # Generate a unique ID for the slideshow
        slideshow_id = f"{uuid.uuid4()}.mp4"
        output_path = UPLOADS_DIR / slideshow_id
        
        # Get the latest subtitle file (you might want to pass this as a parameter instead)
        ass_file = get_latest_ass_file()
        if not ass_file:
            return jsonify({'error': 'No subtitle file found. Please generate subtitles first.'}), 400
            
        # Get audio duration for the slideshow
        duration = get_audio_duration(str(audio_path))
        
        # Generate image slides JSON
        image_slides_json = UPLOADS_DIR / f"{uuid.uuid4()}.image_slides.json"
        image_dir = UPLOADS_DIR / "images"
        image_dir.mkdir(exist_ok=True)
        
        # Generate the slideshow
        from app.utils.slides.image_slides import generate_image_slides
        generate_image_slides(
            ass_path=str(ass_file),
            out_path=str(image_slides_json),
            total_duration=duration,
            video_profile=video_profile,
        )
        
        # Generate the video from the slides
        from app.utils.slides.generate_slideshow import main as generate_slideshow_video
        generate_slideshow_video(
            json_path=str(image_slides_json),
            image_dir=str(image_dir),
            output_path=str(output_path),
            total_duration=duration,
            resolution=profile_config['video_resolution'],
        )
        
        return jsonify({
            'slideshow_video_id': slideshow_id,
            'message': 'Slideshow generated successfully',
            'video_profile': video_profile,
        })
        
    except Exception as e:
        logger.error(f"Error generating slideshow: {str(e)}")
        return jsonify({
            'error': f'Failed to generate slideshow: {str(e)}'
        }), 500


@app.route('/api/v1/upload-audio', methods=['POST'])
def upload_audio():
    """
    Upload audio file for processing.

    Returns:
        JSON: {id: str, duration: float, message: str}
    """
    try:
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file provided'
            }), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({
                'error': 'No selected file'
            }), 400

        # Create uploads directory if it doesn't exist
        create_uploads_dir()

        # Save the file
        file_path = save_audio_file(file, file.filename)
        original_path = file_path
        
        try:
            # Convert to WAV if not already
            wav_path = file_path.rsplit('.', 1)[0] + '.wav'
            if file_path.lower().endswith('.mp3'):
                convert_to_wav(file_path, wav_path)
                file_path = wav_path
            
            # Get duration
            duration = get_audio_duration(file_path)
            
            # If we converted to WAV, remove the original file
            if original_path != file_path and os.path.exists(original_path):
                os.remove(original_path)

            return jsonify({
                'id': os.path.basename(file_path),
                'duration': duration,
                'message': 'Audio uploaded successfully'
            })
            
        except Exception as e:
            # Clean up if there was an error during processing
            if os.path.exists(file_path):
                os.remove(file_path)
            if 'original_path' in locals() and os.path.exists(original_path):
                os.remove(original_path)
            raise

    except ValueError as e:
        return jsonify({
            'error': str(e)
        }), 400

    except Exception as e:
        logger.error(f"Error uploading audio: {str(e)}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@app.route('/api/v1/upload-manga-pdf', methods=['POST'])
def upload_manga_pdf():
    """
    Upload and process an English manga PDF for the PDF-to-video workflow.
    """
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400

        file = request.files['pdf']
        if not file.filename:
            return jsonify({'error': 'No selected file'}), 400
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_size > MAX_PDF_SIZE:
            return jsonify({'error': f'PDF file too large. Maximum size: {MAX_PDF_SIZE/1024/1024:.1f}MB'}), 400

        result = process_manga_pdf(file)
        return jsonify({
            'status': 'success',
            'message': 'Manga PDF processed successfully',
            'output': result
        })
    except Exception as e:
        logger.error(f"Error uploading manga PDF: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/generate-script-from-pdf', methods=['POST'])
def generate_script_from_pdf_endpoint():
    """
    Generate an English script grounded in a processed manga PDF.
    """
    try:
        data = request.json or {}
        pdf_id = data.get('pdf_id')
        topic = data.get('topic') or data.get('angle')
        video_profile = normalize_video_profile(data.get('video_profile'))
        if not pdf_id:
            return jsonify({'error': 'pdf_id is required'}), 400

        try:
            manifest = load_manga_pdf_manifest(pdf_id)
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404

        try:
            result = _generate_pdf_script_from_manifest(
                manifest,
                topic,
                video_profile=video_profile,
            )
        except ValueError as e:
            context = _build_pdf_script_context(manifest)
            return jsonify({
                'error': str(e),
                'details': {
                    'chapter_number': context['chapter_number'],
                    'text_quality': context['text_quality'],
                    'warnings': context['warnings'],
                }
            }), 400

        update_manga_session(pdf_id, "script", "completed", {
            "title": result.get('title', ''),
            "script": result.get('script', ''),
            "description": result.get('description', ''),
            "hashtags": result.get('hashtags', ''),
            "resolved_topic": result.get('resolved_topic', ''),
            "quality_warnings": result.get('quality_warnings', []),
            "chapter_number": result.get('chapter_number'),
            "text_quality": result.get('text_quality'),
            "context_sources": result.get('context_sources', []),
            "video_profile": video_profile,
        })
        return jsonify({
            'status': 'success',
            'message': 'Script generated from manga PDF successfully',
            'output': {
                'title': result.get('title', ''),
                'script': result.get('script', ''),
                'description': result.get('description', ''),
                'hashtags': result.get('hashtags', ''),
                'resolved_topic': result.get('resolved_topic', ''),
                'quality_warnings': result.get('quality_warnings', []),
                'pdf_id': pdf_id,
                'chapter_number': result.get('chapter_number'),
                'text_quality': result.get('text_quality'),
                'context_sources': result.get('context_sources', []),
                'warnings': result.get('warnings', []),
                'video_profile': video_profile,
            }
        })
    except Exception as e:
        logger.error(f"Error generating PDF script: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/generate-pdf-slides', methods=['POST'])
def generate_pdf_slides_endpoint():
    """
    Generate slide JSON from local manga PDF panel crops.
    """
    try:
        data = request.json or {}
        pdf_id = data.get('pdf_id')
        audio_id = data.get('audio_id')
        video_profile = normalize_video_profile(data.get('video_profile'))
        if not pdf_id:
            return jsonify({'error': 'pdf_id is required'}), 400
        if not audio_id:
            return jsonify({'error': 'audio_id is required'}), 400

        audio_path = UPLOADS_DIR / audio_id
        if not audio_path.exists():
            return jsonify({'error': f'Audio file not found: {audio_id}'}), 404

        try:
            load_manga_pdf_manifest(pdf_id)
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404

        duration = get_audio_duration(str(audio_path))
        result = create_pdf_slides(pdf_id, duration)
        update_manga_session(pdf_id, "slides", "completed", {
            "audio_id": audio_id,
            "slides_json": result['slides_json'],
            "slide_count": result['slide_count'],
            "video_profile": video_profile,
        })
        return jsonify({
            'status': 'success',
            'message': 'PDF slides generated successfully',
            'output_path': result['slides_json'],
            'slides_json': result['slides_json'],
            'slides': result['slides'],
            'slide_count': result['slide_count'],
            'video_profile': video_profile,
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating PDF slides: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/generate-manga-video', methods=['POST'])
def generate_manga_video_endpoint():
    """
    One-click Manga PDF pipeline:
    PDF context script -> Qwen voiceover -> subtitles/expressions -> PDF panel slides -> final video.
    """
    pdf_id = None
    try:
        data = request.json or {}
        pdf_id = data.get('pdf_id')
        topic = data.get('topic') or data.get('angle')
        language = data.get('language', 'English')
        subtitle_style = data.get('subtitle_style', 'pro')
        visual_style = normalize_visual_style(data.get('visual_style'))
        quality_mode = data.get('quality_mode', 'pro')
        background_music_path = data.get('background_music_path')
        video_profile = normalize_video_profile(data.get('video_profile'))

        if not pdf_id:
            return jsonify({'error': 'pdf_id is required'}), 400

        try:
            manifest = load_manga_pdf_manifest(pdf_id)
        except FileNotFoundError as e:
            return jsonify({'error': str(e)}), 404

        update_manga_session(pdf_id, "pipeline", "running", {
            "quality_mode": quality_mode,
            "subtitle_style": subtitle_style,
            "language": language,
            "video_profile": video_profile,
        })

        update_manga_session(pdf_id, "script", "running")
        script_result = _generate_pdf_script_from_manifest(
            manifest,
            topic,
            video_profile=video_profile,
        )
        update_manga_session(pdf_id, "script", "completed", {
            "title": script_result.get('title', ''),
            "script": script_result.get('script', ''),
            "description": script_result.get('description', ''),
            "hashtags": script_result.get('hashtags', ''),
            "resolved_topic": script_result.get('resolved_topic', ''),
            "quality_warnings": script_result.get('quality_warnings', []),
            "chapter_number": script_result.get('chapter_number'),
            "text_quality": script_result.get('text_quality'),
            "context_sources": script_result.get('context_sources', []),
            "video_profile": video_profile,
        })

        update_manga_session(pdf_id, "voiceover", "running")
        voiceover_result = generate_voiceover(
            text=script_result['script'],
            language=language,
            video_profile=video_profile,
        )
        audio_id = voiceover_result['id']
        update_manga_session(pdf_id, "voiceover", "completed", {
            "audio_id": audio_id,
            "duration": voiceover_result.get('duration'),
            "path": voiceover_result.get('path'),
            "video_profile": video_profile,
        })

        update_manga_session(pdf_id, "subtitles", "running")
        subtitle_assets = _generate_subtitle_assets(
            audio_id,
            script_result['script'],
            subtitle_style,
            video_profile=video_profile,
            visual_style=visual_style,
        )
        update_manga_session(pdf_id, "subtitles", "completed", {
            "ass_file": subtitle_assets['ass_file'],
            "ass_path": subtitle_assets['ass_path'],
            "expressions_file": subtitle_assets['expressions_file'],
            "expressions_path": subtitle_assets['expressions_path'],
            "phrase_count": len(subtitle_assets['phrases']),
            "video_profile": video_profile,
        })

        update_manga_session(pdf_id, "slides", "running")
        duration = get_audio_duration(str(UPLOADS_DIR / audio_id))
        slides_result = create_pdf_slides(pdf_id, duration)
        update_manga_session(pdf_id, "slides", "completed", {
            "slides_json": slides_result['slides_json'],
            "slide_count": slides_result['slide_count'],
            "video_profile": video_profile,
        })

        update_manga_session(pdf_id, "final_video", "running")
        final_video_path = generate_final_video(
            audio_id=audio_id,
            slides_json_path=slides_result['slides_json'],
            subtitle_path=subtitle_assets['ass_path'],
            expressions_path=subtitle_assets['expressions_path'],
            generate_slides=False,
            quality_mode=quality_mode,
            background_music_path=background_music_path,
            video_profile=video_profile,
            visual_style=visual_style,
        )
        video_filename = os.path.basename(final_video_path)
        video_url = url_for('download_file', filename=video_filename, _external=True)
        update_manga_session(pdf_id, "final_video", "completed", {
            "video_file": video_filename,
            "video_path": final_video_path,
            "download_url": video_url,
            "video_profile": video_profile,
        })
        session = update_manga_session(pdf_id, "completed", "completed", {
            "video_file": video_filename,
            "audio_id": audio_id,
            "slides_json": slides_result['slides_json'],
            "video_profile": video_profile,
        })

        return jsonify({
            'status': 'success',
            'message': 'Manga video generated successfully',
            'output': {
                'pdf_id': pdf_id,
                'script': script_result,
                'audio': {
                    'id': audio_id,
                    'duration': voiceover_result.get('duration'),
                },
                'subtitles': {
                    'ass_file': subtitle_assets['ass_file'],
                    'ass_path': subtitle_assets['ass_path'],
                    'expressions_file': subtitle_assets['expressions_file'],
                    'expressions_path': subtitle_assets['expressions_path'],
                },
                'slides': {
                    'slides_json': slides_result['slides_json'],
                    'slide_count': slides_result['slide_count'],
                },
                'video_file': video_filename,
                'download_url': video_url,
                'video_profile': video_profile,
                'session': session,
            }
        })

    except ValueError as e:
        if pdf_id:
            update_manga_session(pdf_id, "failed", "error", error=str(e))
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error generating one-click manga video: {str(e)}", exc_info=True)
        if pdf_id:
            update_manga_session(pdf_id, "failed", "error", error=str(e))
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/manga-session/<pdf_id>', methods=['GET'])
def get_manga_session_endpoint(pdf_id):
    try:
        load_manga_pdf_manifest(pdf_id)
        return jsonify({
            'status': 'success',
            'session': load_manga_session(pdf_id)
        })
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/v1/generate-voiceover', methods=['POST'])
def generate_voiceover_endpoint():
    """
    Generate a voiceover audio file from script text using Qwen3-TTS voice cloning.

    Request:
    {
        "script": "Text to speak",
        "language": "English"
    }

    Returns:
        JSON: {id: str, duration: float, message: str}
    """
    try:
        data = request.json or {}
        script = data.get('script') or data.get('text')
        language = data.get('language', 'English')
        video_profile = normalize_video_profile(data.get('video_profile'))

        if not script or not script.strip():
            return jsonify({'error': 'script is required'}), 400

        result = generate_voiceover(
            text=script,
            language=language,
            video_profile=video_profile,
        )

        return jsonify({
            'id': result['id'],
            'duration': result['duration'],
            'video_profile': video_profile,
            'message': 'Voiceover generated successfully'
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        logger.error(f"Voiceover generation runtime error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Error generating voiceover: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to generate voiceover'}), 500


@app.route('/api/v1/generate-subtitles', methods=['POST'])
def generate_subtitles():
    """
    Generate subtitles for uploaded audio, with optional script.
    If script is not provided, it will be automatically transcribed from the audio.

    Request:
    {
        "audio_id": "file_id",
        "script": "Optional script text. If not provided, will be transcribed from audio.",
        "subtitle_style": "Optional style (default: 'epic')"
    }
    """
    try:
        data = request.json
        audio_id = data.get('audio_id')
        script = data.get('script')  # This is now optional
        subtitle_style = data.get('subtitle_style', 'pro')
        visual_style = normalize_visual_style(data.get('visual_style'))
        video_profile = normalize_video_profile(data.get('video_profile'))

        if not audio_id:
            return jsonify({
                'error': 'audio_id is required'
            }), 400

        # Get audio file path
        audio_path = os.path.join(UPLOADS_DIR, audio_id)
        if not os.path.exists(audio_path):
            return jsonify({
                'error': f'Audio file not found: {audio_id}'
            }), 404

        # Log whether we're using provided script or transcribing
        if script and script.strip():
            logger.info(f"Generating subtitles with provided script (length: {len(script)} chars)")
        else:
            logger.info("No script provided, will transcribe from audio")

        try:
            subtitle_assets = _generate_subtitle_assets(
                audio_id,
                script,
                subtitle_style,
                video_profile=video_profile,
                visual_style=visual_style,
            )
        except Exception as e:
            logger.error(f"Error generating subtitle assets: {str(e)}")
            return jsonify({
                'error': f'Failed to process audio: {str(e)}'
            }), 500

        return jsonify({
            'status': 'success',
            'message': 'Subtitles generated successfully',
            'output': {
                'ass_file': subtitle_assets['ass_file'],
                'phrases': subtitle_assets['phrases'],
                'expressions_file': subtitle_assets['expressions_file'],
                'expressions': subtitle_assets['expressions'],
                'video_profile': video_profile,
            }
        })

    except Exception as e:
        logger.error(f"Error generating subtitles: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/v1/generate-image-slides', methods=['POST'])
def generate_image_slides_endpoint():
    """
    Generate image slide prompts from an ASS subtitle file.
    Request JSON:
    {
        "ass_path": "relative/or/absolute/path/to/file.ass",
        "audio_id": "file_id", # Added audio_id
        "out_path": "relative/or/absolute/path/to/output.json" (optional)
    }
    """
    try:
        data = request.json
        logger.info(f"Received request data: {data}")
        
        ass_path = data.get('ass_path')
        audio_id = data.get('audio_id')
        out_path = data.get('out_path')
        video_profile = normalize_video_profile(data.get('video_profile'))
        
        if not audio_id:
            return jsonify({'error': 'audio_id is required'}), 400
            
        # Handle automatic ASS file detection
        if ass_path == 'auto' or not ass_path:
            logger.info("Looking for latest ASS file...")
            ass_path = _find_latest_ass_file()
            if not ass_path:
                return jsonify({'error': 'No ASS file found. Please generate subtitles first.'}), 400
                
            logger.info(f"Found ASS file: {ass_path}")
        
        logger.info(f"Using ASS file: {ass_path}")
        
        # Ensure ass_path is a string and create Path object
        ass_path = str(ass_path)  # Convert to string first in case it's a Path object
        ass_path_obj = Path(ass_path)
        
        logger.info(f"Checking if ASS file exists at: {ass_path_obj.absolute()}")
        if not ass_path_obj.exists():
            return jsonify({'error': f'ASS file not found: {ass_path_obj.absolute()}'}), 404
        
        # Get audio duration
        audio_path = UPLOADS_DIR / audio_id
        logger.info(f"Looking for audio file at: {audio_path.absolute()}")
        if not audio_path.exists():
            return jsonify({'error': f'Audio file not found: {audio_id}'}), 404
            
        total_audio_duration = get_audio_duration(str(audio_path))
        logger.info(f"Total audio duration for image slides: {total_audio_duration:.2f}s")

        # Set default output path next to audio (upload images per slide in the UI)
        if not out_path:
            out_path = str(UPLOADS_DIR / f"{audio_stem(audio_id)}.image_slides.json")
            logger.info(f"Using default output path: {out_path}")

        slides_images_dir(audio_id)

        logger.info(
            "Generating image slides with params: ass_path=%s, out_path=%s, duration=%.2f",
            ass_path,
            out_path,
            total_audio_duration,
        )

        result_path = generate_image_slides(
            ass_path,
            out_path,
            total_audio_duration,
            video_profile=video_profile,
        )

        logger.info(f"Image slides generated successfully at: {result_path}")

        slides = load_slides(result_path)
        payload = build_slides_response(audio_id, result_path, slides)

        return jsonify({
            'status': 'success',
            'slides': slides,
            'output_path': result_path,
            'video_profile': video_profile,
            'upload_status': {
                'total': payload['total'],
                'uploaded': payload['uploaded'],
                'complete': payload['complete'],
            },
            'slides_detail': payload['slides'],
        })
        
    except TimeoutError as e:
        logger.error(f"LLM timeout in generate_image_slides_endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_type': 'llm_timeout',
        }), 504
    except ConnectionError as e:
        logger.error(f"LLM connectivity error in generate_image_slides_endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_type': 'llm_unreachable',
        }), 503
    except Exception as e:
        logger.error(f"Error in generate_image_slides_endpoint: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/image-slides', methods=['GET'])
def get_image_slides():
    """Return slide definitions and upload status for an audio session."""
    audio_id = request.args.get('audio_id')
    if not audio_id:
        return jsonify({'status': 'error', 'message': 'audio_id is required'}), 400

    slides_path = request.args.get('slides_json')
    if slides_path:
        slides_path = str(Path(slides_path))
        if not os.path.exists(slides_path):
            return jsonify({'status': 'error', 'message': f'Slides file not found: {slides_path}'}), 404
    else:
        resolved = slides_json_path_for_audio(audio_id, UPLOADS_DIR)
        if not resolved:
            return jsonify({'status': 'error', 'message': 'No image slides JSON found for this audio'}), 404
        slides_path = str(resolved)

    slides = load_slides(slides_path)
    payload = build_slides_response(audio_id, slides_path, slides)
    return jsonify({'status': 'success', **payload})


@app.route('/api/v1/image-slides/upload', methods=['POST'])
def upload_image_slide():
    """
    Upload an image for one slide (multipart form).

    Form fields: audio_id, slide_index, slides_json (optional), file (image)
    """
    audio_id = request.form.get('audio_id')
    slide_index_raw = request.form.get('slide_index')
    slides_json = request.form.get('slides_json')

    if not audio_id:
        return jsonify({'status': 'error', 'message': 'audio_id is required'}), 400
    if slide_index_raw is None:
        return jsonify({'status': 'error', 'message': 'slide_index is required'}), 400

    try:
        slide_index = int(slide_index_raw)
    except ValueError:
        return jsonify({'status': 'error', 'message': 'slide_index must be an integer'}), 400

    upload_file = request.files.get('file') or request.files.get('image')
    if not upload_file:
        return jsonify({'status': 'error', 'message': 'file is required'}), 400

    if not slides_json:
        resolved = slides_json_path_for_audio(audio_id, UPLOADS_DIR)
        if not resolved:
            return jsonify({'status': 'error', 'message': 'No image slides JSON found. Generate slides first.'}), 404
        slides_json = str(resolved)

    try:
        updated_slide = save_slide_upload(audio_id, slide_index, upload_file, slides_json)
        slides = load_slides(slides_json)
        payload = build_slides_response(audio_id, slides_json, slides)
        return jsonify({
            'status': 'success',
            'message': f'Uploaded image for slide {slide_index + 1}',
            'slide': updated_slide,
            'upload_status': {
                'total': payload['total'],
                'uploaded': payload['uploaded'],
                'complete': payload['complete'],
            },
            'slides_detail': payload['slides'],
        })
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Error uploading slide image: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/image-slides/preview', methods=['GET'])
def preview_image_slide():
    """Serve an uploaded slide image for inline preview in the UI."""
    audio_id = request.args.get('audio_id')
    slide_index_raw = request.args.get('slide_index')

    if not audio_id or slide_index_raw is None:
        return jsonify({'status': 'error', 'message': 'audio_id and slide_index are required'}), 400

    try:
        # Tolerate malformed cache-bust URLs (slide_index=0?t=123)
        slide_index = int(str(slide_index_raw).split("?")[0].split("&")[0])
    except ValueError:
        return jsonify({'status': 'error', 'message': 'slide_index must be an integer'}), 400

    image_path = slides_images_dir(audio_id) / f"slide_{slide_index + 1:03d}.jpg"
    if not image_path.exists():
        return jsonify({'status': 'error', 'message': 'Image not found for this slide'}), 404

    return send_file(image_path, mimetype='image/jpeg', max_age=0)


@app.route("/api/v1/health")
def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "vivre_cards": vivre_card_status(),
    }


@app.route("/api/v1/vivre-cards/status", methods=["GET"])
def vivre_cards_status_endpoint():
    """Vivre Card asset pack scan status (app/data/vivre-card by default)."""
    status = vivre_card_status()
    return jsonify({"status": "success", **status})


@app.route('/api/v1/vivre-cards/preview', methods=['GET'])
def vivre_cards_preview():
    """Inline preview for a Vivre PNG (relative path under the pack)."""
    relative = request.args.get('relative', '').strip()
    asset_path = vivre_asset_path_from_relative(relative)
    if not asset_path:
        return jsonify({'status': 'error', 'message': 'Asset not found'}), 404
    return send_file(asset_path, mimetype='image/png', max_age=3600)


@app.route('/api/v1/vivre-cards/suggest', methods=['GET'])
def vivre_cards_suggest():
    """Suggest symbol/location/character PNGs for a search query."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'status': 'error', 'message': 'q is required'}), 400
    limit = min(int(request.args.get('limit', 5)), 12)
    suggestions = suggest_vivre_assets(query, limit=limit)
    return jsonify({'status': 'success', 'query': query, 'suggestions': suggestions})


@app.route('/api/v1/image-slides/use-vivre', methods=['POST'])
def use_vivre_for_slide():
    """Apply a Vivre Card PNG to a slide slot (from suggestions)."""
    data = request.get_json() or {}
    audio_id = data.get('audio_id') or request.form.get('audio_id')
    slide_index = data.get('slide_index')
    vivre_relative = data.get('vivre_relative') or data.get('relative')
    slides_json = data.get('slides_json')

    if not audio_id or slide_index is None or not vivre_relative:
        return jsonify({
            'status': 'error',
            'message': 'audio_id, slide_index, and vivre_relative are required',
        }), 400

    try:
        slide_index = int(slide_index)
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'message': 'slide_index must be an integer'}), 400

    if not slides_json:
        resolved = slides_json_path_for_audio(audio_id, UPLOADS_DIR)
        if not resolved:
            return jsonify({'status': 'error', 'message': 'No image slides JSON found'}), 404
        slides_json = str(resolved)

    try:
        apply_vivre_asset_to_slide(audio_id, slide_index, vivre_relative, slides_json)
        slides = load_slides(slides_json)
        payload = build_slides_response(audio_id, slides_json, slides)
        return jsonify({
            'status': 'success',
            'message': f'Applied Vivre asset to slide {slide_index + 1}',
            'upload_status': {
                'total': payload['total'],
                'uploaded': payload['uploaded'],
                'complete': payload['complete'],
            },
            'slides_detail': payload['slides'],
        })
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"use_vivre_for_slide failed: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/output-usage', methods=['GET'])
def output_usage_endpoint():
    """
    Return generated output folder usage.
    """
    try:
        return jsonify({
            'status': 'success',
            'usage': get_output_usage()
        })
    except Exception as e:
        logger.error(f"Error getting output usage: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/cleanup-output', methods=['POST'])
def cleanup_output_endpoint():
    """
    Clean generated output files.

    Request:
    {
        "max_age_hours": 24,
        "force": false
    }
    """
    try:
        data = request.json or {}
        max_age_hours = data.get('max_age_hours', 24)
        force = bool(data.get('force', False))

        result = cleanup_output(max_age_hours=max_age_hours, force=force)
        return jsonify({
            'status': 'success',
            'message': 'Output cleanup completed',
            'result': result
        })
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Error cleaning output folder: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/download/<filename>')
def download_file(filename):
    """
    Download a file from either the uploads or compiled_videos directory.
    Supports range requests for video files.
    """
    try:
        # Security check to prevent directory traversal
        if '..' in filename or filename.startswith('/'):
            logger.warning(f"Invalid file path attempted: {filename}")
            return jsonify({'error': 'Invalid file path'}), 400
            
        # Check in UPLOADS_DIR first
        file_path = os.path.join(UPLOADS_DIR, filename)
        
        # If not found in UPLOADS_DIR, check COMPILED_VIDEO_DIR
        if not os.path.exists(file_path):
            from app.config import COMPILED_VIDEO_DIR
            file_path = os.path.join(COMPILED_VIDEO_DIR, filename)
            
        logger.info(f"Download requested for file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found in any directory: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        # Get file size for range requests
        size = os.path.getsize(file_path)
        
        # For video files, handle range requests
        if filename.lower().endswith(('.mp4', '.mkv')):
            range_header = request.headers.get('Range', None)
            if range_header:
                # Parse the range header
                try:
                    range_ = range_header.replace('bytes=', '').split('-')
                    byte1 = int(range_[0]) if range_[0] else 0
                    byte2 = int(range_[1]) if len(range_) > 1 and range_[1] else None
                    
                    # Calculate length based on range
                    if byte2 is not None:
                        length = byte2 - byte1 + 1
                    else:
                        length = size - byte1
                    
                    # Ensure length is not negative and doesn't exceed file size
                    if length <= 0 or byte1 >= size:
                        return jsonify({'error': 'Requested range not satisfiable'}), 416
                        
                    # Ensure we don't read past the end of the file
                    if byte1 + length > size:
                        length = size - byte1
                    
                    # Read the file chunk
                    data = None
                    with open(file_path, 'rb') as f:
                        f.seek(byte1)
                        data = f.read(length)
                        
                except (ValueError, IndexError) as e:
                    logger.error(f"Invalid range header {range_header}: {str(e)}")
                    return jsonify({'error': 'Invalid range header'}), 400
                
                rv = Response(
                    data,
                    206,  # Partial Content
                    mimetype='video/mp4',
                    direct_passthrough=True
                )
                rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
                rv.headers.add('Accept-Ranges', 'bytes')
                rv.headers.add('Content-Length', str(length))
                rv.headers.add('Content-Disposition', 'attachment', filename=filename)
                return rv
        
        # For non-video files or non-range requests, use send_file
        mimetype = 'video/mp4' if filename.lower().endswith('.mp4') else \
                 ('video/x-matroska' if filename.lower().endswith('.mkv') else None)
                
        logger.info(f"Serving file {file_path} with mimetype {mimetype}")
        return send_file(
            file_path,
            as_attachment=True,
            mimetype=mimetype,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {str(e)}")
        return jsonify({'error': 'Failed to download file'}), 500


@app.route('/api/v1/preview/<filename>')
def preview_file(filename):
    """
    Stream a file from the uploads directory for inline preview (with HTTP range support).
    """
    try:
        if '..' in filename or filename.startswith('/'):
            logger.warning(f"Invalid file path attempted: {filename}")
            return jsonify({'error': 'Invalid file path'}), 400
        file_path = os.path.join(UPLOADS_DIR, filename)
        logger.info(f"Preview requested for file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found for preview: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        mimetype = 'video/mp4' if filename.lower().endswith('.mp4') else (
            'video/x-matroska' if filename.lower().endswith('.mkv') else None)
        range_header = request.headers.get('Range', None)
        if not range_header:
            return send_from_directory(
                UPLOADS_DIR,
                filename,
                as_attachment=False,
                mimetype=mimetype,
                download_name=filename
            )
        # Handle HTTP Range requests
        size = os.path.getsize(file_path)
        byte1, byte2 = 0, None
        m = re.search(r'bytes=(\d+)-(\d*)', range_header)
        if m:
            g = m.groups()
            byte1 = int(g[0])
            if g[1]:
                byte2 = int(g[1])
        length = size - byte1 if byte2 is None else byte2 - byte1 + 1
        with open(file_path, 'rb') as f:
            f.seek(byte1)
            data = f.read(length)
        rv = Response(data, 206, mimetype=mimetype, direct_passthrough=True)
        rv.headers.add('Content-Range',
                       f'bytes {byte1}-{byte1 + length - 1}/{size}')
        rv.headers.add('Accept-Ranges', 'bytes')
        rv.headers.add('Content-Length', str(length))
        return rv
    except Exception as e:
        logger.error(f"Error previewing file {filename}: {str(e)}")
        return jsonify({'error': 'Failed to preview file'}), 500


@app.route('/api/v1/generate-final-video', methods=['POST'])
def generate_final_video_endpoint():
    """
    Generate the final video with slideshow, audio, subtitles, and expressions.
    This endpoint will handle both slides generation and final video creation.
    
    Request JSON:
    {
        "audio_id": "file_id",
        "subtitle_file": "path/to/subtitle.ass",
        "expressions_file": "path/to/expressions.json" (optional),
        "force_regenerate_slides": false (optional, set to true to force regenerate slides)
    }
    """
    logger.info("=== Starting generate_final_video_endpoint ===")
    logger.info(f"Request data: {json.dumps(request.json, indent=2) if request.is_json else 'No JSON data'}")
    
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in request")
            return jsonify({'status': 'error', 'message': 'No JSON data received'}), 400
            
        logger.info(f"Request data: {json.dumps(data, indent=2)}")
        
        audio_id = data.get('audio_id')
        subtitle_file = data.get('subtitle_file')
        expressions_file = data.get('expressions_file')
        force_regenerate = data.get('force_regenerate_slides', False)
        quality_mode = data.get('quality_mode', 'pro')
        background_music_path = data.get('background_music_path')
        video_profile = normalize_video_profile(data.get('video_profile'))
        visual_style = normalize_visual_style(data.get('visual_style'))
        
        logger.info(f"Processing request with audio_id={audio_id}, subtitle_file={subtitle_file}")

        if not all([audio_id, subtitle_file]):
            error_msg = 'audio_id and subtitle_file are required'
            logger.error(error_msg)
            return jsonify({'status': 'error', 'message': error_msg}), 400
            
        # Verify the subtitle file exists
        if not os.path.exists(subtitle_file):
            error_msg = f'Subtitle file not found: {subtitle_file}'
            logger.error(error_msg)
            return jsonify({'status': 'error', 'message': error_msg}), 400

        # Handle slides generation based on parameters
        use_existing_slides = data.get('use_existing_slides', True)
        slides_json = data.get('slides_json')
        
        logger.info(f"Slides generation parameters - use_existing_slides: {use_existing_slides}, force_regenerate: {force_regenerate}, provided_slides: {slides_json is not None}")
        
        # If we're forcing regeneration or don't have existing slides to use
        if force_regenerate or not use_existing_slides or not slides_json or not os.path.exists(slides_json):
            if force_regenerate:
                logger.info("Forcing regeneration of slides as requested...")
            elif not use_existing_slides:
                logger.info("Generating new slides as use_existing_slides is False...")
            else:
                logger.info("Existing slides not found or invalid, generating new slides...")
                
            try:
                # Generate new slides with the current audio_id
                from flask import jsonify as flask_jsonify
                
                # Create a new request context to call the endpoint
                with app.test_request_context(
                    '/api/v1/generate-image-slides',
                    method='POST',
                    json={
                        'audio_id': audio_id,
                        'ass_path': 'auto',
                        'video_profile': video_profile,
                    }
                ):
                    # Call the endpoint function directly with the test context
                    slides_result = generate_image_slides_endpoint()
                    
                    if isinstance(slides_result, tuple):  # If there was an error
                        logger.error(f"Error generating slides: {slides_result[0].get_json()}")
                        return slides_result
                    
                    # Get the JSON response
                    if hasattr(slides_result, 'get_json'):
                        slides_data = slides_result.get_json()
                        slides_json = slides_data.get('output_path')
                        logger.info(f"Generated new slides at: {slides_json}")
                        
                        if not slides_json or not os.path.exists(slides_json):
                            error_msg = f'Failed to generate slides. Output path not found: {slides_json}'
                            logger.error(error_msg)
                            return jsonify({'status': 'error', 'message': error_msg}), 500
                    else:
                        error_msg = 'Invalid response format from generate_image_slides_endpoint'
                        logger.error(error_msg)
                        return jsonify({'status': 'error', 'message': error_msg}), 500
                    
            except Exception as e:
                error_msg = f'Error generating slides: {str(e)}'
                logger.error(error_msg, exc_info=True)
                return jsonify({'status': 'error', 'message': error_msg}), 500
        else:
            logger.info(f"Using existing slides from: {slides_json}")
            # Verify the slides file exists
            if not os.path.exists(slides_json):
                error_msg = f'Specified slides file not found: {slides_json}'
                logger.error(error_msg)
                return jsonify({'status': 'error', 'message': error_msg}), 400

        auto_resolve_status = auto_resolve_slide_assets(audio_id, slides_json)
        slides = auto_resolve_status["slides"]
        upload_status = slides_upload_status(slides)
        if not upload_status['complete']:
            return jsonify({
                'status': 'error',
                'message': (
                    f"Upload or resolve an image for each slide before generating video "
                    f"({upload_status['uploaded']}/{upload_status['total']} ready, "
                    f"{auto_resolve_status.get('resolved', 0)} auto-resolved)."
                ),
                'upload_status': upload_status,
                'auto_resolve_status': {
                    'resolved': auto_resolve_status.get('resolved', 0),
                    'unresolved': auto_resolve_status.get('unresolved', 0),
                },
            }), 400

        images_dir = str(slides_images_dir(audio_id))

        logger.info("Starting final video generation...")
        try:
            final_video_path = generate_final_video(
                audio_id=audio_id,
                slides_json_path=slides_json,
                subtitle_path=subtitle_file,
                expressions_path=expressions_file,
                generate_slides=force_regenerate,
                quality_mode=quality_mode,
                background_music_path=background_music_path,
                video_profile=video_profile,
                visual_style=visual_style,
                images_dir=images_dir,
            )
            logger.info(f"Final video generated at: {final_video_path}")
            
            if not os.path.exists(final_video_path):
                error_msg = f'Final video file not found at: {final_video_path}'
                logger.error(error_msg)
                return jsonify({'status': 'error', 'message': error_msg}), 500
                
            video_filename = os.path.basename(final_video_path)
            video_url = url_for('download_file', filename=video_filename, _external=True)
            
            logger.info("Video generation completed successfully")
            
            return jsonify({
                'status': 'success',
                'message': 'Final video generated successfully',
                'video_file': video_filename,
                'download_url': video_url,
                'slides_json': slides_json,
                'video_profile': video_profile,
            })
            
        except Exception as e:
            error_msg = f'Error in generate_final_video: {str(e)}'
            logger.error(error_msg, exc_info=True)
            return jsonify({'status': 'error', 'message': error_msg}), 500

    except Exception as e:
        error_msg = f'Unexpected error in generate_final_video_endpoint: {str(e)}'
        logger.error(error_msg, exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred while generating the video',
            'error': str(e)
        }), 500
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _find_latest_ass_file():
    """Helper function to find the latest .ass file in the uploads directory.
    
    Returns:
        str: Path to the latest .ass file, or None if not found
    """
    try:
        # Ensure UPLOADS_DIR is a Path object
        uploads_dir = Path(UPLOADS_DIR)
        if not uploads_dir.exists():
            logger.error(f"Uploads directory not found: {uploads_dir}")
            return None
            
        # Find all .ass files in the uploads directory
        ass_files = list(uploads_dir.glob('*.ass'))
        
        if not ass_files:
            logger.info("No .ass files found in uploads directory")
            return None
            
        # Get the most recently modified file
        latest_file = max(ass_files, key=os.path.getmtime)
        logger.info(f"Found latest ASS file: {latest_file}")
        
        # Return the absolute path as a string
        return str(latest_file.absolute())
        
    except Exception as e:
        logger.error(f"Error finding latest .ass file: {str(e)}", exc_info=True)
        return None

@app.route('/api/v1/latest-ass-file', methods=['GET'])
def get_latest_ass_file():
    """
    Returns the path to the latest .ass file in data/uploads.
    """
    try:
        latest_file = _find_latest_ass_file()
        return jsonify({'path': latest_file})
    except Exception as e:
        logger.error(f"Error in get_latest_ass_file: {str(e)}")
        return jsonify({'path': None, 'error': str(e)}), 500


@app.route('/api/v1/latest-image-slides-json', methods=['GET'])
def get_latest_image_slides_json():
    """
    Returns the path to the latest .image_slides.json file in data/uploads.
    """
    try:
        json_files = glob.glob(f'{UPLOADS_DIR}/*.image_slides.json')
        if not json_files:
            return jsonify({'path': None})
        latest_file = max(json_files, key=os.path.getmtime)
        # Extract audio_id from the filename (e.g., 6844824f-a880-4c56-918f-faaebf876ccd.image_slides.json)
        audio_id_match = re.match(r'([a-f0-9-]+)\.image_slides\.json', os.path.basename(latest_file))
        audio_id = audio_id_match.group(1) if audio_id_match else None
        return jsonify({'path': latest_file, 'audio_id': audio_id})
    except Exception as e:
        logger.error(f"Error finding latest image slides JSON: {str(e)}")
        return jsonify({'path': None, 'error': str(e)}), 500


def _find_latest_expressions_file():
    """Helper function to find the latest .expressions.json file in the uploads directory.
    
    Returns:
        str: Path to the latest .expressions.json file, or None if not found
    """
    try:
        # Ensure UPLOADS_DIR is a Path object
        uploads_dir = Path(UPLOADS_DIR)
        if not uploads_dir.exists():
            logger.error(f"Uploads directory not found: {uploads_dir}")
            return None
            
        # Find all .expressions.json files in the uploads directory
        expr_files = list(uploads_dir.glob('*.expressions.json'))
        
        if not expr_files:
            logger.info("No .expressions.json files found in uploads directory")
            return None
            
        # Get the most recently modified file
        latest_file = max(expr_files, key=os.path.getmtime)
        logger.info(f"Found latest expressions file: {latest_file}")
        
        # Return the absolute path as a string
        return str(latest_file.absolute())
        
    except Exception as e:
        logger.error(f"Error finding latest .expressions.json file: {str(e)}", exc_info=True)
        return None


@app.route('/api/v1/latest-expressions-file', methods=['GET'])
def get_latest_expressions_file():
    """
    Returns the path to the latest .expressions.json file in data/uploads.
    
    Returns:
        JSON: { "path": str, "exists": bool, "status": str }
    """
    try:
        expressions_file = _find_latest_expressions_file()
        if expressions_file and os.path.exists(expressions_file):
            return jsonify({
                'status': 'success',
                'path': expressions_file,
                'exists': True
            })
        
        return jsonify({
            'status': 'not_found',
            'message': 'No expressions file found',
            'exists': False
        })
    except Exception as e:
        error_msg = f'Error getting latest expressions file: {str(e)}'
        logger.error(error_msg, exc_info=True)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'exists': False
        }), 500


@app.route('/api/v1/latest-image-slides', methods=['GET'])
def get_latest_image_slides():
    """
    Returns the path to the latest .image_slides.json file in data/uploads.
    
    Returns:
        JSON: { "path": str, "exists": bool, "status": str }
    """
    try:
        # Get the JSON response from get_latest_image_slides_json
        response = get_latest_image_slides_json()
        
        # If we got a response and it has a path
        if response and 'path' in response.get_json():
            image_slides_file = response.get_json()['path']
            
            if image_slides_file and os.path.exists(image_slides_file):
                return jsonify({
                    'status': 'success',
                    'path': image_slides_file,
                    'exists': True
                })
        
        # If we get here, no valid file was found
        return jsonify({
            'status': 'not_found',
                'message': 'No image slides file found',
                'exists': False
            })
    except Exception as e:
        error_msg = f'Error getting latest image slides: {str(e)}'
        logger.error(error_msg, exc_info=True)
        return jsonify({
            'status': 'error',
            'message': error_msg,
            'exists': False
        }), 500


# ── AssetDatabase endpoints ──────────────────────────────────────────


@app.route('/api/v1/assets/search', methods=['POST'])
def asset_search():
    """Unified asset search.

    Request JSON:
    {
        "query": "free text",          // for semantic search
        "tags": ["tag1", "tag2"],       // for tag search
        "emotions": ["grief", "hope"], // for emotion search
        "arc": "Marineford",           // for arc search
        "mode": "semantic",            // "semantic" | "tag" | "emotion" | "arc" | "auto"
        "top_k": 10
    }

    When mode is "auto" (default), the endpoint picks the best mode
    based on which fields are provided, or combines results.
    """
    try:
        from app.utils.assets import get_asset_database

        data = request.json or {}
        query = (data.get('query') or '').strip()
        tags = data.get('tags') or []
        emotions = data.get('emotions') or []
        arc = (data.get('arc') or '').strip()
        mode = (data.get('mode') or 'auto').strip().lower()
        top_k = min(int(data.get('top_k', 10)), 50)

        db = get_asset_database()
        results = []

        if mode == 'semantic' and query:
            results = db.semantic_search(query, top_k=top_k)
        elif mode == 'tag' and tags:
            results = db.tag_search(tags, top_k=top_k)
        elif mode == 'emotion' and emotions:
            results = db.emotion_search(emotions, top_k=top_k)
        elif mode == 'arc' and arc:
            results = db.arc_search(arc, top_k=top_k)
        elif mode == 'auto':
            # Combine available signals into a beat-style query.
            if query or tags or emotions or arc:
                beat = {
                    'text': query,
                    'tags': tags,
                    'emotion': ', '.join(emotions) if emotions else '',
                    'entities': [],
                }
                # Extract entities from query for better ranking.
                if query:
                    from app.utils.slides.image_slides import _extract_context_entities
                    beat['entities'] = _extract_context_entities(query, limit=6)

                # Adjust weights based on what's provided.
                weights = {
                    'semantic': 0.35 if query else 0.0,
                    'tag': 0.25 if tags else 0.0,
                    'emotion': 0.25 if emotions else 0.0,
                    'arc': 0.15 if arc else 0.0,
                    'importance': 0.10,
                }
                # Normalize weights to sum to ~1.
                total_w = sum(weights.values())
                if total_w > 0:
                    weights = {k: v / total_w for k, v in weights.items()}

                results = db.rank_for_beat(beat, top_k=top_k, weights=weights)
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Provide at least one of: query, tags, emotions, arc',
                }), 400
        else:
            return jsonify({
                'status': 'error',
                'message': f'Invalid mode "{mode}" or missing required fields for that mode',
            }), 400

        return jsonify({
            'status': 'success',
            'mode': mode,
            'count': len(results),
            'results': [r.to_dict() for r in results],
        })

    except Exception as e:
        logger.error(f"Asset search error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/assets/rank-for-beat', methods=['POST'])
def asset_rank_for_beat():
    """Rank assets for a storyboard beat.

    Request JSON:
    {
        "text": "Luffy unlocks Gear 5 at Onigashima",
        "beat_type": "reveal",
        "entities": ["Luffy", "Kaidou"],
        "emotion": "hope",
        "tags": ["nika", "gear_5"],
        "top_k": 5
    }

    Returns ranked assets with per-component score breakdowns.
    """
    try:
        from app.utils.assets import get_asset_database

        data = request.json or {}
        top_k = min(int(data.get('top_k', 5)), 50)

        beat = {
            'text': data.get('text', ''),
            'beat_type': data.get('beat_type', ''),
            'entities': data.get('entities', []),
            'emotion': data.get('emotion', ''),
            'tags': data.get('tags', []),
        }

        db = get_asset_database()
        results = db.rank_for_beat(beat, top_k=top_k)

        return jsonify({
            'status': 'success',
            'beat_type': beat['beat_type'],
            'count': len(results),
            'results': [r.to_dict() for r in results],
        })

    except Exception as e:
        logger.error(f"Asset rank-for-beat error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/assets/status', methods=['GET'])
def asset_database_status():
    """Health check for the AssetDatabase.

    Returns index size, embedding status, categories, and cache info.
    """
    try:
        from app.utils.assets import get_asset_database

        db = get_asset_database()
        return jsonify({
            'status': 'success',
            **db.status(),
        })

    except Exception as e:
        logger.error(f"Asset database status error: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'total_records': 0,
        }), 500


# ── Character Relationship endpoints ─────────────────────────────────


@app.route('/api/v1/characters/relationships', methods=['POST'])
def character_relationships():
    """Get ranked relationships for a character.

    Request JSON:
    {
        "character": "Blackbeard",
        "top_k": 10,
        "narration": "optional narration text for context boosting"
    }

    Returns ranked relationships with types and evidence.
    """
    try:
        from app.utils.assets import get_relationship_engine

        data = request.json or {}
        character = (data.get('character') or '').strip()
        if not character:
            return jsonify({'status': 'error', 'message': 'character is required'}), 400

        top_k = min(int(data.get('top_k', 10)), 50)
        narration = data.get('narration')

        engine = get_relationship_engine()
        results = engine.get_relationships(
            character,
            top_k=top_k,
            narration=narration,
        )

        return jsonify({
            'status': 'success',
            'character': character,
            'count': len(results),
            'results': [r.to_dict() for r in results],
        })

    except Exception as e:
        logger.error(f"Character relationship error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/characters/mutual', methods=['POST'])
def character_mutual_connections():
    """Find mutual connections between two characters.

    Request JSON:
    {
        "character_a": "Luffy",
        "character_b": "Blackbeard",
        "top_k": 5
    }
    """
    try:
        from app.utils.assets import get_relationship_engine

        data = request.json or {}
        char_a = (data.get('character_a') or '').strip()
        char_b = (data.get('character_b') or '').strip()
        if not char_a or not char_b:
            return jsonify({
                'status': 'error',
                'message': 'character_a and character_b are required',
            }), 400

        top_k = min(int(data.get('top_k', 5)), 50)

        engine = get_relationship_engine()
        results = engine.get_mutual_connections(char_a, char_b, top_k=top_k)

        return jsonify({
            'status': 'success',
            'character_a': char_a,
            'character_b': char_b,
            'count': len(results),
            'results': [r.to_dict() for r in results],
        })

    except Exception as e:
        logger.error(f"Character mutual connections error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Emotion Curve endpoint ───────────────────────────────────────────


@app.route('/api/v1/emotion-curve', methods=['POST'])
def emotion_curve():
    """Generate an emotion curve from story beats or raw script text.

    Request JSON (option A — pre-classified beats):
    {
        "beats": [
            {"beat_type": "hook", "text": "Why did Shanks..."},
            {"beat_type": "reveal", "text": "The truth is..."},
            ...
        ]
    }

    Request JSON (option B — raw script, auto-analyzed):
    {
        "script": "Why did Shanks sacrifice his arm? Before the Great Pirate Era..."
    }

    Returns emotion scores per beat and a curve summary.
    """
    try:
        from app.utils.slides.emotion_curve import EmotionCurveGenerator

        data = request.json or {}
        beats = data.get('beats')

        if not beats:
            script = (data.get('script') or '').strip()
            if not script:
                return jsonify({
                    'status': 'error',
                    'message': 'Provide either "beats" (list of beat dicts) or "script" (raw text)',
                }), 400

            from app.utils.slides.story_analyzer import StoryAnalyzer
            analyzer = StoryAnalyzer()
            beats = analyzer.analyze(script)

        gen = EmotionCurveGenerator()
        curve = gen.generate(beats)

        return jsonify({
            'status': 'success',
            **curve.to_dict(),
        })

    except Exception as e:
        logger.error(f"Emotion curve error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Motion Planner endpoints ─────────────────────────────────────────


@app.route('/api/v1/motion/plan', methods=['POST'])
def motion_plan():
    """Plan motion style for a single beat.

    Request JSON:
    {
        "emotion": "fear",
        "visual_intent": "curiosity_gap",
        "beat_type": "hook",
        "visual_role": "character"
    }
    """
    try:
        from app.utils.slides.motion_planner import MotionPlanner

        data = request.json or {}
        planner = MotionPlanner()
        plan = planner.plan(
            emotion=data.get('emotion', 'neutral'),
            visual_intent=data.get('visual_intent', ''),
            beat_type=data.get('beat_type', ''),
            visual_role=data.get('visual_role', ''),
            previous_style=data.get('previous_style', ''),
        )
        return jsonify({'status': 'success', **plan.to_dict()})

    except Exception as e:
        logger.error(f"Motion plan error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/motion/plan-sequence', methods=['POST'])
def motion_plan_sequence():
    """Plan motion for a full beat sequence.

    Request JSON:
    {
        "beats": [
            {"emotion": "fear", "beat_type": "hook", "visual_intent": "curiosity_gap"},
            {"emotion": "curious", "beat_type": "evidence", "visual_intent": "proof"},
            ...
        ]
    }
    """
    try:
        from app.utils.slides.motion_planner import MotionPlanner

        data = request.json or {}
        beats = data.get('beats') or []
        if not beats:
            return jsonify({'status': 'error', 'message': 'beats list is required'}), 400

        planner = MotionPlanner()
        plans = planner.plan_sequence(beats)

        return jsonify({
            'status': 'success',
            'count': len(plans),
            'plans': [p.to_dict() for p in plans],
        })

    except Exception as e:
        logger.error(f"Motion plan sequence error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Visual Diversity endpoint ────────────────────────────────────────


@app.route('/api/v1/visual-diversity', methods=['POST'])
def visual_diversity():
    """Score visual diversity of storyboard slides.

    Request JSON:
    {
        "slides": [...],
        "rejection_threshold": 0.7
    }

    Returns per-section scores, rejected indices, and summary.
    """
    try:
        from app.utils.slides.visual_diversity import VisualDiversityScorer

        data = request.json or {}
        slides = data.get('slides') or []
        if not slides:
            return jsonify({'status': 'error', 'message': 'slides list is required'}), 400

        threshold = float(data.get('rejection_threshold', 0.7))
        scorer = VisualDiversityScorer(rejection_threshold=threshold)
        report = scorer.score(slides)

        return jsonify({
            'status': 'success',
            **report.to_dict(),
        })

    except Exception as e:
        logger.error(f"Visual diversity error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500
