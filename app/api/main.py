from flask import Flask, request, jsonify, send_from_directory, send_file, url_for, Response
from flask_cors import CORS
import os
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
from app.utils.script_generator import generate_script
from app.utils.audio_processor import save_audio_file, convert_to_wav, get_audio_duration, create_uploads_dir
from app.utils.subtitle_generator import SubtitleGenerator
from app.utils.image_slides import generate_image_slides
from app.utils.generate_final_video import generate_final_video
import re
import json
import glob
from config.config import (
    UPLOADS_DIR,
    IMAGE_SLIDES_DIR,
    EXPRESSIONS_DIR,
    VIDEO_RESOLUTION,
    VIDEO_BACKGROUND_COLOR,
    SUBTITLE_RESOLUTION,
    HOST,
    PORT,
    DEBUG,
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

# Serve static files


@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('../static/js', filename)


@app.route('/')
def index():
    return send_from_directory('../static', 'index.html')


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
        data = request.json

        # Generate script
        script = generate_script()

        return jsonify({
            'status': 'success',
            'message': 'Script generated successfully',
            'output': {
                'script': script
            }
        })
    except Exception as e:
        logger.error(f"Error in generate_script_endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'output': None
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

        # Convert to WAV if not already
        wav_path = file_path.rsplit('.', 1)[0] + '.wav'
        if file_path.lower().endswith('.mp3'):
            convert_to_wav(file_path, wav_path)
            file_path = wav_path

        # Get duration
        duration = get_audio_duration(file_path)

        return jsonify({
            'id': os.path.basename(file_path),
            'duration': duration,
            'message': 'Audio uploaded successfully'
        })

    except ValueError as e:
        return jsonify({
            'error': str(e)
        }), 400

    except Exception as e:
        logger.error(f"Error uploading audio: {str(e)}")
        return jsonify({
            'error': 'Internal server error'
        }), 500


@app.route('/api/v1/generate-subtitles', methods=['POST'])
def generate_subtitles():
    """
    Generate subtitles for uploaded audio and script.

    Request:
    {
        "audio_id": "file_id",
        "script": "Your script text here"
    }
    """
    try:
        data = request.json
        audio_id = data.get('audio_id')
        script = data.get('script')

        if not audio_id or not script:
            return jsonify({
                'error': 'audio_id and script are required'
            }), 400

        # Get audio file path
        audio_path = os.path.join(UPLOADS_DIR, audio_id)
        if not os.path.exists(audio_path):
            return jsonify({
                'error': f'Audio file not found: {audio_id}'
            }), 404

        # Generate subtitles
        subtitle_generator = SubtitleGenerator(script, audio_path)

        # Get word-level timestamps
        logger.info("Generating word-level timestamps...")
        timestamps = subtitle_generator.generate_timestamps()
        logger.info(f"Generated {len(timestamps)} word timestamps")

        if not timestamps:
            logger.error("No timestamps were generated")
            return jsonify({
                'error': 'Failed to generate subtitles: No timestamps generated'
            }), 500

        # Group words into phrases
        logger.info("Grouping words into phrases...")
        phrases = subtitle_generator.group_words_into_phrases(timestamps)
        logger.info(f"Grouped into {len(phrases)} phrases")

        if not phrases:
            logger.error("No phrases were generated from timestamps")
            return jsonify({
                'error': 'Failed to generate subtitles: No phrases generated'
            }), 500

        # Generate ASS file
        ass_path = os.path.join(
            UPLOADS_DIR, f"{os.path.splitext(audio_id)[0]}.ass")
        logger.info(f"Generating ASS file at {ass_path}...")
        # Use the same resolution as the video generator
        subtitle_generator.generate_ass_file(
            phrases, ass_path, resolution=SUBTITLE_RESOLUTION)

        # Verify ASS file was created
        if not os.path.exists(ass_path) or os.path.getsize(ass_path) < 100:
            logger.error(f"ASS file was not generated properly at {ass_path}")
            return jsonify({
                'error': 'Failed to generate subtitles: ASS file not generated correctly'
            }), 500

        # --- Expression Mapping Integration ---
        try:
            from app.utils.expression_mapper import gemini_expression_mapping_from_ass, parse_ass_file
            gemini_api_key = os.environ.get('GEMINI_API_KEY')
            expr_json_path = ass_path.rsplit('.', 1)[0] + '.expressions.json'
            if gemini_api_key:
                logger.info("Running Gemini-based expression mapping...")
                expressions = gemini_expression_mapping_from_ass(
                    ass_path, gemini_api_key)
            else:
                logger.info(
                    "No Gemini API key found, using rule-based expression mapping...")
                expressions = parse_ass_file(ass_path)
            with open(expr_json_path, 'w', encoding='utf-8') as f:
                json.dump(expressions, f, indent=2, ensure_ascii=False)
            logger.info(f"Expression mapping written to {expr_json_path}")
        except Exception as e:
            logger.error(f"Expression mapping failed: {e}")
            expr_json_path = None
            expressions = None

        return jsonify({
            'status': 'success',
            'message': 'Subtitles generated successfully',
            'output': {
                'ass_file': os.path.basename(ass_path),
                'phrases': phrases,
                'expressions_file': os.path.basename(expr_json_path) if expr_json_path else None,
                'expressions': expressions
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
        ass_path = data.get('ass_path')
        audio_id = data.get('audio_id') # Get audio_id
        out_path = data.get('out_path')
        if not ass_path or not audio_id: # Validate audio_id
            return jsonify({'error': 'ass_path and audio_id are required'}), 400
        
        # Get audio duration
        audio_path = UPLOADS_DIR / audio_id
        if not audio_path.exists():
            return jsonify({'error': f'Audio file not found: {audio_id}'}), 404
        total_audio_duration = get_audio_duration(str(audio_path))
        logger.info(f"Total audio duration for image slides: {total_audio_duration:.2f}s")

        if not out_path:
            out_path = ass_path.rsplit('.', 1)[0] + '.image_slides.json'
        result_path = generate_image_slides(
            ass_path, out_path, total_audio_duration, image_dir=IMAGE_SLIDES_DIR)
        with open(result_path, 'r', encoding='utf-8') as f:
            slides = json.load(f)
        return jsonify({
            'status': 'success',
            'slides': slides,
            'output_path': result_path
        })
    except Exception as e:
        logger.error(f"Error in generate_image_slides_endpoint: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route("/api/v1/health")
def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@app.route('/api/v1/download/<filename>')
def download_file(filename):
    """
    Download a file from the uploads directory.
    """
    try:
        # Security check to prevent directory traversal
        if '..' in filename or filename.startswith('/'):
            logger.warning(f"Invalid file path attempted: {filename}")
            return jsonify({'error': 'Invalid file path'}), 400
        file_path = os.path.join(UPLOADS_DIR, filename)
        logger.info(f"Download requested for file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found for download: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        # Set appropriate MIME type based on file extension
        mimetype = 'video/x-matroska' if filename.lower().endswith('.mkv') else None
        logger.info(f"Serving file {file_path} with mimetype {mimetype}")
        return send_from_directory(
            UPLOADS_DIR,
            filename,
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
    Request JSON:
    {
        "audio_id": "file_id",
        "slides_json": "path/to/slides.json",
        "subtitle_file": "path/to/subtitle.ass",
        "expressions_file": "path/to/expressions.json" (optional)
    }
    """
    try:
        data = request.json
        audio_id = data.get('audio_id')
        slides_json = data.get('slides_json')
        subtitle_file = data.get('subtitle_file')
        expressions_file = data.get('expressions_file')

        if not all([audio_id, slides_json, subtitle_file]):
            return jsonify({'error': 'audio_id, slides_json, and subtitle_file are required'}), 400

        final_video_path = generate_final_video(
            audio_id=audio_id,
            slides_json_path=slides_json,
            subtitle_path=subtitle_file,
            expressions_path=expressions_file,
        )

        video_filename = os.path.basename(final_video_path)
        video_url = url_for('download_file', filename=video_filename, _external=True)

        return jsonify({
            'status': 'success',
            'message': 'Final video generated successfully',
            'video_file': video_filename,
            'download_url': video_url,
        })

    except Exception as e:
        logger.error(f"Error in generate_final_video_endpoint: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/v1/latest-ass-file', methods=['GET'])
def get_latest_ass_file():
    """
    Returns the path to the latest .ass file in data/uploads.
    """
    try:
        ass_files = glob.glob(f'{UPLOADS_DIR}/*.ass')
        if not ass_files:
            return jsonify({'path': None})
        latest_file = max(ass_files, key=os.path.getmtime)
        return jsonify({'path': latest_file})
    except Exception as e:
        logger.error(f"Error finding latest .ass file: {str(e)}")
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
        logger.error(f"Error finding latest image slides json: {str(e)}")
        return jsonify({'path': None, 'error': str(e)}), 500



