from flask import Flask, request, jsonify, send_from_directory, send_file, url_for, Response
from flask_cors import CORS
import os
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
import uuid
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
        from app.utils.image_slides import generate_image_slides
        generate_image_slides(
            ass_path=str(ass_file),
            out_path=str(image_slides_json),
            total_duration=duration,
            image_dir=str(image_dir)
        )
        
        # Generate the video from the slides
        from app.utils.generate_slideshow import main as generate_slideshow_video
        generate_slideshow_video(
            json_path=str(image_slides_json),
            image_dir=str(image_dir),
            output_path=str(output_path),
            total_duration=duration
        )
        
        return jsonify({
            'slideshow_video_id': slideshow_id,
            'message': 'Slideshow generated successfully'
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
        subtitle_style = data.get('subtitle_style', 'karaoke')

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
        logger.info(f"Received request data: {data}")
        
        ass_path = data.get('ass_path')
        audio_id = data.get('audio_id')
        out_path = data.get('out_path')
        
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

        # Set default output path if not provided
        if not out_path:
            out_path = str(Path(ass_path).with_suffix('.image_slides.json'))
            logger.info(f"Using default output path: {out_path}")
            
        logger.info(f"Generating image slides with params: ass_path={ass_path}, out_path={out_path}, duration={total_audio_duration}")
        
        # Ensure the image directory exists
        image_dir = Path(IMAGE_SLIDES_DIR)
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Call the image slides generator
        result_path = generate_image_slides(
            ass_path,  # Pass as string to avoid any Path object issues
            out_path, 
            total_audio_duration, 
            image_dir=str(image_dir.absolute())
        )
        
        logger.info(f"Image slides generated successfully at: {result_path}")
        
        # Read and return the result
        with open(result_path, 'r', encoding='utf-8') as f:
            slides = json.load(f)
            
        return jsonify({
            'status': 'success',
            'slides': slides,
            'output_path': result_path
        })
        
    except Exception as e:
        logger.error(f"Error in generate_image_slides_endpoint: {str(e)}", exc_info=True)
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
            from config.config import COMPILED_VIDEO_DIR
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
                    json={'audio_id': audio_id, 'ass_path': 'auto'}
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

        logger.info("Starting final video generation...")
        try:
            final_video_path = generate_final_video(
                audio_id=audio_id,
                slides_json_path=slides_json,
                subtitle_path=subtitle_file,
                expressions_path=expressions_file,
                generate_slides=force_regenerate
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
                'slides_json': slides_json
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
