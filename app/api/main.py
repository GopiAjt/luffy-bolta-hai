from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
from datetime import datetime
from typing import Optional
from app.utils.script_generator import generate_script
from app.utils.audio_processor import save_audio_file, convert_to_wav, get_audio_duration, create_uploads_dir, UPLOADS_DIR
from app.utils.subtitle_generator import SubtitleGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../static', static_url_path='')
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
        ass_path = os.path.join(UPLOADS_DIR, f"{os.path.splitext(audio_id)[0]}.ass")
        logger.info(f"Generating ASS file at {ass_path}...")
        subtitle_generator.generate_ass_file(phrases, ass_path)
        
        # Verify ASS file was created
        if not os.path.exists(ass_path) or os.path.getsize(ass_path) < 100:  # Check if file is too small
            logger.error(f"ASS file was not generated properly at {ass_path}")
            return jsonify({
                'error': 'Failed to generate subtitles: ASS file not generated correctly'
            }), 500

        return jsonify({
            'status': 'success',
            'message': 'Subtitles generated successfully',
            'output': {
                'ass_file': os.path.basename(ass_path),
                'phrases': phrases
            }
        })

    except Exception as e:
        logger.error(f"Error generating subtitles: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.get("/api/v1/health")
def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)