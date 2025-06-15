from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
from datetime import datetime
from typing import Optional
from app.utils.script_generator import generate_script
from app.utils.audio_processor import save_audio_file, convert_to_wav, get_audio_duration

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
        
    except Exception as e:
        logger.error(f"Error generating script: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }