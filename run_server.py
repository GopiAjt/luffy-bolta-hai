import os
from app.api.main import app
from config.config import HOST, PORT, DEBUG

if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get port from environment variable or use default
    port = int(os.getenv('FLASK_PORT', PORT))
    
    # Start Flask server
    app.run(host=HOST, port=port, debug=DEBUG)
