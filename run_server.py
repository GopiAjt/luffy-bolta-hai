import os
from app.api.main import app

if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get port from environment variable or use default
    port = int(os.getenv('FLASK_PORT', '5050'))
    
    # Start Flask server
    app.run(host='0.0.0.0', port=port, debug=True)
