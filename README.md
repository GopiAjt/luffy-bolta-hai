# Luffy Bolta Hai - One Piece AI Voice & Video Generator

This project is a web application that uses AI to generate video clips of the anime character Luffy, with user-provided audio and subtitles. The application leverages Retrieval-Augmented Generation (RAG) to create scripts and map character expressions to the dialogue.

## Features

*   **AI-powered script generation:** Creates scripts based on user prompts using a RAG model trained on One Piece subtitles.
*   **Audio-to-video synchronization:** Generates a video slideshow that matches the duration of the user-uploaded audio.
*   **Dynamic expression mapping:** Analyzes the script and maps Luffy's facial expressions to the dialogue.
*   **Subtitle generation:** Creates subtitles from the generated script and synchronizes them with the audio.
*   **Web-based interface:** Provides a user-friendly interface for generating and previewing videos.

## Project Structure

The project is organized as follows:

*   `app/`: Contains the main Flask application.
    *   `api/`: Defines the API endpoints for the web application.
    *   `core/`: Core logic for PDF processing, RAG models, and text processing.
    *   `static/`: Frontend assets (HTML, CSS, JavaScript).
    *   `utils/`: Utility scripts for audio processing, video generation, etc.
*   `config/`: Configuration files.
*   `credentials/`: API keys and other credentials.
*   `data/`: Raw and processed data, including subtitles and vector databases.
*   `scripts/`: Standalone scripts for data analysis, model training, and video generation.
*   `tests/`: Test files.

## Setup and Running

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up credentials:**
    *   Place your Google Cloud credentials in a file named `google_cloud_credentials.json` inside the `credentials/` directory.

3.  **Process data and train the RAG model:**
    *   Run the scripts in the `scripts/` directory to process subtitles, analyze characters and dialogue, and build the RAG model.

4.  **Start the server:**
    ```bash
    python run_server.py
    ```

5.  **Access the application:**
    *   Open your web browser and navigate to `http://127.0.0.1:5000`.

## API Endpoints

The following API endpoints are available:

*   `POST /api/v1/generate-script`: Generates a script from a user prompt.
*   `POST /api/v1/upload-audio`: Uploads an audio file.
*   `POST /api/v1/generate-subtitles`: Generates subtitles for the uploaded audio and script.
*   `POST /api/v1/generate-final-video`: Creates the final video with audio, subtitles, and expressions.
*   `GET /api/v1/latest-ass-file`: Returns the path to the latest generated subtitle file.
*   `GET /api/v1/latest-expressions-file`: Returns the path to the latest generated expressions file.
*   `GET /api/v1/download/<filename>`: Downloads a generated file.

## Scripts

The `scripts/` directory contains various standalone scripts for data processing, analysis, and model training:

*   `analyze_characters.py`: Analyzes character dialogues and relationships.
*   `analyze_dialogue.py`: Analyzes dialogue patterns and emotional content.
*   `analyze_subtitles.py`: Performs statistical analysis on subtitle files.
*   `expression_mapper.py`: Maps character expressions to dialogue.
*   `generate_script.py`: Generates a script using the RAG model.
*   `generate_video_with_expressions.py`: Generates a video with character expressions.
*   `process_subtitles.py`: Processes raw subtitle files into a structured format.
*   `setup.py`: Sets up the project environment.
*   `test_rag.py`: Tests the RAG model.
*   `translate_subtitles.py`: Translates subtitles to English.
*   `translate_subtitles_libre.py`: Translates subtitles using the LibreTranslate API.

## RAG Model

The project uses a Retrieval-Augmented Generation (RAG) model to generate scripts. The RAG model is built using `langchain` and `llama-index` and is trained on a large corpus of One Piece subtitles. The core logic for the RAG model can be found in `app/core/rag_processor.py` and `app/core/subtitle_rag.py`.