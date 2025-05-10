# One Piece Hindi Voiceover Script Generator

A Retrieval-Augmented Generation (RAG) system that automatically generates Hindi voiceover scripts from One Piece manga chapters.

## Features

- PDF to image conversion with preprocessing
- Speech bubble detection using YOLOv8
- Multi-language OCR (Japanese/English) for text extraction
- Character attribution based on bubble geometry
- Vector database for storing and retrieving past dialogues
- Hindi script generation with character-specific tone preservation
- REST API for easy integration

## Project Structure

```
.
├── app/                    # Main application code
│   ├── api/               # FastAPI endpoints
│   ├── core/              # Core functionality
│   ├── models/            # ML models and utilities
│   └── utils/             # Helper functions
├── data/                  # Data storage
│   ├── raw/              # Raw manga PDFs
│   ├── processed/        # Processed images and text
│   └── vector_db/        # Vector database storage
├── tests/                # Unit tests
├── config/               # Configuration files
└── scripts/              # Utility scripts
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-jpn
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Start the API server:
```bash
uvicorn app.api.main:app --reload
```

2. Process a new manga chapter:
```bash
python scripts/process_chapter.py --input path/to/chapter.pdf
```

3. Generate Hindi script:
```bash
python scripts/generate_script.py --chapter 1 --page 1
```

## API Endpoints

- `POST /api/v1/process-chapter`: Upload and process a new chapter
- `GET /api/v1/generate-script`: Generate Hindi script for a specific page
- `GET /api/v1/characters`: List all characters with their dialogue history

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 