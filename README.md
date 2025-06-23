# Luffy Bolta Hai - One Piece Subtitle Analysis & Narration Video Pipeline

A Python project for processing, analyzing, and translating One Piece anime subtitles, and for generating narration videos with clean, phrase-based subtitles.

## Features

- Process and clean subtitle files (SRT format)
- Extract character dialogue and speaking patterns
- Analyze subtitle statistics and trends
- Translate Japanese subtitles to English using Google Cloud Translation API
- Generate visualizations for character interactions and episode patterns
- **NEW:** Generate narration videos with:
  - Script generation (random or topic-based)
  - Audio alignment using WhisperX (word-level timestamps)
  - Phrase-based, bold, clean subtitles (max 5 words/line, no effects)
  - Subtitles well-synced to audio using silence gap detection and word count
  - Burned-in subtitles on vertical (9:16) videos (1080x1920)

## Project Structure

```
luffy-bolta-hai/
├── data/
│   ├── raw/              # Raw subtitle files (not tracked in git)
│   ├── processed/        # Processed subtitle files (not tracked in git)
│   └── analysis/         # Analysis results and visualizations (not tracked in git)
├── scripts/
│   ├── process_subtitles.py    # Subtitle processing script
│   ├── analyze_subtitles.py    # Subtitle statistics analysis
│   ├── analyze_dialogue.py     # Dialogue analysis
│   ├── analyze_characters.py   # Character analysis
│   └── translate_subtitles.py  # Subtitle translation
├── credentials/          # Google Cloud credentials (not tracked in git)
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/luffy-bolta-hai.git
cd luffy-bolta-hai
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up Google Cloud credentials:

   - Create a Google Cloud project
   - Enable the Cloud Translation API
   - Create a service account and download the credentials JSON file
   - Create a `credentials` directory in the project root
   - Place your credentials file in the `credentials` directory
   - Rename it to `google_cloud_credentials.json`

5. Prepare your data:
   - Place your raw subtitle files in `data/raw/`
   - The processed files will be saved in `data/processed/`
   - Analysis results will be saved in `data/analysis/`

## Usage

### Subtitle & Analysis Scripts

1. Process subtitles:

```bash
python scripts/process_subtitles.py
```

2. Analyze subtitles:

```bash
python scripts/analyze_subtitles.py
```

3. Analyze dialogue:

```bash
python scripts/analyze_dialogue.py
```

4. Analyze characters:

```bash
python scripts/analyze_characters.py
```

5. Translate subtitles:

```bash
python scripts/translate_subtitles.py
```

### Narration Video Generation Pipeline (WhisperX-based)

1. **Generate a script** (random or topic-based):

   - Use the FastAPI backend or the provided script generator utility.

2. **Generate audio narration** for the script (TTS or upload your own).

3. **Generate word-level timestamps** using WhisperX:

   - The pipeline uses WhisperX to transcribe and align audio, producing accurate word-level timestamps.
   - No need for aeneas; WhisperX is now fully integrated.

4. **Group words into subtitle phrases**:

   - Phrases are grouped by silence gaps (using pydub) and a max word count (default: 5 words/line).
   - Subtitles are formatted as clean, bold, ASS files (no effects/tags).

5. **Burn subtitles into a vertical (9:16) video**:

   - The video generator uses the phrase-based ASS file and the audio to create a 1080x1920 video with burned-in subtitles.

6. **Preview or download the video** via the FastAPI backend or static output directory.

#### Example (CLI):

```bash
# (Assuming you have a script and audio ready)
python app/utils/subtitle_generator.py --audio path/to/audio.mp3 --script path/to/script.txt --output path/to/output.ass
# Then use the video generator to burn subtitles
python app/utils/video_generator.py --audio path/to/audio.mp3 --sub path/to/output.ass --output path/to/video.mp4
```

#### Example (API):

- Use the `/api/generate` endpoint to trigger the full pipeline from script to video.

### Requirements for WhisperX Pipeline

- Python 3.8+
- torch (see WhisperX docs for compatible versions)
- whisperx
- pydub
- ffmpeg (system dependency)

#### Troubleshooting

- If you see float16 errors on CPU, WhisperX is forced to use `compute_type="float32"`.
- Ensure ffmpeg is installed and in your PATH.
- For best results, use a GPU (but CPU is supported).

## Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- japanize-matplotlib
- networkx
- wordcloud
- google-cloud-translate
- torch
- whisperx
- pydub
- ffmpeg (system)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
