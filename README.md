# Luffy Bolta Hai - One Piece Subtitle Analysis

A Python project for processing, analyzing, and translating One Piece anime subtitles.

## Features

- Process and clean subtitle files (SRT format)
- Extract character dialogue and speaking patterns
- Analyze subtitle statistics and trends
- Translate Japanese subtitles to English using Google Cloud Translation API
- Generate visualizations for character interactions and episode patterns

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 