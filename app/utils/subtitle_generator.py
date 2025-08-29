import os
import json
import textwrap
import re
from typing import List, Dict, Set, Optional, Callable
import logging
import tempfile
import nltk
import numpy as np
from pydub import AudioSegment, silence

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SubtitleGenerator:
    # One Piece specific terms for enhanced styling
    _CHARACTERS = {
        'luffy', 'zoro', 'sanji', 'nami', 'usopp', 'chopper', 'robin', 'franky', 'brook', 'jinbe',
        'xebec', 'rocks', 'whitebeard', 'roger', 'garp', 'shanks', 'blackbeard', 'kaido', 'big mom',
        'imu', 'harald', 'loki', 'ace', 'sabo', 'oden', 'rayleigh', 'mihawk', 'buggy', 'doflamingo'
    }
    
    _POWERS = {
        'haki', 'devil fruit', 'conqueror', 'armament', 'observation', 'gear', 'gomu gomu',
        'rokushiki', 'rokuogan', 'busoshoku', 'kenbunshoku', 'haoshoku', 'diable jambe',
        'santoryu', 'ashura', 'gear second', 'gear third', 'gear fourth', 'gear fifth'
    }
    
    _LOCATIONS = {
        'mary geoise', 'marineford', 'wano', 'dressrosa', 'alabasta', 'skypiea', 'fishman island',
        'enies lobby', 'impel down', 'sabaody', 'amazon lily', 'elbaf', 'hachinosu', 'god valley',
        'drum island', 'water 7', 'thriller bark', 'whole cake island', 'onigashima', 'zou'
    }
    
    _TITLES = {
        'pirate king', 'yonko', 'emperor', 'warlord', 'shichibukai', 'admiral', 'fleet admiral',
        'vice admiral', 'commander', 'supernova', 'worst generation', 'revolutionary', 'dragon',
        'ancient weapon', 'ancient kingdom', 'void century', 'will of d'
    }
    
    _EMOTIONAL_TERMS = {
        'dream', 'freedom', 'friendship', 'family', 'promise', 'hope', 'courage', 'will',
        'nakama', 'treasure', 'adventure', 'journey', 'destiny', 'legend', 'myth'
    }
    def __init__(self, script_text: str, audio_path: str, style: str = 'epic'):
        """
        Initialize subtitle generator with script, audio paths, and style.

        Args:
            script_text: Text content of the narration script
            audio_path: Path to the audio file
            style: Subtitle style ('epic', 'dramatic', 'manga', 'adventure')
        """
        self.style = style.lower() if style else 'epic'  # Default to 'epic' style
        # Validate script text
        if not script_text or not script_text.strip():
            raise ValueError("Script text cannot be empty")

        # Validate audio file
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Validate and preprocess audio file
        try:
            import wave
            import pydub

            # First check audio format
            with wave.open(audio_path, 'r') as wav:
                channels = wav.getnchannels()
                framerate = wav.getframerate()
                logger.info(
                    f"Audio file details: channels={channels}, framerate={framerate}")

                # If not 16kHz mono, resample it
                if channels != 1 or framerate != 16000:
                    logger.info("Resampling audio to 16kHz mono")
                    audio = pydub.AudioSegment.from_wav(audio_path)
                    audio = audio.set_frame_rate(16000).set_channels(1)

                    # Save resampled audio to temp file
                    self.temp_dir = tempfile.mkdtemp()
                    resampled_path = os.path.join(
                        self.temp_dir, "resampled.wav")
                    audio.export(resampled_path, format="wav")
                    self.audio_path = resampled_path
                    logger.info(f"Saved resampled audio to {resampled_path}")
                else:
                    self.audio_path = audio_path
                    self.temp_dir = tempfile.mkdtemp()
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise

        self.script_text = script_text

        # Log initialization details
        logger.info(
            f"Subtitle generator initialized with audio: {self.audio_path}")
        logger.info(f"Script length: {len(script_text)} characters")

    def detect_silence_gaps(self, min_silence_len=300, silence_thresh=-40):
        """
        Detect silence gaps in the audio file using pydub.
        Returns a list of (start, end) tuples for each non-silent segment.
        """
        audio = AudioSegment.from_wav(self.audio_path)
        non_silent_ranges = silence.detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        # Convert ms to seconds
        return [(start/1000, end/1000) for start, end in non_silent_ranges]

    def generate_timestamps(self, model_size: str = "small", device: str = "cpu") -> List[Dict]:
        """
        Generate word-level timestamps using WhisperX (transcribe + align).

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2)
            device: 'cpu' or 'cuda'

        Returns:
            List of dicts: [{"word": str, "start": float, "end": float}]
        """
        try:
            import whisperx
        except ImportError:
            logger.error(
                "whisperx is not installed. Please install with 'pip install git+https://github.com/m-bain/whisperx'")
            raise

        logger.info(
            f"Loading WhisperX model: {model_size} on {device} (compute_type=float32)")
        model = whisperx.load_model(model_size, device, compute_type="float32")
        audio = whisperx.load_audio(self.audio_path)
        logger.info(f"Transcribing audio with WhisperX...")
        result = model.transcribe(audio)
        logger.info(
            f"Transcription done. Performing alignment for word-level timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device)
        logger.info(f"Alignment done. Extracting word-level timestamps...")
        words = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                words.append({
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"]
                })
        logger.info(
            f"Extracted {len(words)} word-level timestamps from WhisperX.")
        if words:
            logger.info(f"First word: {words[0]}")
        return words

    def group_words_into_phrases(
        self,
        timestamps: List[Dict],
        min_gap: float = 1.0,
        max_phrase_duration: float = 5.0,
        max_words: int = 5,
        use_audio_gaps: bool = True,
        min_silence_len: int = 300,
        silence_thresh: int = -40
    ) -> List[Dict]:
        """
        Hybrid: Group words into phrases using detected silence gaps and max word count.
        min_silence_len and silence_thresh are now adjustable for optimization.
        """
        if not timestamps:
            return []
        logger.info(
            f"Grouping {len(timestamps)} words into phrases (hybrid gap+word)...")
        gap_ranges = self.detect_silence_gaps(
            min_silence_len=min_silence_len, silence_thresh=silence_thresh) if use_audio_gaps else []
        logger.info(
            f"Detected {len(gap_ranges)} non-silent segments (silence gaps)")
        if gap_ranges:
            logger.info(f"First 3 gap ranges: {gap_ranges[:3]}")
        phrases = []
        idx = 0
        n = len(timestamps)
        while idx < n:
            phrase_start = timestamps[idx]['start']
            phrase_end = None
            for gap_start, gap_end in gap_ranges:
                if gap_start > phrase_start:
                    phrase_end = gap_start
                    break
            chunk = []
            chunk_start = idx
            while idx < n and len(chunk) < max_words:
                word = timestamps[idx]
                if phrase_end is not None and word['end'] > phrase_end:
                    break
                chunk.append(word)
                idx += 1
            if chunk:
                start_time = chunk[0]['start']
                end_time = chunk[-1]['end']
                text = ' '.join([w.get('word', '') for w in chunk])
                phrases.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "words": chunk  # Include word-level timing for karaoke
                })
                logger.info(
                    f"Phrase: {start_time:.2f}-{end_time:.2f}s: '{text}'")
            else:
                # Instead of skipping, forcibly add the current word as a single-word phrase
                word = timestamps[idx]
                start_time = word['start']
                end_time = word['end']
                text = word.get('word', '')
                phrases.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text
                })
                logger.warning(
                    f"No chunk formed at idx={idx}, forcibly adding single word phrase: '{text}'")
                idx += 1
        logger.info(f"Created {len(phrases)} hybrid phrases")
        if phrases:
            logger.info(f"Sample phrases:")
            for i, p in enumerate(phrases[:3]):
                logger.info(
                    f"  {i+1}. [{p['start']:.2f}-{p['end']:.2f}s] {p['text'][:60]}...")
        return phrases

    def seconds_to_ass_format(self, seconds: float) -> str:
        """Convert seconds to ASS format (H:MM:SS.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:05.2f}".replace('.', '.')

    def _ensure_nltk(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')

    def _is_one_piece_term(self, word: str, term_set: Set[str]) -> bool:
        """Check if a word matches any term in the given set (case-insensitive)."""
        word_lower = word.lower()
        return any(term in word_lower for term in term_set)

    def _get_style_config(self, style_name: str) -> dict:
        """
        Get the style configuration for the specified style name with enhanced One Piece theming.
        Italics are only applied to One Piece specific terms (character, power, location, title, emotion).
        
        Returns:
            dict: Style configuration with font properties and colors for different term types
        """
        # Base style configurations with enhanced One Piece theming
        styles = {
            'epic': {
                'name': 'Epic Battles',
                'default': {'size': 95, 'bold': False, 'italic': False, 'font': 'NimbusSansNarrow', 'color': 'FFFFFF'},
                'character': {'size': 118, 'bold': True, 'italic': True, 'font': 'NimbusSans-BoldItalic', 'color': '00D4FF', 'outline': 'FF8800', 'glow': 40},
                'power': {'size': 110, 'bold': True, 'italic': True, 'font': 'NimbusSans-BoldItalic', 'color': '0000FF', 'shadow': '000088'},
                'location': {'size': 108, 'bold': True, 'italic': True, 'font': 'NimbusSans-BoldItalic', 'color': 'FF8000'},
                'title': {'size': 112, 'bold': True, 'italic': True, 'font': 'NimbusSans-BoldItalic', 'color': 'FF00AA', 'outline': '00D4FF'},
                'emotion': {'size': 106, 'bold': True, 'italic': True, 'font': 'NimbusSans-BoldItalic', 'color': '800080', 'glow': 40},
                'proper_noun': {'size': 115, 'bold': True, 'italic': False, 'font': 'NimbusSans-Bold', 'color': 'FFFFFF', 'outline': '888888'},
                'verb': {'size': 102, 'bold': False, 'italic': False, 'font': 'NimbusSans', 'color': 'FFFFFF'},
                'noun': {'size': 105, 'bold': False, 'italic': False, 'font': 'NimbusSans', 'color': 'FFFFFF'},
                'other': {'size': 95, 'bold': False, 'italic': False, 'font': 'NimbusSansNarrow', 'color': 'CCCCCC'}
            },
            'dramatic': {
                'name': 'Dramatic Anime',
                'default': {'size': 90, 'bold': False, 'italic': False, 'font': 'DejaVuSans', 'color': 'FFFFFF'},
                'character': {'size': 110, 'bold': True, 'italic': True, 'font': 'DejaVuSans-BoldOblique', 'color': '00D4FF', 'outline': '000000', 'glow': 80},
                'power': {'size': 105, 'bold': True, 'italic': True, 'font': 'DejaVuSans-BoldOblique', 'color': '0000FF'},
                'location': {'size': 108, 'bold': True, 'italic': True, 'font': 'DejaVuSans-BoldOblique', 'color': 'FF8000'},
                'title': {'size': 100, 'bold': True, 'italic': True, 'font': 'DejaVuSans-BoldOblique', 'color': 'FFD700'},
                'emotion': {'size': 100, 'bold': True, 'italic': True, 'font': 'DejaVuSans-BoldOblique', 'color': 'FF69B4'},
                'proper_noun': {'size': 115, 'bold': True, 'italic': False, 'font': 'DejaVuSans-Bold', 'color': '00D4FF'},
                'verb': {'size': 100, 'bold': True, 'italic': False, 'font': 'DejaVuSans-Bold', 'color': 'FFFFFF'},
                'noun': {'size': 100, 'bold': False, 'italic': False, 'font': 'DejaVuSans', 'color': 'FFFFFF'},
                'other': {'size': 90, 'bold': False, 'italic': False, 'font': 'DejaVuSans', 'color': 'FFFFFF'}
            },
            'manga': {
                'name': 'Manga-Inspired',
                'default': {'size': 100, 'bold': True, 'italic': False, 'font': 'NimbusSans-Bold', 'color': '000000', 'outline': 'FFFFFF'},
                'character': {'size': 125, 'bold': True, 'italic': True, 'font': 'Arial-Black', 'color': '000000', 'outline': 'FFFFFF', 'outline_width': 3},
                'power': {'size': 110, 'bold': True, 'italic': True, 'font': 'NimbusSans-BoldItalic', 'color': '00FFFF', 'shadow': 'FFFFFF'},
                'location': {'size': 102, 'bold': True, 'italic': True, 'font': 'NimbusSans-BoldItalic', 'color': 'FFCC99', 'outline': 'FFFFFF'},
                'title': {'size': 120, 'bold': True, 'italic': True, 'font': 'Arial-Black', 'color': 'FFD700', 'outline': 'FFFFFF'},
                'emotion': {'size': 110, 'bold': True, 'italic': True, 'font': 'NimbusSans-BoldItalic', 'color': 'FF69B4', 'outline': 'FFFFFF'},
                'proper_noun': {'size': 125, 'bold': True, 'italic': False, 'font': 'NimbusSans-Bold', 'color': '000000', 'outline': 'FFFFFF'},
                'verb': {'size': 100, 'bold': True, 'italic': False, 'font': 'NimbusSans-Bold', 'color': '000000', 'outline': 'FFFFFF'},
                'noun': {'size': 110, 'bold': True, 'italic': False, 'font': 'NimbusSans-Bold', 'color': '000000', 'outline': 'FFFFFF'},
                'other': {'size': 95, 'bold': True, 'italic': False, 'font': 'NimbusSansNarrow-Bold', 'color': '000000', 'outline': 'FFFFFF'}
            },
            'adventure': {
                'name': 'Adventure Theme',
                'default': {'size': 95, 'bold': False, 'italic': False, 'font': 'URWGothic-Book', 'color': 'EEEEEE'},
                'character': {'size': 112, 'bold': True, 'italic': True, 'font': 'URWGothic-DemiOblique', 'color': '0099FF'},
                'power': {'size': 108, 'bold': True, 'italic': True, 'font': 'URWGothic-DemiOblique', 'color': '0000FF'},
                'location': {'size': 105, 'bold': True, 'italic': True, 'font': 'URWGothic-BookOblique', 'color': 'FF6600'},
                'title': {'size': 110, 'bold': True, 'italic': True, 'font': 'URWGothic-DemiOblique', 'color': 'FF9900'},
                'emotion': {'size': 106, 'bold': True, 'italic': True, 'font': 'URWGothic-BookOblique', 'color': '00FFFF', 'glow': 60},
                'proper_noun': {'size': 110, 'bold': True, 'italic': False, 'font': 'URWGothic-Demi', 'color': '0099FF'},
                'verb': {'size': 100, 'bold': True, 'italic': False, 'font': 'URWGothic-Demi', 'color': 'FFFFFF'},
                'noun': {'size': 100, 'bold': False, 'italic': False, 'font': 'URWGothic-Book', 'color': 'FFFFFF'},
                'other': {'size': 90, 'bold': False, 'italic': False, 'font': 'URWGothic-Book', 'color': 'EEEEEE'}
            }
        }
        
        # Return the requested style or default to 'epic' if not found
        return styles.get(style_name.lower(), styles['epic'])

    def _get_word_style_type(self, word: str, tag: str, style: dict) -> str:
        """Determine the most appropriate style type for a word based on its content and POS tag."""
        word_lower = word.lower()
        
        # Check for One Piece specific terms first
        if self._is_one_piece_term(word, self._CHARACTERS):
            return 'character'
        elif self._is_one_piece_term(word, self._POWERS):
            return 'power'
        elif self._is_one_piece_term(word, self._LOCATIONS):
            return 'location'
        elif self._is_one_piece_term(word, self._TITLES):
            return 'title'
        elif self._is_one_piece_term(word, self._EMOTIONAL_TERMS):
            return 'emotion'
        # Fall back to POS-based styling
        elif tag == 'NNP' or (word.istitle() and len(word) > 2):
            return 'proper_noun'
        elif tag.startswith('NN'):
            return 'noun'
        elif tag.startswith('VB'):
            return 'verb'
        return 'other'

    def _apply_style(self, word: str, style_config: dict) -> str:
        """Apply the given style configuration to a word and return the styled ASS tag."""
        clean_word = word.replace('{', '').replace('}', '')
        style_parts = []
        
        # Add color (default to white if not specified)
        color = style_config.get('color', 'FFFFFF')
        style_parts.append(f'\\c&H{color}&')
        
        # Add outline if specified
        if 'outline' in style_config:
            style_parts.append(f'\\3c&H{style_config["outline"]}&')
        if 'outline_width' in style_config:
            style_parts.append(f'\\bord{style_config["outline_width"]}')
        
        # Add shadow if specified
        if 'shadow' in style_config:
            style_parts.append(f'\\4c&H{style_config["shadow"]}&')
            style_parts.append('\\shad1')
        
        # Add glow effect if specified
        if 'glow' in style_config:
            style_parts.append(f'\\3a&H{style_config["glow"]:02X}')
        
        # Add font size
        style_parts.append(f'\\fs{style_config.get("size", 95)}')
        
        # Add bold/italic
        style_parts.append('\\b1' if style_config.get('bold', False) else '\\b0')
        style_parts.append('\\i1' if style_config.get('italic', False) else '\\i0')
        
        # Add font family
        style_parts.append(f'\\fn{style_config.get("font", "NimbusSans")}')
        
        # Combine all style parts and apply to word
        style_str = '\\'.join(style_parts)
        return f"{{\\{style_str}}}{clean_word}{{\\r}}"

    def _style_phrase(self, phrase: str) -> str:
        """
        Style text with enhanced One Piece theming based on word content and part of speech.
        
        Args:
            phrase: The phrase to style
            
        Returns:
            str: The phrase with ASS formatting codes applied
        """
        self._ensure_nltk()
        tokens = nltk.word_tokenize(phrase)
        tags = nltk.pos_tag(tokens)
        styled = []
        
        # Get the style configuration
        style = self._get_style_config(self.style)
        
        for word, tag in tags:
            # Skip empty words
            if not word.strip():
                continue
                
            # Determine the most appropriate style for this word
            style_type = self._get_word_style_type(word, tag, style)
            word_style = style.get(style_type, style['default'])
            
            # Apply the style to the word
            styled_word = self._apply_style(word, word_style)
            styled.append(styled_word)
            
        return ' '.join(styled)

    def generate_ass_file(self, phrases: List[Dict], output_path: str, resolution: str = '1080x1920') -> None:
        """
        Generate a visually rich, dynamically styled ASS subtitle file using NLTK.
        """
        logger.info(f"Generating visually rich ASS file at {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Parse resolution
        try:
            res_x, res_y = map(int, resolution.lower().split('x'))
        except Exception:
            logger.warning(
                f"Invalid resolution '{resolution}', defaulting to 1080x1920")
            res_x, res_y = 1080, 1920

        header = textwrap.dedent(f"""
            [Script Info]
            Title: Generated Subtitles
            ScriptType: v4.00+

            PlayResX: {res_x}
            PlayResY: {res_y}
            WrapStyle: 0
            ScaledBorderAndShadow: yes
            YCbCr Matrix: TV.601

            [V4+ Styles]
            Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
            Style: Default,NimbusSans,72,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,1,5,30,30,0,1

            [Events]
            Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
            """)
        content = [header.strip() + '\n']

        if not phrases:
            logger.warning("No phrases provided for subtitle generation")
            return

        last_end = -1
        for i, phrase in enumerate(phrases):
            start = self.seconds_to_ass_format(phrase['start'])
            end = self.seconds_to_ass_format(phrase['end'])
            phrase_text = phrase['text']
            phrase_text = phrase_text.replace('{', '').replace('}', '')
            phrase_text = phrase_text.replace('\\N', ' ').replace('\\n', ' ')
            phrase_text = phrase_text.replace('\n', ' ').replace('\r', ' ')
            phrase_text = phrase_text.strip()
            word_count = len(phrase_text.split())
            logger.info(f"Subtitle line {i+1}: {word_count} words: '{phrase_text}'")
            
            # Only skip if the overlap is significant (more than 0.1 seconds)
            if phrase['start'] < last_end:
                overlap = last_end - phrase['start']
                if overlap > 0.1:  # More than 0.1 seconds of overlap
                    logger.warning(f"Skipping significantly overlapping phrase ({overlap:.2f}s): {phrase_text}")
                    continue
                else:
                    logger.debug(f"Allowing minor overlap ({overlap:.2f}s) for: {phrase_text}")
            
            ass_text = self._style_phrase(phrase_text)
            content.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{ass_text}\n")
            last_end = max(last_end, phrase['end'])  # Update to the latest end time

        try:
            with open(output_path, 'w', encoding='utf-8-sig') as f:
                f.writelines(content)
                logger.info(
                    f"Successfully wrote {len(content)} lines to {output_path}")
        except Exception as e:
            logger.error(f"Error writing ASS file: {e}")
            raise

    def _add_phrase_line(self, content, phrase_words, start_time, end_time):
        """Add a full phrase line to the content."""
        if not phrase_words:
            return

        phrase_text = ' '.join(phrase_words)
        phrase_start = self.seconds_to_ass_format(start_time)
        phrase_end = self.seconds_to_ass_format(end_time)
        phrase_line = f"Dialogue: 0,{phrase_start},{phrase_end},Default,,0,0,0,,{phrase_text}\\n"
        logger.debug(f"Adding phrase line: {phrase_line.strip()}")
        content.append(phrase_line)
        logger.info(
            f"Added phrase {len(phrase_words)} words: {phrase_start} --> {phrase_end}: {phrase_text[:50]}...")

    def __del__(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {e}")
