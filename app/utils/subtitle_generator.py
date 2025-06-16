import os
import json
from typing import List, Dict
import logging
import tempfile
from pathlib import Path
from aeneas.executetask import ExecuteTask
from aeneas.task import Task

logger = logging.getLogger(__name__)

class SubtitleGenerator:
    def __init__(self, script_text: str, audio_path: str):
        """
        Initialize subtitle generator with script and audio paths.
        
        Args:
            script_text: Text content of the narration script
            audio_path: Path to the audio file
        """
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
                logger.info(f"Audio file details: channels={channels}, framerate={framerate}")
                
                # If not 16kHz mono, resample it
                if channels != 1 or framerate != 16000:
                    logger.info("Resampling audio to 16kHz mono")
                    audio = pydub.AudioSegment.from_wav(audio_path)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    
                    # Save resampled audio to temp file
                    self.temp_dir = tempfile.mkdtemp()
                    resampled_path = os.path.join(self.temp_dir, "resampled.wav")
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
        logger.info(f"Subtitle generator initialized with audio: {self.audio_path}")
        logger.info(f"Script length: {len(script_text)} characters")

    def generate_timestamps(self) -> List[Dict]:
        """
        Generate word-level timestamps using forced alignment.
        
        Returns:
            List of dictionaries containing word timestamps
        """
        try:
            # Create temporary files for input
            text_file = os.path.join(self.temp_dir, "script.txt")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(self.script_text)
                logger.info(f"Wrote script text to {text_file}:")
                logger.info(f"Script content length: {len(self.script_text)}")
                logger.info(f"Script content preview: {self.script_text[:100]}...")

            # Configure task with more detailed parameters
            config_string = (
                "task_language=eng|"
                "is_text_type=plain|"
                "os_task_file_format=json|"
                "os_task_file_level=3|"  # Maximum verbosity
                "os_task_file_name=syncmap.json|"
                "task_adjust_boundary_nonspeech_min=0.5|"
                "task_adjust_boundary_nonspeech_string=REMOVE|"
                "task_adjust_boundary_algorithm=percent|"
                "task_adjust_boundary_percent_value=50"
            )
            
            task = Task(config_string=config_string)
            task.audio_file_path_absolute = os.path.abspath(self.audio_path)
            task.text_file_path_absolute = os.path.abspath(text_file)
            task.sync_map_file_path_absolute = os.path.abspath(os.path.join(self.temp_dir, "syncmap.json"))
            
            # Ensure output directory exists and is writable
            os.makedirs(os.path.dirname(task.sync_map_file_path_absolute), exist_ok=True)

            # Log task details
            logger.info(f"Task audio path: {task.audio_file_path_absolute}")
            logger.info(f"Task text path: {task.text_file_path_absolute}")
            logger.info(f"Task output path: {task.sync_map_file_path_absolute}")
            logger.info(f"Temporary directory: {self.temp_dir}")
            
            # Verify file permissions
            if not os.access(self.audio_path, os.R_OK):
                logger.error(f"Audio file not readable: {self.audio_path}")
                raise PermissionError(f"Cannot read audio file: {self.audio_path}")
            
            if not os.access(text_file, os.R_OK):
                logger.error(f"Text file not readable: {text_file}")
                raise PermissionError(f"Cannot read text file: {text_file}")
            
            # Check if files exist before task execution
            logger.info("Checking file existence before task execution...")
            for file_path in [self.audio_path, text_file]:
                if not os.path.exists(file_path):
                    logger.error(f"File does not exist: {file_path}")
                    raise FileNotFoundError(f"Required file not found: {file_path}")
                logger.info(f"File exists and is readable: {file_path}")
            
            # Execute the task
            executor = ExecuteTask(task)
            logger.info("Starting aeneas task execution...")
            
            try:
                # Execute with more detailed logging
                import logging as pylogging
                import sys
                
                # Redirect aeneas output to our logger
                class AeneasLogger:
                    def write(self, message):
                        if message.strip():
                            logger.info(f"Aeneas output: {message.strip()}")
                    def flush(self):
                        pass
                
                # Save original stdout/stderr
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                
                # Redirect stdout/stderr
                sys.stdout = AeneasLogger()
                sys.stderr = AeneasLogger()
                
                # Set up aeneas logging
                pylogging.basicConfig(level=pylogging.DEBUG)
                logger.info("Aeneas configuration:")
                
                # Log configuration using safe attribute access
                try:
                    # Try to get the configuration as a string
                    config_str = str(task.configuration)
                    logger.info(f"Configuration: {config_str}")
                    
                    # Log individual parameters if they exist
                    if hasattr(task.configuration, '__getitem__'):
                        for param in ['task_language', 'is_text_type', 'os_task_file_format']:
                            try:
                                logger.info(f"{param}: {task.configuration[param]}")
                            except KeyError:
                                logger.debug(f"Configuration parameter not found: {param}")
                except Exception as e:
                    logger.warning(f"Could not log configuration: {str(e)}")
                
                # Log paths and configuration
                logger.info(f"Audio path: {task.audio_file_path_absolute}")
                logger.info(f"Text path: {task.text_file_path_absolute}")
                logger.info(f"Output path: {task.sync_map_file_path_absolute}")
                logger.info(f"Temporary directory: {self.temp_dir}")
                
                # Execute task with detailed error handling
                try:
                    # Enable debug output from aeneas
                    import aeneas.globalfunctions as gf
                    gf.PRINT_INFO = True
                    gf.PRINT_WARNING = True
                    gf.PRINT_ERROR = True
                    gf.PRINT_DEBUG = True
                    
                    # Set maximum verbosity
                    executor.verbosity_level = 3
                    
                    # Create a buffer to capture output
                    from io import StringIO
                    import sys
                    
                    # Save original stdout/stderr
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    
                    # Create string buffers
                    stdout_buffer = StringIO()
                    stderr_buffer = StringIO()
                    
                    try:
                        # Redirect stdout/stderr
                        sys.stdout = stdout_buffer
                        sys.stderr = stderr_buffer
                        
                        # Try with default parameters first
                        logger.info("Executing aeneas task...")
                        
                        # Log task details
                        logger.info(f"Task audio path: {task.audio_file_path_absolute}")
                        logger.info(f"Task text path: {task.text_file_path_absolute}")
                        logger.info(f"Output path: {task.sync_map_file_path_absolute}")
                        
                        # Verify input files exist and are readable
                        if not os.path.isfile(task.audio_file_path_absolute):
                            raise FileNotFoundError(f"Audio file not found: {task.audio_file_path_absolute}")
                        if not os.path.isfile(task.text_file_path_absolute):
                            raise FileNotFoundError(f"Text file not found: {task.text_file_path_absolute}")
                            
                        # Execute with error handling
                        try:
                            executor.execute()
                        except Exception as e:
                            # Check if output was created despite the exception
                            if os.path.exists(task.sync_map_file_path_absolute):
                                logger.warning(f"Task raised exception but output file exists: {e}")
                            else:
                                raise
                        
                        # Get captured output
                        stdout_output = stdout_buffer.getvalue()
                        stderr_output = stderr_buffer.getvalue()
                        
                        if stdout_output:
                            logger.info("Aeneas stdout:\n" + stdout_output)
                        if stderr_output:
                            logger.error("Aeneas stderr:\n" + stderr_output)
                            
                    finally:
                        # Restore stdout/stderr
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                    
                    # Define possible output locations
                    possible_outputs = [
                        task.sync_map_file_path_absolute,
                        os.path.join(os.path.dirname(task.sync_map_file_path_absolute), 'output.json'),
                        os.path.join(os.path.dirname(task.sync_map_file_path_absolute), 'output.txt')
                    ]
                    
                    # Try the CLI fallback if Python API fails
                    logger.warning("Python API failed, trying CLI fallback...")
                    if not self._run_aeneas_cli(
                        task.audio_file_path_absolute,
                        task.text_file_path_absolute,
                        task.sync_map_file_path_absolute
                    ):
                        logger.warning("CLI fallback failed, checking for any output files...")
                    
                    # Check which output file was created
                    output_found = False
                    output_path = task.sync_map_file_path_absolute
                    
                    if not os.path.exists(output_path):
                        # Check other possible output locations
                        for path in possible_outputs:
                            if os.path.exists(path):
                                output_path = path
                                logger.info(f"Found output file at alternative location: {output_path}")
                                output_found = True
                                break
                    else:
                        output_found = True
                    
                    if not output_found:
                        raise RuntimeError("No output files were generated by aeneas")
                        
                    # Update the sync map file path to the found file
                    task.sync_map_file_path_absolute = output_path
                    logger.info(f"Using sync map file: {output_path}")
                    
                    # Log file details
                    logger.info(f"Output file: {task.sync_map_file_path_absolute}")
                    logger.info(f"File size: {os.path.getsize(task.sync_map_file_path_absolute)} bytes")
                    logger.info(f"File permissions: {oct(os.stat(task.sync_map_file_path_absolute).st_mode)[-3:]}")
                    
                except Exception as e:
                    logger.error(f"Error during aeneas execution: {str(e)}", exc_info=True)
                    
                    # Log directory contents for debugging
                    if os.path.exists(self.temp_dir):
                        logger.error("Temporary directory contents:")
                        for item in os.listdir(self.temp_dir):
                            item_path = os.path.join(self.temp_dir, item)
                            try:
                                size = os.path.getsize(item_path)
                                logger.error(f"  {item} ({size} bytes)")
                                if item.endswith(('.txt', '.json', '.log')) and size < 1024*1024:  # Only read small text files
                                    try:
                                        with open(item_path, 'r') as f:
                                            content = f.read()
                                            logger.error(f"Content of {item}: {content[:500]}")
                                    except:
                                        pass
                            except:
                                logger.error(f"  {item} (error getting size)")
                    
                    # Restore original stdout/stderr before raising
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    
                    raise RuntimeError(f"Failed to generate subtitle timestamps: {str(e)}")
                
                # Restore original stdout/stderr on success
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                # Check if sync map file exists and validate it
                if not os.path.exists(task.sync_map_file_path_absolute):
                    logger.error(f"Sync map file not found after task execution: {task.sync_map_file_path_absolute}")
                    logger.error("Directory contents:")
                    if os.path.exists(self.temp_dir):
                        for item in os.listdir(self.temp_dir):
                            logger.error(f"  {item}")
                        # Check file permissions
                        if os.path.exists(task.sync_map_file_path_absolute):
                            logger.error(f"Sync map file exists but cannot be accessed: {os.access(task.sync_map_file_path_absolute, os.R_OK)}")
                            logger.error(f"File permissions: {oct(os.stat(task.sync_map_file_path_absolute).st_mode)[-3:]}")
                    
                    # Check if any other files were created
                    logger.error("Checking for other output files:")
                    for file in os.listdir(self.temp_dir):
                        if file != "resampled.wav" and file != "script.txt":
                            logger.error(f"Found unexpected file: {file}")
                            if file.endswith(".json"):
                                try:
                                    with open(os.path.join(self.temp_dir, file), 'r') as f:
                                        content = f.read()
                                        logger.error(f"Content of {file}: {content[:100]}...")
                                except Exception as e:
                                    logger.error(f"Could not read {file}: {str(e)}")
                    
                    raise FileNotFoundError(f"Sync map file not found at {task.sync_map_file_path_absolute}")
                
                # Read and parse the sync map
                with open(task.sync_map_file_path_absolute, "r", encoding="utf-8") as f:
                    content = f.read()
                    logger.info(f"Sync map file content length: {len(content)}")
                    logger.debug(f"Sync map content: {content[:500]}...")  # Log first 500 chars
                    
                    try:
                        # Parse the sync map
                        sync_map_content = json.loads(content)
                        logger.info(f"Successfully parsed sync map: {type(sync_map_content)}")
                        
                        # Log basic structure for debugging
                        if isinstance(sync_map_content, dict):
                            logger.info(f"Sync map keys: {list(sync_map_content.keys())}")
                            if 'fragments' in sync_map_content:
                                logger.info(f"Found {len(sync_map_content['fragments'])} fragments")
                                if sync_map_content['fragments']:
                                    logger.debug(f"First fragment: {sync_map_content['fragments'][0]}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse sync map JSON: {str(e)}")
                        raise
                    
                    # Log the sync map structure for debugging
                    logger.info(f"Successfully parsed sync map with {len(sync_map_content.get('fragments', []))} fragments")
                    
                    # Convert to the expected format
                    timestamps = []
                    if 'fragments' in sync_map_content and isinstance(sync_map_content['fragments'], list):
                        logger.info(f"Processing {len(sync_map_content['fragments'])} fragments")
                        for i, fragment in enumerate(sync_map_content['fragments']):
                            try:
                                if not isinstance(fragment, dict):
                                    logger.warning(f"Skipping invalid fragment at index {i}: {fragment}")
                                    continue
                                    
                                # Get the text from the fragment
                                text = ' '.join(fragment.get('lines', [''])).strip()
                                if not text or text == '---':
                                    logger.debug(f"Skipping empty or separator fragment: {text}")
                                    continue
                                    
                                # Create timestamp entry
                                timestamp = {
                                    'word': text,  # Store the full text as word for now
                                    'start': float(fragment.get('begin', 0)),
                                    'end': float(fragment.get('end', 0)),
                                    'text': text  # Also store in text for backward compatibility
                                }
                                
                                logger.debug(f"Added timestamp {i}: {timestamp}")
                                timestamps.append(timestamp)
                                
                            except (ValueError, TypeError) as e:
                                logger.error(f"Error parsing fragment {i} ({fragment}): {e}")
                                continue
                        
                        logger.info(f"Generated {len(timestamps)} valid timestamps")
                        if timestamps:
                            logger.info(f"First timestamp: {timestamps[0]}")
                        else:
                            logger.warning("No valid timestamp fragments found in sync map")
                        
                        # Return the timestamps if we have any
                        if timestamps:
                            return timestamps
                            
                        # If we get here, no timestamps were generated
                        logger.error("No valid timestamps could be generated from the sync map")
                        logger.error(f"Sync map content: {sync_map_content}")
                        return []
                    else:
                        logger.error(f"Unexpected sync map format: {sync_map_content.keys() if isinstance(sync_map_content, dict) else 'not a dict'}")
                        raise ValueError("Invalid sync map format: missing 'fragments' array")
                
                # Convert sync map to word timestamps
                timestamps = []
                for fragment in timestamps:
                    if fragment["id"] == "fragment_0":
                        continue
                    
                    # Extract word-level timestamps
                    for word in fragment["fragments"]:
                        timestamps.append({
                            "word": word["text"],
                            "start": word["begin"],
                            "end": word["end"]
                        })

                return timestamps

            except Exception as e:
                # Capture full traceback for debugging
                import traceback
                logger.error(f"Error during task execution: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                logger.error(f"Task configuration: {config_string}")
                logger.error(f"Temporary directory contents:")
                if os.path.exists(self.temp_dir):
                    for item in os.listdir(self.temp_dir):
                        logger.error(f"  {item}")
                    # Check file permissions
                    if os.path.exists(task.sync_map_file_path_absolute):
                        logger.error(f"Sync map file exists but cannot be accessed: {os.access(task.sync_map_file_path_absolute, os.R_OK)}")
                raise

            # Convert sync map to word timestamps
            timestamps = []
            try:
                if not sync_map_content or 'fragments' not in sync_map_content:
                    logger.error(f"Invalid sync map format: {sync_map_content}")
                    return []
                
                for fragment in sync_map_content['fragments']:
                    if not isinstance(fragment, dict):
                        logger.warning(f"Skipping invalid fragment: {fragment}")
                        continue
                        
                    try:
                        word = {
                            'word': ' '.join(fragment.get('lines', [''])),
                            'start': float(fragment.get('begin', 0)),
                            'end': float(fragment.get('end', 0))
                        }
                        timestamps.append(word)
                        logger.debug(f"Added word: {word}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error processing fragment {fragment}: {e}")
                        continue
                
                logger.info(f"Generated {len(timestamps)} word timestamps")
                if timestamps:
                    logger.debug(f"First timestamp: {timestamps[0]}")
                
                return timestamps
                
            except Exception as e:
                logger.error(f"Error processing sync map: {str(e)}", exc_info=True)
                raise

        except Exception as e:
            logger.error(f"Error in generate_timestamps: {str(e)}")
            logger.error(f"Current directory: {os.getcwd()}")
            logger.error(f"Temporary directory contents:")
            if os.path.exists(self.temp_dir):
                for item in os.listdir(self.temp_dir):
                    logger.error(f"  {item}")
            raise

    def group_words_into_phrases(self, timestamps: List[Dict], min_gap: float = 1.0, max_phrase_duration: float = 5.0) -> List[Dict]:
        """
        Group words into phrases based on timing gaps and punctuation.
        
        Args:
            timestamps: List of word timestamps
            min_gap: Minimum gap (in seconds) to split phrases
            max_phrase_duration: Maximum duration for a single phrase in seconds
            
        Returns:
            List of phrases with start/end times and text
        """
        if not timestamps:
            return []
            
        logger.info(f"Grouping {len(timestamps)} words into phrases...")
        
        # Punctuation that typically ends a sentence
        SENTENCE_ENDERS = {'.', '!', '?', '...'}
        # Punctuation that might indicate a natural break
        PHRASE_BREAKS = {',', ';', ':', '—', '–', '-', '...'}
        
        phrases = []
        current_phrase = []
        
        for i, word in enumerate(timestamps):
            word_text = word.get('word', '').strip()
            if not word_text:
                continue
                
            current_phrase.append(word)
            
            # Check if we should end the current phrase
            should_end = False
            
            # Check for end of input
            if i == len(timestamps) - 1:
                should_end = True
            else:
                next_word = timestamps[i + 1]
                time_gap = next_word["start"] - word["end"]
                
                # Check for sentence enders
                if word_text[-1] in SENTENCE_ENDERS:
                    should_end = True
                # Check for significant time gap
                elif time_gap > min_gap:
                    should_end = True
                # Check if phrase is getting too long
                elif (word["end"] - current_phrase[0]["start"]) > max_phrase_duration:
                    # Find the last natural break if possible
                    last_break = -1
                    for j in range(len(current_phrase) - 1, -1, -1):
                        if current_phrase[j]['word'][-1] in PHRASE_BREAKS.union(SENTENCE_ENDERS):
                            last_break = j
                            break
                    
                    if last_break > 0 and len(current_phrase) - last_break > 2:
                        # Split at the last natural break
                        phrase_part = current_phrase[:last_break + 1]
                        current_phrase = current_phrase[last_break + 1:]
                        
                        if phrase_part:
                            phrases.append({
                                "start": phrase_part[0]["start"],
                                "end": phrase_part[-1]["end"],
                                "text": " ".join(w["word"] for w in phrase_part).strip()
                            })
                        continue
                    else:
                        should_end = True
            
            if should_end and current_phrase:
                # Combine words into a single string, preserving original spacing
                phrase_text = []
                for j, w in enumerate(current_phrase):
                    word_text = w["word"]
                    # Add space before word unless it's punctuation or first word
                    if j > 0 and not any(word_text.startswith(p) for p in '.,!?;:') and not current_phrase[j-1]["word"].endswith('('):
                        phrase_text.append(' ')
                    phrase_text.append(word_text)
                
                phrases.append({
                    "start": current_phrase[0]["start"],
                    "end": current_phrase[-1]["end"],
                    "text": ''.join(phrase_text).strip()
                })
                current_phrase = []
        
        # Log some debug info
        logger.info(f"Created {len(phrases)} phrases")
        if phrases:
            logger.info(f"Sample phrases:")
            for i, p in enumerate(phrases[:3]):
                logger.info(f"  {i+1}. [{p['start']:.2f}-{p['end']:.2f}s] {p['text'][:60]}...")
        
        return phrases

    def seconds_to_ass_format(self, seconds: float) -> str:
        """Convert seconds to ASS format (H:MM:SS.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:05.2f}".replace('.', '.')

    def generate_ass_file(self, timestamps: List[Dict], output_path: str, word_by_word: bool = True):
        """
        Generate an ASS subtitle file from timestamps with word-by-word timing.
        
        Args:
            timestamps: List of word timestamps with start/end times
            output_path: Path to save the ASS file
            word_by_word: If True, create word-by-word timing. If False, use phrase timing.
        """
        if not timestamps:
            logger.warning("No timestamps provided for ASS file generation")
            return
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            # Write ASS header
            import textwrap
            f.write(textwrap.dedent("""
                                    [Script Info]
                                    Title: Generated Subtitles
                                    ScriptType: v4.00+
                                    PlayResX: 1920
                                    PlayResY: 1080
                                    WrapStyle: 0
                                    ScaledBorderAndShadow: yes
                                    YCbCr Matrix: TV.601

                                    [V4+ Styles]
                                    Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
                                    Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,30,30,30,1

                                    [Events]
                                    Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
                                    """))
            
            if not word_by_word:
                # Group into phrases for normal timing
                phrases = self.group_words_into_phrases(timestamps)
                for phrase in phrases:
                    start_time = self.seconds_to_ass_format(phrase['start'])
                    end_time = self.seconds_to_ass_format(phrase['end'])
                    text = phrase['text']
                    f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n")
                return

                
            # Word-by-word timing with karaoke effect
            current_phrase = []
            current_phrase_start = None
            
            for i, word in enumerate(timestamps):
                word_text = word.get('word', '').strip()
                if not word_text:
                    continue
                    
                if current_phrase_start is None:
                    current_phrase_start = word['start']
                
                # Add word with karaoke effect
                start_time = self.seconds_to_ass_format(word['start'])
                end_time = self.seconds_to_ass_format(word['end'])
                duration_ms = int((word['end'] - word['start']) * 100)
                karaoke_effect = f'{{\\kf{duration_ms}}}{word_text} '
                
                # Write the word with karaoke effect
                f.write(f'Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{karaoke_effect}\\N')
                
                # Add to current phrase for full phrase display
                current_phrase.append(word_text)
                
                # Check if we should start a new phrase
                if i < len(timestamps) - 1:
                    next_word = timestamps[i + 1]
                    gap = next_word['start'] - word['end']
                    
                    # New phrase on significant gap or sentence end
                    if gap > 0.5 or word_text[-1] in {'.', '!', '?', '...'}:
                        if current_phrase:
                            # Write the full phrase
                            phrase_text = ' '.join(current_phrase)
                            phrase_start = self.seconds_to_ass_format(current_phrase_start)
                            phrase_end = self.seconds_to_ass_format(word['end'])
                            
                            # Escape ASS special characters in the phrase text
                            phrase_text = phrase_text.replace("{", "\\{").replace("}", "\\}")
                            
                            f.write(f"Dialogue: 0,{phrase_start},{phrase_end},Default,,0,0,0,,{phrase_text}\\n")
                            # Log first few phrases for debugging
                            if i < 3:
                                logger.info(f"Added phrase: {phrase_start} --> {phrase_end}: {phrase_text[:50]}...")
                                
                            current_phrase = []
                            current_phrase_start = None
                        
            logger.info(f"Successfully generated ASS file at {output_path}")


    def _run_aeneas_cli(self, audio_path: str, text_path: str, output_path: str) -> bool:
        """Run aeneas as a command line tool for better error diagnostics."""
        try:
            import subprocess
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Build the command
            cmd = [
                'python', '-m', 'aeneas.tools.execute_task',
                audio_path,
                text_path,
                'task_language=eng|is_text_type=plain|os_task_file_format=json|os_task_file_level=3',
                output_path
            ]
            
            logger.info(f"Running aeneas CLI: {' '.join(cmd)}")
            
            # Run the command with output capture
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log the output
            if result.stdout:
                logger.info(f"aeneas stdout:\n{result.stdout}")
            if result.stderr:
                logger.error(f"aeneas stderr:\n{result.stderr}")
                
            return os.path.exists(output_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"aeneas CLI failed with return code {e.returncode}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error running aeneas CLI: {str(e)}")
            return False

    def __del__(self):
        """Clean up temporary files."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {e}")

    # ... (rest of the code remains the same)
