import os
import json
import textwrap
from typing import List, Dict
import logging
import tempfile
from pathlib import Path
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
import nltk

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
                logger.info(
                    f"Script content preview: {self.script_text[:100]}...")

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
            task.sync_map_file_path_absolute = os.path.abspath(
                os.path.join(self.temp_dir, "syncmap.json"))

            # Ensure output directory exists and is writable
            os.makedirs(os.path.dirname(
                task.sync_map_file_path_absolute), exist_ok=True)

            # Log task details
            logger.info(f"Task audio path: {task.audio_file_path_absolute}")
            logger.info(f"Task text path: {task.text_file_path_absolute}")
            logger.info(
                f"Task output path: {task.sync_map_file_path_absolute}")
            logger.info(f"Temporary directory: {self.temp_dir}")

            # Verify file permissions
            if not os.access(self.audio_path, os.R_OK):
                logger.error(f"Audio file not readable: {self.audio_path}")
                raise PermissionError(
                    f"Cannot read audio file: {self.audio_path}")

            if not os.access(text_file, os.R_OK):
                logger.error(f"Text file not readable: {text_file}")
                raise PermissionError(f"Cannot read text file: {text_file}")

            # Check if files exist before task execution
            logger.info("Checking file existence before task execution...")
            for file_path in [self.audio_path, text_file]:
                if not os.path.exists(file_path):
                    logger.error(f"File does not exist: {file_path}")
                    raise FileNotFoundError(
                        f"Required file not found: {file_path}")
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
                                logger.info(
                                    f"{param}: {task.configuration[param]}")
                            except KeyError:
                                logger.debug(
                                    f"Configuration parameter not found: {param}")
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
                        logger.info(
                            f"Task audio path: {task.audio_file_path_absolute}")
                        logger.info(
                            f"Task text path: {task.text_file_path_absolute}")
                        logger.info(
                            f"Output path: {task.sync_map_file_path_absolute}")

                        # Verify input files exist and are readable
                        if not os.path.isfile(task.audio_file_path_absolute):
                            raise FileNotFoundError(
                                f"Audio file not found: {task.audio_file_path_absolute}")
                        if not os.path.isfile(task.text_file_path_absolute):
                            raise FileNotFoundError(
                                f"Text file not found: {task.text_file_path_absolute}")

                        # Execute with error handling
                        try:
                            executor.execute()
                        except Exception as e:
                            # Check if output was created despite the exception
                            if os.path.exists(task.sync_map_file_path_absolute):
                                logger.warning(
                                    f"Task raised exception but output file exists: {e}")
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
                        os.path.join(os.path.dirname(
                            task.sync_map_file_path_absolute), 'output.json'),
                        os.path.join(os.path.dirname(
                            task.sync_map_file_path_absolute), 'output.txt')
                    ]

                    # Try the CLI fallback if Python API fails
                    logger.warning("Python API failed, trying CLI fallback...")
                    if not self._run_aeneas_cli(
                        task.audio_file_path_absolute,
                        task.text_file_path_absolute,
                        task.sync_map_file_path_absolute
                    ):
                        logger.warning(
                            "CLI fallback failed, checking for any output files...")

                    # Check which output file was created
                    output_found = False
                    output_path = task.sync_map_file_path_absolute

                    if not os.path.exists(output_path):
                        # Check other possible output locations
                        for path in possible_outputs:
                            if os.path.exists(path):
                                output_path = path
                                logger.info(
                                    f"Found output file at alternative location: {output_path}")
                                output_found = True
                                break
                    else:
                        output_found = True

                    if not output_found:
                        raise RuntimeError(
                            "No output files were generated by aeneas")

                    # Update the sync map file path to the found file
                    task.sync_map_file_path_absolute = output_path
                    logger.info(f"Using sync map file: {output_path}")

                    # Log file details
                    logger.info(
                        f"Output file: {task.sync_map_file_path_absolute}")
                    logger.info(
                        f"File size: {os.path.getsize(task.sync_map_file_path_absolute)} bytes")
                    logger.info(
                        f"File permissions: {oct(os.stat(task.sync_map_file_path_absolute).st_mode)[-3:]}")

                except Exception as e:
                    logger.error(
                        f"Error during aeneas execution: {str(e)}", exc_info=True)

                    # Log directory contents for debugging
                    if os.path.exists(self.temp_dir):
                        logger.error("Temporary directory contents:")
                        for item in os.listdir(self.temp_dir):
                            item_path = os.path.join(self.temp_dir, item)
                            try:
                                size = os.path.getsize(item_path)
                                logger.error(f"  {item} ({size} bytes)")
                                # Only read small text files
                                if item.endswith(('.txt', '.json', '.log')) and size < 1024*1024:
                                    try:
                                        with open(item_path, 'r') as f:
                                            content = f.read()
                                            logger.error(
                                                f"Content of {item}: {content[:500]}")
                                    except:
                                        pass
                            except:
                                logger.error(f"  {item} (error getting size)")

                    # Restore original stdout/stderr before raising
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr

                    raise RuntimeError(
                        f"Failed to generate subtitle timestamps: {str(e)}")

                # Restore original stdout/stderr on success
                sys.stdout = original_stdout
                sys.stderr = original_stderr

                # Check if sync map file exists and validate it
                if not os.path.exists(task.sync_map_file_path_absolute):
                    logger.error(
                        f"Sync map file not found after task execution: {task.sync_map_file_path_absolute}")
                    logger.error("Directory contents:")
                    if os.path.exists(self.temp_dir):
                        for item in os.listdir(self.temp_dir):
                            logger.error(f"  {item}")
                        # Check file permissions
                        if os.path.exists(task.sync_map_file_path_absolute):
                            logger.error(
                                f"Sync map file exists but cannot be accessed: {os.access(task.sync_map_file_path_absolute, os.R_OK)}")
                            logger.error(
                                f"File permissions: {oct(os.stat(task.sync_map_file_path_absolute).st_mode)[-3:]}")

                    # Check if any other files were created
                    logger.error("Checking for other output files:")
                    for file in os.listdir(self.temp_dir):
                        if file != "resampled.wav" and file != "script.txt":
                            logger.error(f"Found unexpected file: {file}")
                            if file.endswith(".json"):
                                try:
                                    with open(os.path.join(self.temp_dir, file), 'r') as f:
                                        content = f.read()
                                        logger.error(
                                            f"Content of {file}: {content[:100]}...")
                                except Exception as e:
                                    logger.error(
                                        f"Could not read {file}: {str(e)}")

                    raise FileNotFoundError(
                        f"Sync map file not found at {task.sync_map_file_path_absolute}")

                # Read and parse the sync map
                with open(task.sync_map_file_path_absolute, "r", encoding="utf-8") as f:
                    content = f.read()
                    logger.info(
                        f"Sync map file content length: {len(content)}")
                    # Log first 500 chars
                    logger.debug(f"Sync map content: {content[:500]}...")

                    try:
                        # Parse the sync map
                        sync_map_content = json.loads(content)
                        logger.info(
                            f"Successfully parsed sync map: {type(sync_map_content)}")

                        # Log basic structure for debugging
                        if isinstance(sync_map_content, dict):
                            logger.info(
                                f"Sync map keys: {list(sync_map_content.keys())}")
                            if 'fragments' in sync_map_content:
                                logger.info(
                                    f"Found {len(sync_map_content['fragments'])} fragments")
                                if sync_map_content['fragments']:
                                    logger.debug(
                                        f"First fragment: {sync_map_content['fragments'][0]}")
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to parse sync map JSON: {str(e)}")
                        raise

                    # Log the sync map structure for debugging
                    logger.info(
                        f"Successfully parsed sync map with {len(sync_map_content.get('fragments', []))} fragments")

                    # Convert to the expected format
                    timestamps = []
                    if 'fragments' in sync_map_content and isinstance(sync_map_content['fragments'], list):
                        logger.info(
                            f"Processing {len(sync_map_content['fragments'])} fragments")
                        for i, fragment in enumerate(sync_map_content['fragments']):
                            try:
                                if not isinstance(fragment, dict):
                                    logger.warning(
                                        f"Skipping invalid fragment at index {i}: {fragment}")
                                    continue

                                # Get the text from the fragment
                                text = ' '.join(fragment.get(
                                    'lines', [''])).strip()
                                if not text or text == '---':
                                    logger.debug(
                                        f"Skipping empty or separator fragment: {text}")
                                    continue

                                # Create timestamp entry
                                timestamp = {
                                    'word': text,  # Store the full text as word for now
                                    'start': float(fragment.get('begin', 0)),
                                    'end': float(fragment.get('end', 0)),
                                    'text': text  # Also store in text for backward compatibility
                                }

                                logger.debug(
                                    f"Added timestamp {i}: {timestamp}")
                                timestamps.append(timestamp)

                            except (ValueError, TypeError) as e:
                                logger.error(
                                    f"Error parsing fragment {i} ({fragment}): {e}")
                                continue

                        logger.info(
                            f"Generated {len(timestamps)} valid timestamps")
                        if timestamps:
                            logger.info(f"First timestamp: {timestamps[0]}")
                        else:
                            logger.warning(
                                "No valid timestamp fragments found in sync map")

                        # Return the timestamps if we have any
                        if timestamps:
                            return timestamps

                        # If we get here, no timestamps were generated
                        logger.error(
                            "No valid timestamps could be generated from the sync map")
                        logger.error(f"Sync map content: {sync_map_content}")
                        return []
                    else:
                        logger.error(
                            f"Unexpected sync map format: {sync_map_content}")
                        raise ValueError(
                            "Invalid sync map format: missing 'fragments' array")

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
                        logger.error(
                            f"Sync map file exists but cannot be accessed: {os.access(task.sync_map_file_path_absolute, os.R_OK)}")
                raise

            # Convert sync map to word timestamps
            timestamps = []
            try:
                if not sync_map_content or 'fragments' not in sync_map_content:
                    logger.error(
                        f"Invalid sync map format: {sync_map_content}")
                    return []

                for fragment in sync_map_content['fragments']:
                    if not isinstance(fragment, dict):
                        logger.warning(
                            f"Skipping invalid fragment: {fragment}")
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
                        logger.error(
                            f"Error processing fragment {fragment}: {e}")
                        continue

                logger.info(f"Generated {len(timestamps)} word timestamps")
                if timestamps:
                    logger.debug(f"First timestamp: {timestamps[0]}")

                return timestamps

            except Exception as e:
                logger.error(
                    f"Error processing sync map: {str(e)}", exc_info=True)
                raise

        except Exception as e:
            logger.error(f"Error in generate_timestamps: {str(e)}")
            logger.error(f"Current directory: {os.getcwd()}")
            logger.error(f"Temporary directory contents:")
            if os.path.exists(self.temp_dir):
                for item in os.listdir(self.temp_dir):
                    logger.error(f"  {item}")
            raise

    def group_words_into_phrases(
        self,
        timestamps: List[Dict],
        min_gap: float = 1.0,
        max_phrase_duration: float = 5.0,
        max_words: int = 5  # Changed from 8 to 5
    ) -> List[Dict]:
        """
        Group words into short, readable phrases by max word count (ignoring sentence boundaries).
        """
        if not timestamps:
            return []

        logger.info(
            f"Grouping {len(timestamps)} words into fixed-size phrases (max {max_words} words)...")

        phrases = []
        idx = 0
        n = len(timestamps)
        while idx < n:
            chunk = timestamps[idx:idx+max_words]
            if not chunk:
                break
            start_time = chunk[0]['start']
            end_time = chunk[-1]['end']
            text = ' '.join([w['word'] for w in chunk])
            phrases.append({
                "start": start_time,
                "end": end_time,
                "text": text
            })
            idx += max_words
        logger.info(f"Created {len(phrases)} fixed-size phrases")
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

    def generate_ass_file(self, phrases: List[Dict], output_path: str) -> None:
        """
        Generate a clean, phrase-based ASS subtitle file (no effects, no karaoke, no formatting).

        Args:
            phrases: List of dicts with 'start', 'end', 'text' for each phrase
            output_path: Path to save the ASS file
        """
        logger.info(f"Generating clean phrase-based ASS file at {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Strictly use the clean template for header
        header = textwrap.dedent("""
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
        """)
        content = [header.strip() + '\n']

        if not phrases:
            logger.warning("No phrases provided for subtitle generation")
            return

        # Write each phrase as a single Dialogue line, with bold formatting
        last_end = -1
        for phrase in phrases:
            start = self.seconds_to_ass_format(phrase['start'])
            end = self.seconds_to_ass_format(phrase['end'])
            # Remove any curly braces, effect tags, or markup
            text = phrase['text']
            text = text.replace('{', '').replace('}', '')
            text = text.replace('\\N', ' ').replace('\\n', ' ')
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = text.strip()
            # Ensure no overlap or duplicate
            if phrase['start'] < last_end:
                logger.warning(f"Skipping overlapping phrase: {text}")
                continue
            # Add bold override tags
            text = "{\\b1}" + text + "{\\b0}"
            content.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")
            last_end = phrase['end']

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
