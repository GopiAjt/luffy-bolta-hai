import { generateScript } from './api/scriptApi.js';
import { uploadAudio, generateVoiceover, getAudioMetadata } from './api/audioApi.js';
import { generateSubtitles, getLatestSubtitleFile, getLatestExpressionsFile } from './api/subtitleApi.js';
import { generateVideo, generateSlideshow, getOutputUsage, cleanupOutput } from './api/videoApi.js';
import { LoadingIndicator } from './ui/loadingIndicator.js';
import { AudioPlayer } from './ui/audioPlayer.js';
import { VideoPlayer } from './ui/videoPlayer.js';
import { withErrorHandling, showError } from './utils/errorHandler.js';

/**
 * Main application class that ties all components together
 */
export class LuffyBoltHaiApp {
    constructor() {
        console.log('=== LuffyBoltHaiApp constructor called ===');
        console.trace('Stack trace for constructor call');

        if (window.__LuffyBoltHaiAppInstance) {
            console.warn('LuffyBoltHaiApp is already initialized!', window.__LuffyBoltHaiAppInstance);
            return window.__LuffyBoltHaiAppInstance;
        }

        try {
            console.log('Initializing LuffyBoltHaiApp...');
            // Initialize UI components
            console.log('Initializing UI components...');
            this.scriptLoading = new LoadingIndicator('loading');
            this.subtitleLoading = new LoadingIndicator('subtitleLoading');
            this.videoLoading = new LoadingIndicator('videoLoading');
            this.audioPlayer = new AudioPlayer('previewAudio', 'audioName', 'audioDuration');
            this.videoPlayer = new VideoPlayer('videoPreview', 'videoOutput');

            // State
            this.currentAudioId = null;

            // Bind event handlers
            console.log('Setting up event listeners...');
            this.initializeEventListeners();
            console.log('LuffyBoltHaiApp initialized successfully');

            // Store instance to prevent multiple initializations
            window.__LuffyBoltHaiAppInstance = this;
        } catch (error) {
            console.error('Error initializing LuffyBoltHaiApp:', error);
            throw error;
        }
    }

    initializeEventListeners() {
        console.log('=== initializeEventListeners called ===');

        // Script generation
        const generateButton = document.getElementById('generateButton');
        console.log('Generate button element:', generateButton);
        if (generateButton) {
            // Remove any existing event listeners to prevent duplicates
            if (generateButton.getAttribute('data-event-listener-attached') === 'true') {
                console.log('Generate button already has event listener, removing old one');
                const newGenerateButton = generateButton.cloneNode(true);
                generateButton.parentNode.replaceChild(newGenerateButton, generateButton);
                newGenerateButton.setAttribute('data-event-listener-attached', 'true');

                console.log('Adding new click event listener to generate button');
                newGenerateButton.addEventListener('click', (event) => {
                    console.log('Generate button clicked', event);
                    this.handleGenerateScript().catch(error => {
                        console.error('Error in handleGenerateScript:', error);
                    });
                });
            } else {
                console.log('Adding click event listener to generate button');
                generateButton.addEventListener('click', (event) => {
                    console.log('Generate button clicked', event);
                    this.handleGenerateScript().catch(error => {
                        console.error('Error in handleGenerateScript:', error);
                    });
                });
                generateButton.setAttribute('data-event-listener-attached', 'true');
            }
        }

        // Audio upload
        const audioFileInput = document.getElementById('audioFile');
        if (audioFileInput) {
            console.log('Found audio file input');
            if (audioFileInput.getAttribute('data-event-listener-attached') !== 'true') {
                console.log('Adding change event listener to audio file input');
                audioFileInput.addEventListener('change', (e) => {
                    console.log('Audio file selected');
                    this.handleAudioUpload(e);
                });
                audioFileInput.setAttribute('data-event-listener-attached', 'true');
            } else {
                console.log('Event listener already attached to audio file input');
            }
        } else {
            console.error('Audio file input not found in the DOM');
        }

        const generateVoiceoverButton = document.getElementById('generateVoiceoverButton');
        if (generateVoiceoverButton) {
            console.log('Found generate voiceover button');
            if (generateVoiceoverButton.getAttribute('data-event-listener-attached') !== 'true') {
                generateVoiceoverButton.addEventListener('click', () => {
                    console.log('Generate voiceover button clicked');
                    this.handleGenerateVoiceover().catch(error => {
                        console.error('Error in handleGenerateVoiceover:', error);
                    });
                });
                generateVoiceoverButton.setAttribute('data-event-listener-attached', 'true');
            }
        } else {
            console.warn('Generate voiceover button not found');
        }

        // Subtitle generation
        const generateSubtitlesButton = document.getElementById('generateSubtitlesButton');
        if (generateSubtitlesButton) {
            console.log('Found generate subtitles button');
            generateSubtitlesButton.addEventListener('click', () => {
                console.log('Generate subtitles button clicked');
                this.handleGenerateSubtitles();
            });
        } else {
            console.warn('Generate subtitles button not found');
        }

        // Final Video Generation Section
        const generateFinalVideoButton = document.getElementById('generateFinalVideoButton');
        if (generateFinalVideoButton) {
            console.log('Found generate final video button');
            // Disable by default, will be enabled when slides are generated
            generateFinalVideoButton.disabled = true;

            if (generateFinalVideoButton.getAttribute('data-event-listener-attached') !== 'true') {
                console.log('Adding click event listener to generate final video button');
                // Handle generate video button click
                const handleGenerateVideoClick = () => {
                    console.log('=== Generate Final Video Button Clicked ===');
                    console.log('Current audio ID:', this.currentAudioId);

                    this.handleGenerateVideo().catch(error => {
                        console.error('Error in handleGenerateVideo:', error);
                    });
                };
                generateFinalVideoButton.addEventListener('click', handleGenerateVideoClick);
                generateFinalVideoButton.setAttribute('data-event-listener-attached', 'true');
            } else {
                console.log('Event listener already attached to generate final video button');
            }
        } else {
            console.error('Generate final video button not found in the DOM');
            console.log('Available buttons:', {
                generateButton: !!document.getElementById('generateButton'),
                generateSubtitlesButton: !!document.getElementById('generateSubtitlesButton'),
                generateImageSlidesButton: !!document.getElementById('generateImageSlidesButton'),
                generateFinalVideoButton: !!document.getElementById('generateFinalVideoButton')
            });
        }

        // Image slides generation
        const generateImageSlidesButton = document.getElementById('generateImageSlidesButton');
        if (generateImageSlidesButton) {
            console.log('Found generate image slides button');
            if (generateImageSlidesButton.getAttribute('data-event-listener-attached') !== 'true') {
                console.log('Adding click event listener to generate image slides button');
                generateImageSlidesButton.addEventListener('click', () => {
                    console.log('=== Generate Image Slides Button Clicked ===');
                    console.log('Current audio ID:', this.currentAudioId);
                    this.handleGenerateImageSlides().catch(error => {
                        console.error('Error in handleGenerateImageSlides:', error);
                    });
                });
                generateImageSlidesButton.setAttribute('data-event-listener-attached', 'true');
            } else {
                console.log('Event listener already attached to generate image slides button');
            }
        } else {
            console.warn('Generate image slides button not found');
        }

        // Slideshow generation from existing images
        const generateSlideshowButton = document.getElementById('generateSlideshowButton');
        if (generateSlideshowButton) {
            console.log('Found generate slideshow button');
            if (generateSlideshowButton.getAttribute('data-event-listener-attached') !== 'true') {
                console.log('Adding click event listener to generate slideshow button');
                generateSlideshowButton.addEventListener('click', () => {
                    console.log('=== Generate Slideshow from Latest Image Slides Button Clicked ===');
                    console.log('Current audio ID:', this.currentAudioId);
                    if (!this.currentAudioId) {
                        console.error('No audio file uploaded');
                        alert('Please upload an audio file first');
                        return;
                    }
                    this.handleGenerateImageSlides().catch(error => {
                        console.error('Error in handleGenerateImageSlides:', error);
                    });
                });
                generateSlideshowButton.setAttribute('data-event-listener-attached', 'true');
            } else {
                console.log('Event listener already attached to generate slideshow button');
            }
        } else {
            console.warn('Generate slideshow button not found');
        }

        const cleanupOutputButton = document.getElementById('cleanupOutputButton');
        if (cleanupOutputButton) {
            cleanupOutputButton.addEventListener('click', () => {
                this.handleCleanupOutput().catch(error => {
                    console.error('Error in handleCleanupOutput:', error);
                });
            });
        }

        this.refreshOutputUsage().catch(error => {
            console.warn('Could not load output usage:', error);
        });

        console.log('=== Finished initializeEventListeners ===');
    }

    // Event handlers with error handling
    handleGenerateScript = withErrorHandling(async () => {
        console.log('handleGenerateScript called');
        const scriptOutput = document.getElementById('scriptOutput');
        const generateButton = document.getElementById('generateButton');

        if (!scriptOutput) {
            throw new Error('scriptOutput element not found');
        }
        if (!generateButton) {
            throw new Error('generateButton element not found');
        }

        console.log('Showing loading state');
        this.scriptLoading.show();
        generateButton.disabled = true;
        scriptOutput.textContent = 'Generating script...';
        scriptOutput.style.display = 'block';

        try {
            console.log('Calling generateScript API');
            const data = await generateScript();
            console.log('Received response from generateScript:', data);

            if (!data || !data.output || !data.output.script) {
                throw new Error('Invalid response format from server');
            }

            // Update title if available
            const titleElement = document.getElementById('titleText');
            const titleContainer = document.getElementById('scriptTitle');
            if (data.output.title && titleElement && titleContainer) {
                titleElement.textContent = data.output.title;
                titleContainer.style.display = 'block';

                // Also update the page title for better UX
                document.title = data.output.title + ' | One Piece Script Generator';
            }

            // Update script output
            scriptOutput.textContent = data.output.script;

            // Update description if available
            const descriptionElement = document.getElementById('descriptionText');
            const descriptionContainer = document.getElementById('scriptDescription');
            if (data.output.description && descriptionElement && descriptionContainer) {
                // Convert newlines to <br> tags for better formatting
                const formattedDescription = data.output.description.replace(/\n/g, '<br>');
                descriptionElement.innerHTML = formattedDescription;
                descriptionContainer.style.display = 'block';
            }

            // Update hashtags if available
            const hashtagsContainer = document.getElementById('hashtagsContainer');
            const hashtagsElement = document.getElementById('scriptHashtags');
            if (data.output.hashtags && hashtagsContainer && hashtagsElement) {
                // Clear existing hashtags
                hashtagsContainer.innerHTML = '';

                // Split hashtags by space or comma and create badges for each
                const hashtags = data.output.hashtags.split(/[\s,]+/).filter(tag => tag.trim() !== '');
                hashtags.forEach(tag => {
                    if (!tag.startsWith('#')) {
                        tag = '#' + tag;
                    }
                    const badge = document.createElement('span');
                    badge.className = 'badge bg-secondary me-1 mb-1';
                    badge.textContent = tag;
                    hashtagsContainer.appendChild(badge);
                });

                hashtagsElement.style.display = 'block';
            }

            // Copy script to subtitle input
            const scriptInput = document.getElementById('scriptInput');
            if (scriptInput) {
                scriptInput.value = data.output.script;
                const subtitleControls = document.getElementById('subtitleControls');
                if (subtitleControls) {
                    subtitleControls.style.display = 'block';
                }
            }
        } catch (error) {
            console.error('Error in handleGenerateScript:', error);
            scriptOutput.textContent = `Error: ${error.message}`;
            throw error;
        } finally {
            console.log('Hiding loading state');
            this.scriptLoading.hide();
            generateButton.disabled = false;
        }
    }, {
        errorElement: document.getElementById('scriptOutput')
    });

    handleAudioUpload = withErrorHandling(async (event) => {
        console.log('=== Audio Upload Started ===');
        const file = event.target.files[0];
        if (!file) {
            console.log('No file selected');
            return;
        }

        console.log('Selected file:', file.name, 'size:', file.size, 'type:', file.type);

        const uploadProgress = document.querySelector('.progress');
        const uploadStatus = document.getElementById('uploadStatus');

        // Show upload progress
        uploadProgress.style.display = 'block';
        uploadStatus.textContent = 'Uploading...';

        try {
            console.log('Starting file upload...');
            // Upload the file
            const uploadResult = await uploadAudio(file, (percent) => {
                const status = `Uploading... ${percent}%`;
                console.log(status);
                uploadStatus.textContent = status;
                if (percent === 100) {
                    console.log('Upload complete, processing audio...');
                    uploadStatus.textContent = 'Processing audio...';
                }
            });

            console.log('Upload result:', uploadResult);

            if (!uploadResult || !uploadResult.id) {
                console.error('Invalid upload response:', uploadResult);
                throw new Error('Invalid upload response: missing file ID');
            }

            // Update UI with audio metadata
            this.currentAudioId = uploadResult.id; // Changed from uploadResult.audio_id to uploadResult.id
            console.log('Audio uploaded successfully. Current audio ID:', this.currentAudioId);

            this.audioPlayer.setName(file.name);
            this.audioPlayer.setAudioSource(`/api/v1/audio/${this.currentAudioId}`);

            // Show audio player
            const audioPreview = document.getElementById('audioPreview');
            if (audioPreview) {
                audioPreview.style.display = 'block';
                console.log('Audio preview element shown');
            } else {
                console.warn('Audio preview element not found');
            }

            // Enable subtitle generation if script is available
            const scriptInput = document.getElementById('scriptInput');
            const generateSubtitlesButton = document.getElementById('generateSubtitlesButton');

            console.log('Script input value:', scriptInput ? scriptInput.value : 'Script input not found');
            console.log('Generate subtitles button:', generateSubtitlesButton);

            if (scriptInput && scriptInput.value && generateSubtitlesButton) {
                console.log('Enabling generate subtitles button');
                generateSubtitlesButton.disabled = false;
            } else {
                console.log('Subtitles button not enabled. Missing:', {
                    hasScriptInput: !!scriptInput,
                    hasScriptValue: scriptInput && !!scriptInput.value,
                    hasButton: !!generateSubtitlesButton
                });
            }

            uploadStatus.textContent = 'Upload complete!';
            console.log('=== Audio Upload Completed Successfully ===');
        } catch (error) {
            console.error('Error in handleAudioUpload:', error);
            throw error;
        } finally {
            setTimeout(() => {
                uploadProgress.style.display = 'none';
            }, 2000);
        }
    }, {
        errorElement: document.getElementById('uploadStatus')
    });

    handleGenerateVoiceover = withErrorHandling(async () => {
        console.log('=== Generate Voiceover Started ===');
        const scriptInput = document.getElementById('scriptInput');
        const voiceLanguage = document.getElementById('voiceLanguage');
        const voiceInstruct = document.getElementById('voiceInstruct');
        const voiceoverButton = document.getElementById('generateVoiceoverButton');
        const uploadStatus = document.getElementById('uploadStatus');
        const uploadProgress = document.getElementById('uploadProgress');

        const scriptText = scriptInput ? scriptInput.value.trim() : '';
        if (!scriptText) {
            throw new Error('Please generate or paste a script first');
        }

        if (voiceoverButton) {
            voiceoverButton.disabled = true;
            voiceoverButton.textContent = 'Generating voiceover...';
        }
        if (uploadStatus) {
            uploadStatus.textContent = 'Generating voiceover with Qwen3-TTS. First run can take several minutes on CPU...';
        }
        if (uploadProgress) {
            uploadProgress.style.display = 'block';
        }

        const loadingMessages = [
            'Loading Qwen3-TTS on CPU...',
            'Designing the voice...',
            'Generating speech audio...',
            'Still working. CPU generation can take a while...',
            'Almost there. Saving the voiceover soon...'
        ];
        let loadingMessageIndex = 0;
        const loadingStartedAt = Date.now();
        const voiceoverStatusTimer = uploadStatus ? setInterval(() => {
            loadingMessageIndex = (loadingMessageIndex + 1) % loadingMessages.length;
            const elapsedSeconds = Math.round((Date.now() - loadingStartedAt) / 1000);
            uploadStatus.textContent = `${loadingMessages[loadingMessageIndex]} (${elapsedSeconds}s elapsed)`;
        }, 8000) : null;

        try {
            const result = await generateVoiceover(
                scriptText,
                voiceLanguage ? voiceLanguage.value : 'English',
                voiceInstruct ? voiceInstruct.value : ''
            );

            if (!result || !result.id) {
                throw new Error('Invalid voiceover response: missing file ID');
            }

            this.currentAudioId = result.id;
            this.audioPlayer.setName(`Generated voiceover (${result.id})`);
            this.audioPlayer.setAudioSource(`/api/v1/audio/${this.currentAudioId}`);
            this.audioPlayer.setDuration(result.duration);

            const audioPreview = document.getElementById('audioPreview');
            if (audioPreview) {
                audioPreview.style.display = 'block';
            }

            const generateSubtitlesButton = document.getElementById('generateSubtitlesButton');
            if (generateSubtitlesButton) {
                generateSubtitlesButton.disabled = false;
            }

            if (uploadStatus) {
                uploadStatus.textContent = 'Voiceover generated successfully!';
            }

            console.log('=== Generate Voiceover Completed Successfully ===');
            return result;
        } finally {
            if (voiceoverStatusTimer) {
                clearInterval(voiceoverStatusTimer);
            }
            if (voiceoverButton) {
                voiceoverButton.disabled = false;
                voiceoverButton.innerHTML = '<i class="bi bi-soundwave me-2"></i>Generate Voiceover';
            }
            if (uploadProgress) {
                setTimeout(() => {
                    uploadProgress.style.display = 'none';
                }, 2000);
            }
        }
    }, {
        errorElement: document.getElementById('uploadStatus')
    });

    handleGenerateSubtitles = withErrorHandling(async () => {
        console.log('=== Generate Subtitles Started ===');
        console.log('Current audio ID:', this.currentAudioId);

        if (!this.currentAudioId) {
            console.error('No audio file uploaded. Current audio ID is null/undefined.');
            throw new Error('Please upload an audio file first');
        }

        // Make script input optional
        const scriptInput = document.getElementById('scriptInput');
        const scriptText = scriptInput ? scriptInput.value : null;

        console.log('Script input:', scriptText ? 'Provided' : 'Not provided, will transcribe from audio');

        this.subtitleLoading.show();
        const subtitleOutput = document.getElementById('subtitleOutput');
        const generateSubtitlesButton = document.getElementById('generateSubtitlesButton');

        if (generateSubtitlesButton) {
            console.log('Disabling generate subtitles button');
            generateSubtitlesButton.disabled = true;
        }

        try {
            // Get selected subtitle style
            const subtitleStyleSelector = document.getElementById('subtitleStyle');
            let subtitleStyle = 'epic'; // Default to 'epic' (Epic Battles)
            if (subtitleStyleSelector && subtitleStyleSelector.value) {
                subtitleStyle = subtitleStyleSelector.value;
                console.log('Selected subtitle style:', subtitleStyle);
            } else {
                console.warn('Subtitle style selector not found, using default style:', subtitleStyle);
            }

            const result = await generateSubtitles(this.currentAudioId, scriptText, subtitleStyle);
            subtitleOutput.textContent = 'Subtitles generated successfully!';
            subtitleOutput.style.display = 'block';

            // Enable video generation
            const generateVideoButton = document.getElementById('generateVideoButton');
            if (generateVideoButton) {
                generateVideoButton.disabled = false;
            }

            return result;
        } finally {
            this.subtitleLoading.hide();
            if (generateSubtitlesButton) {
                generateSubtitlesButton.disabled = false;
            }
        }
    }, {
        errorElement: document.getElementById('subtitleOutput')
    });

    handleGenerateVideo = withErrorHandling(async () => {
        console.log('=== handleGenerateVideo started ===');

        if (!this.currentAudioId) {
            const errorMsg = 'Please upload an audio file first';
            console.error(errorMsg);
            throw new Error(errorMsg);
        }

        console.log('Getting latest subtitle file...');
        // Get the latest subtitle file
        const subtitleFile = await getLatestSubtitleFile();
        console.log('Latest subtitle file:', subtitleFile);

        if (!subtitleFile || !subtitleFile.path) {
            const errorMsg = 'No subtitle file available. Please generate subtitles first.';
            console.error(errorMsg);
            throw new Error(errorMsg);
        }

        const videoOutput = document.getElementById('videoOutput');
        const generateVideoButton = document.getElementById('generateFinalVideoButton');

        console.log('Showing loading state...');
        this.videoLoading.show();
        videoOutput.style.display = 'none';
        videoOutput.textContent = 'Generating final video... This may take a few minutes.';

        if (generateVideoButton) {
            generateVideoButton.disabled = true;
            generateVideoButton.textContent = 'Generating...';
        }

        try {
            console.log('Starting final video generation with audio:', this.currentAudioId);

            // Get the latest expressions file
            const expressionsFile = await getLatestExpressionsFile();
            console.log('Latest expressions file:', expressionsFile);

            // Get the latest image slides JSON if it exists
            const imageSlidesResponse = await fetch('/api/v1/latest-image-slides');
            let slidesJson = null;
            if (imageSlidesResponse.ok) {
                const data = await imageSlidesResponse.json();
                if (data && data.path) {
                    slidesJson = data.path;
                    console.log('Found existing image slides:', slidesJson);
                }
            }

            const requestData = {
                audio_id: this.currentAudioId,
                subtitle_file: subtitleFile.path,
                expressions_file: expressionsFile && expressionsFile.path ? expressionsFile.path : null,
                force_regenerate_slides: false,  // Don't force regenerate slides
                slides_json: slidesJson,         // Pass the existing slides JSON if available
                use_existing_slides: true        // Indicate we want to use existing slides
            };

            console.log('Request data for final video generation:', JSON.stringify(requestData, null, 2));

            if (!requestData.expressions_file) {
                console.warn('No expressions file found. Video will be generated without expressions.');
            }

            console.log('Sending request to /api/v1/generate-final-video with data:', JSON.stringify(requestData, null, 2));

            // Call the final video generation endpoint
            const response = await fetch('/api/v1/generate-final-video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            const responseText = await response.text();
            console.log('Raw response from server:', responseText);

            if (!response.ok) {
                let errorData;
                try {
                    errorData = JSON.parse(responseText);
                } catch (e) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}\n${responseText}`);
                }
                throw new Error(errorData.message || 'Failed to generate final video');
            }

            let result;
            try {
                result = JSON.parse(responseText);
                console.log('Parsed response:', result);
            } catch (e) {
                console.error('Failed to parse response as JSON:', e);
                throw new Error('Invalid response format from server');
            }

            if (!result || !result.video_file) {
                console.error('Invalid response structure:', result);
                throw new Error('Invalid response from video generation service');
            }

            console.log('Final video generation successful, updating UI...');

            // Update UI with video result
            const videoUrl = `/api/v1/download/${result.video_file}`;
            console.log('Video URL:', videoUrl);

            // Ensure the video player container is visible
            const videoContainer = document.getElementById('videoPreviewContainer');
            if (videoContainer) {
                videoContainer.style.display = 'block';
            }

            // Set the video source and handle loading
            this.videoPlayer.setSource(videoUrl);
            this.videoPlayer.show();

            // Create download link
            this.videoPlayer.createDownloadLink(videoUrl, result.video_file);

            // Auto-play the video if user interaction has occurred
            try {
                const videoElement = document.getElementById('videoPreview');
                if (videoElement) {
                    const playPromise = videoElement.play();
                    if (playPromise !== undefined) {
                        playPromise.catch(error => {
                            console.log('Auto-play was prevented:', error);
                            // Show play button or other UI to indicate video is ready
                        });
                    }
                }
            } catch (e) {
                console.error('Error playing video:', e);
            }

            console.log('=== handleGenerateVideo completed successfully ===');
            return result;
        } catch (error) {
            console.error('Error in handleGenerateVideo:', error);
            throw error;
        } finally {
            console.log('Cleaning up...');
            this.videoLoading.hide();
            if (generateVideoButton) {
                generateVideoButton.disabled = false;
                generateVideoButton.textContent = 'Generate Final Video';
            }
        }
    }, {
        errorElement: document.getElementById('videoOutput')
    });

    handleGenerateImageSlides = withErrorHandling(async () => {
        console.log('=== handleGenerateImageSlides started ===');

        if (!this.currentAudioId) {
            const errorMsg = 'Please upload an audio file first';
            console.error(errorMsg);
            throw new Error(errorMsg);
        }

        const generateSlidesButton = document.getElementById('generateSlidesButton');
        const slidesStatus = document.getElementById('slidesStatus');

        console.log('Showing loading state for image slides generation...');
        this.videoLoading.show();
        if (slidesStatus) {
            slidesStatus.textContent = 'Generating image slides...';
            slidesStatus.style.display = 'block';
        }

        if (generateSlidesButton) {
            generateSlidesButton.disabled = true;
            generateSlidesButton.textContent = 'Generating...';
        }

        try {
            console.log('Starting image slides generation with audio:', this.currentAudioId);

            // Call the API to generate image slides
            const response = await fetch('/api/v1/generate-image-slides', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    audio_id: this.currentAudioId,
                    ass_path: 'auto'  // Let server find the latest ASS file
                })
            });

            const responseText = await response.text();
            console.log('Raw response from server:', responseText);

            if (!response.ok) {
                let errorData;
                try {
                    errorData = JSON.parse(responseText);
                } catch (e) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}\n${responseText}`);
                }
                throw new Error(errorData.message || 'Failed to generate image slides');
            }

            let result;
            try {
                result = JSON.parse(responseText);
                console.log('Image slides generation result:', result);
            } catch (e) {
                console.error('Failed to parse response as JSON:', e);
                throw new Error('Invalid response format from server');
            }

            // Update UI to show success and enable final video generation
            if (slidesStatus) {
                slidesStatus.textContent = 'Image slides generated successfully!';
            }

            // Enable the final video generation button
            const generateVideoButton = document.getElementById('generateFinalVideoButton');
            if (generateVideoButton) {
                generateVideoButton.disabled = false;
            }

            console.log('=== handleGenerateImageSlides completed successfully ===');
            return result;

        } catch (error) {
            console.error('Error in handleGenerateImageSlides:', error);
            if (slidesStatus) {
                slidesStatus.textContent = `Error: ${error.message}`;
            }
            throw error;
        } finally {
            console.log('Cleaning up image slides generation...');
            this.videoLoading.hide();
            if (generateSlidesButton) {
                generateSlidesButton.disabled = false;
                generateSlidesButton.textContent = 'Generate Image Slides';
            }
        }
    }, {
        errorElement: document.getElementById('slidesStatus')
    });

    refreshOutputUsage = async () => {
        const cleanupStatus = document.getElementById('cleanupStatus');
        if (!cleanupStatus) {
            return;
        }

        const data = await getOutputUsage();
        if (data && data.usage) {
            cleanupStatus.textContent = `${data.usage.file_count} generated files, ${data.usage.total_mb} MB used`;
        }
    };

    handleCleanupOutput = withErrorHandling(async () => {
        const cleanupButton = document.getElementById('cleanupOutputButton');
        const cleanupStatus = document.getElementById('cleanupStatus');
        const forceCheckbox = document.getElementById('cleanupForce');
        const ageInput = document.getElementById('cleanupAgeHours');
        const force = forceCheckbox ? forceCheckbox.checked : false;
        const maxAgeHours = ageInput && ageInput.value ? Number(ageInput.value) : 24;

        if (cleanupButton) {
            cleanupButton.disabled = true;
        }
        if (cleanupStatus) {
            cleanupStatus.textContent = force ? 'Deleting generated output files...' : `Deleting files older than ${maxAgeHours} hours...`;
        }

        try {
            const result = await cleanupOutput({ maxAgeHours, force });
            const cleanupResult = result.result;
            if (cleanupStatus && cleanupResult) {
                cleanupStatus.textContent = `Deleted ${cleanupResult.deleted_count} files (${cleanupResult.mb_deleted} MB). Remaining: ${cleanupResult.remaining.file_count} files, ${cleanupResult.remaining.total_mb} MB.`;
            }
            return result;
        } finally {
            if (cleanupButton) {
                cleanupButton.disabled = false;
            }
        }
    }, {
        errorElement: document.getElementById('cleanupStatus')
    });
}

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new LuffyBoltHaiApp();
});
