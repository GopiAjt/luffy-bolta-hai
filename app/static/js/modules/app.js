import { generateScript } from './api/scriptApi.js';
import { uploadAudio, getAudioMetadata } from './api/audioApi.js';
import { generateSubtitles, getLatestSubtitleFile, getLatestExpressionsFile } from './api/subtitleApi.js';
import { generateVideo, generateSlideshow } from './api/videoApi.js';
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
        
        console.log('=== Finished initializeEventListeners ===');
    }
    
    // Event handlers with error handling
    handleGenerateScript = withErrorHandling(async () => {
        console.log('handleGenerateScript called');
        const scriptContainer = document.getElementById('scriptOutput');
        const generateButton = document.getElementById('generateButton');
        
        if (!scriptContainer) {
            throw new Error('scriptOutput element not found');
        }
        if (!generateButton) {
            throw new Error('generateButton element not found');
        }
        
        console.log('Showing loading state');
        this.scriptLoading.show();
        generateButton.disabled = true;
        scriptContainer.innerHTML = '<div class="text-center">Generating script...</div>';
        scriptContainer.style.display = 'block';
        
        try {
            console.log('Calling generateScript API');
            const scriptData = await generateScript();
            console.log('Received response from generateScript:', scriptData);
            
            if (!scriptData || !scriptData.script) {
                throw new Error('Invalid response format from server');
            }
            
            // Create HTML for the script output
            let html = `
                <div class="script-output">
                    <h3 class="script-title mb-3">${scriptData.title || 'Generated Script'}</h3>
                    
                    <div class="script-description mb-3 p-3 bg-light rounded">
                        ${scriptData.description || ''}
                    </div>
                    
                    <div class="script-content p-3 bg-white border rounded mb-3">
                        ${scriptData.script.replace(/\n/g, '<br>')}
                    </div>
                    
                    ${scriptData.tags && scriptData.tags.length ? `
                    <div class="script-tags mb-3">
                        ${scriptData.tags.map(tag => 
                            `<span class="badge bg-primary me-1">${tag}</span>`
                        ).join('')}
                    </div>` : ''}
                </div>
            `;
            
            scriptContainer.innerHTML = html;
            
            // Copy script to subtitle input
            const scriptInput = document.getElementById('scriptInput');
            if (scriptInput) {
                scriptInput.value = scriptData.script;
                const subtitleControls = document.getElementById('subtitleControls');
                if (subtitleControls) {
                    subtitleControls.style.display = 'block';
                }
            }
        } catch (error) {
            console.error('Error in handleGenerateScript:', error);
            scriptContainer.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    <strong>Error:</strong> ${error.message}
                </div>
            `;
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
    
    handleGenerateSubtitles = withErrorHandling(async () => {
        console.log('=== Generate Subtitles Started ===');
        console.log('Current audio ID:', this.currentAudioId);
        
        if (!this.currentAudioId) {
            console.error('No audio file uploaded. Current audio ID is null/undefined.');
            throw new Error('Please upload an audio file first');
        }
        
        const scriptInput = document.getElementById('scriptInput');
        console.log('Script input element:', scriptInput);
        console.log('Script input value:', scriptInput ? scriptInput.value : 'N/A');
        
        if (!scriptInput || !scriptInput.value) {
            console.error('No script available. Please generate or enter a script first.');
            throw new Error('Please generate or enter a script first');
        }
        
        this.subtitleLoading.show();
        const subtitleOutput = document.getElementById('subtitleOutput');
        const generateSubtitlesButton = document.getElementById('generateSubtitlesButton');
        
        console.log('Generate subtitles button state:', {
            element: generateSubtitlesButton,
            disabled: generateSubtitlesButton ? generateSubtitlesButton.disabled : 'N/A'
        });
        
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
            const result = await generateSubtitles(this.currentAudioId, scriptInput.value, subtitleStyle);
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
}

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new LuffyBoltHaiApp();
});
