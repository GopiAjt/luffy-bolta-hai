import { generateScript } from './api/scriptApi.js';
import { uploadAudio, getAudioMetadata } from './api/audioApi.js';
import { generateSubtitles, getLatestSubtitleFile } from './api/subtitleApi.js';
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
        console.log('Initializing LuffyBoltHaiApp...');
        try {
            // Initialize UI components
            console.log('Initializing UI components...');
            this.scriptLoading = new LoadingIndicator('loading');
            this.subtitleLoading = new LoadingIndicator('subtitleLoading');
            this.videoLoading = new LoadingIndicator('videoLoading');
            this.audioPlayer = new AudioPlayer('previewAudio', 'audioName', 'audioDuration');
            this.videoPlayer = new VideoPlayer('videoPreview', 'videoOutput');
            
            // State
            this.currentAudioId = null;
            this.currentSlideshowVideoId = null;
            
            // Bind event handlers
            console.log('Setting up event listeners...');
            this.initializeEventListeners();
            console.log('LuffyBoltHaiApp initialized successfully');
        } catch (error) {
            console.error('Error initializing LuffyBoltHaiApp:', error);
            throw error;
        }
    }
    
    initializeEventListeners() {
        console.log('Initializing event listeners...');
        // Script generation
        const generateButton = document.getElementById('generateButton');
        console.log('Generate button element:', generateButton);
        if (generateButton) {
            // Remove any existing event listeners to prevent duplicates
            const newGenerateButton = generateButton.cloneNode(true);
            generateButton.parentNode.replaceChild(newGenerateButton, generateButton);
            
            console.log('Adding click event listener to generate button');
            newGenerateButton.addEventListener('click', (event) => {
                console.log('Generate button clicked', event);
                this.handleGenerateScript().catch(error => {
                    console.error('Error in handleGenerateScript:', error);
                });
            });
            
            // Store a flag to prevent duplicate initialization
            newGenerateButton.setAttribute('data-event-listener-attached', 'true');
        }
        
        // Audio upload
        const audioFileInput = document.getElementById('audioFile');
        if (audioFileInput) {
            audioFileInput.addEventListener('change', (e) => this.handleAudioUpload(e));
        }
        
        // Subtitle generation
        const generateSubtitlesButton = document.getElementById('generateSubtitlesButton');
        if (generateSubtitlesButton) {
            generateSubtitlesButton.addEventListener('click', () => this.handleGenerateSubtitles());
        }
        
        // Video generation
        const generateVideoButton = document.getElementById('generateVideoButton');
        if (generateVideoButton) {
            generateVideoButton.addEventListener('click', () => this.handleGenerateVideo());
        }
        
        // Image slides generation
        const generateImageSlidesButton = document.getElementById('generateImageSlidesButton');
        if (generateImageSlidesButton) {
            generateImageSlidesButton.addEventListener('click', () => this.handleGenerateImageSlides());
        }
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
            
            scriptOutput.textContent = data.output.script;
            
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
        const file = event.target.files[0];
        if (!file) return;
        
        const uploadProgress = document.querySelector('.progress');
        const uploadStatus = document.getElementById('uploadStatus');
        
        // Show upload progress
        uploadProgress.style.display = 'block';
        uploadStatus.textContent = 'Uploading...';
        
        try {
            // Upload the file
            const uploadResult = await uploadAudio(file, (percent) => {
                uploadStatus.textContent = `Uploading... ${percent}%`;
                if (percent === 100) {
                    uploadStatus.textContent = 'Processing audio...';
                }
            });
            
            // Update UI with audio metadata
            this.currentAudioId = uploadResult.audio_id;
            this.audioPlayer.setName(file.name);
            this.audioPlayer.setAudioSource(`/api/v1/audio/${this.currentAudioId}`);
            
            // Show audio player
            const audioPreview = document.getElementById('audioPreview');
            if (audioPreview) {
                audioPreview.style.display = 'block';
            }
            
            // Enable subtitle generation if script is available
            const scriptInput = document.getElementById('scriptInput');
            const generateSubtitlesButton = document.getElementById('generateSubtitlesButton');
            if (scriptInput && scriptInput.value && generateSubtitlesButton) {
                generateSubtitlesButton.disabled = false;
            }
            
            uploadStatus.textContent = 'Upload complete!';
        } finally {
            setTimeout(() => {
                uploadProgress.style.display = 'none';
            }, 2000);
        }
    }, {
        errorElement: document.getElementById('uploadStatus')
    });
    
    handleGenerateSubtitles = withErrorHandling(async () => {
        if (!this.currentAudioId) {
            throw new Error('Please upload an audio file first');
        }
        
        const scriptInput = document.getElementById('scriptInput');
        if (!scriptInput || !scriptInput.value) {
            throw new Error('Please generate or enter a script first');
        }
        
        this.subtitleLoading.show();
        const subtitleOutput = document.getElementById('subtitleOutput');
        const generateSubtitlesButton = document.getElementById('generateSubtitlesButton');
        
        if (generateSubtitlesButton) {
            generateSubtitlesButton.disabled = true;
        }
        
        try {
            const result = await generateSubtitles(this.currentAudioId, scriptInput.value);
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
        if (!this.currentAudioId) {
            throw new Error('Please upload an audio file first');
        }
        
        // Get the latest subtitle file
        const subtitleFile = await getLatestSubtitleFile();
        if (!subtitleFile || !subtitleFile.path) {
            throw new Error('No subtitle file available. Please generate subtitles first.');
        }
        
        if (!this.currentSlideshowVideoId) {
            throw new Error('No slideshow video available. Please generate slides first.');
        }
        
        const videoOutput = document.getElementById('videoOutput');
        const generateVideoButton = document.getElementById('generateVideoButton');
        
        this.videoLoading.show();
        videoOutput.style.display = 'none';
        videoOutput.textContent = 'Preparing video generation...';
        
        if (generateVideoButton) {
            generateVideoButton.disabled = true;
        }
        
        try {
            const result = await generateVideo(
                this.currentAudioId,
                subtitleFile.path,
                this.currentSlideshowVideoId
            );
            
            // Update UI with video result
            this.videoPlayer.setSource(result.video_url);
            this.videoPlayer.show();
            this.videoPlayer.createDownloadLink(result.video_url, 'video.mp4');
            
            return result;
        } finally {
            this.videoLoading.hide();
            if (generateVideoButton) {
                generateVideoButton.disabled = false;
            }
        }
    }, {
        errorElement: document.getElementById('videoOutput')
    });
    
    handleGenerateImageSlides = withErrorHandling(async () => {
        if (!this.currentAudioId) {
            throw new Error('Please upload an audio file first');
        }
        
        const imageSlidesStatus = document.getElementById('imageSlidesStatus');
        const imageSlidesOutput = document.getElementById('imageSlidesOutput');
        const generateImageSlidesButton = document.getElementById('generateImageSlidesButton');
        
        imageSlidesStatus.textContent = 'Generating slides...';
        imageSlidesOutput.style.display = 'none';
        
        if (generateImageSlidesButton) {
            generateImageSlidesButton.disabled = true;
        }
        
        try {
            const result = await generateSlideshow(this.currentAudioId);
            this.currentSlideshowVideoId = result.slideshow_video_id;
            
            imageSlidesStatus.textContent = 'Slides generated successfully!';
            imageSlidesOutput.style.display = 'block';
            
            // Enable video generation if not already enabled
            const generateVideoButton = document.getElementById('generateVideoButton');
            if (generateVideoButton) {
                generateVideoButton.disabled = false;
            }
            
            return result;
        } finally {
            if (generateImageSlidesButton) {
                generateImageSlidesButton.disabled = false;
            }
        }
    }, {
        errorElement: document.getElementById('imageSlidesStatus')
    });
}

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new LuffyBoltHaiApp();
});
