document.addEventListener('DOMContentLoaded', function() {
    const generateButton = document.getElementById('generateButton');
    const loading = document.getElementById('loading');
    const scriptOutput = document.getElementById('scriptOutput');
    const audioFileInput = document.getElementById('audioFile');
    const uploadProgress = document.querySelector('.progress');
    const uploadStatus = document.getElementById('uploadStatus');
    const audioPreview = document.getElementById('audioPreview');
    const previewAudio = document.getElementById('previewAudio');
    const audioName = document.getElementById('audioName');
    const audioDuration = document.getElementById('audioDuration');
    const subtitleControls = document.getElementById('subtitleControls');
    const scriptInput = document.getElementById('scriptInput');
    const generateSubtitlesButton = document.getElementById('generateSubtitlesButton');
    const subtitleLoading = document.getElementById('subtitleLoading');
    const subtitleOutput = document.getElementById('subtitleOutput');
    let currentAudioId = null;

    // Script Generation
    generateButton.addEventListener('click', async function() {
        loading.style.display = 'block';
        generateButton.disabled = true;

        try {
            const response = await fetch('/api/v1/generate-script', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    script: "Generate a 30-60 second One Piece narration script"
                })
            });

            if (!response.ok) {
                throw new Error('Failed to generate script');
            }

            const data = await response.json();
            scriptOutput.textContent = data.output.script;
            scriptOutput.style.display = 'block';
            
            // Copy script to subtitle input
            scriptInput.value = data.output.script;
            subtitleControls.style.display = 'block';
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to generate script. Please try again.');
        } finally {
            loading.style.display = 'none';
            generateButton.disabled = false;
        }
    });

    // Audio Upload
    audioFileInput.addEventListener('change', async function(e) {
        const file = e.target.files[0];
        if (!file) return;

        // Show file info
        audioName.textContent = file.name;
        
        // Create audio preview
        const audioContext = new AudioContext();
        const arrayBuffer = await file.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const duration = audioBuffer.duration;
        audioDuration.textContent = `Duration: ${Math.floor(duration)}s`;
        
        // Show preview controls
        previewAudio.src = URL.createObjectURL(file);
        previewAudio.style.display = 'block';
        
        // Upload file
        const formData = new FormData();
        formData.append('audio', file);

        uploadStatus.textContent = 'Uploading...';
        uploadProgress.style.width = '0%';

        try {
            const response = await fetch('/api/v1/upload-audio', {
                method: 'POST',
                body: formData,
                onprogress: function(progressEvent) {
                    if (progressEvent.lengthComputable) {
                        const percentComplete = (progressEvent.loaded / progressEvent.total) * 100;
                        uploadProgress.style.width = `${percentComplete}%`;
                    }
                }
            });

            if (!response.ok) {
                throw new Error('Failed to upload audio');
            }

            const data = await response.json();
            currentAudioId = data.id;
            uploadStatus.textContent = 'Upload successful!';
            
            // Enable subtitle generation
            subtitleControls.style.display = 'block';
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to upload audio. Please try again.');
            uploadStatus.textContent = 'Upload failed';
        }
    });

    // Subtitle Generation
    generateSubtitlesButton.addEventListener('click', async function() {
        if (!currentAudioId) {
            alert('Please upload an audio file first');
            return;
        }

        const script = scriptInput.value.trim();
        if (!script) {
            alert('Please enter a script');
            return;
        }

        subtitleLoading.style.display = 'block';
        generateSubtitlesButton.disabled = true;

        try {
            const response = await fetch('/api/v1/generate-subtitles', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    audio_id: currentAudioId,
                    script: script
                })
            });

            if (!response.ok) {
                throw new Error('Failed to generate subtitles');
            }

            const data = await response.json();
            subtitleOutput.textContent = JSON.stringify(data.output, null, 2);
            subtitleOutput.style.display = 'block';
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to generate subtitles. Please try again.');
        } finally {
            subtitleLoading.style.display = 'none';
            generateSubtitlesButton.disabled = false;
        }
    });
});
