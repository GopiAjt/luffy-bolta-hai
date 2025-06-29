document.addEventListener('DOMContentLoaded', function () {
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
    generateButton.addEventListener('click', async function () {
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
    audioFileInput.addEventListener('change', async function (e) {
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
                onprogress: function (progressEvent) {
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
    generateSubtitlesButton.addEventListener('click', async function () {
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
            subtitleOutput.textContent = 'Subtitles generated successfully!';
            subtitleOutput.style.display = 'block';

            // Store the subtitle file name for video generation
            const subtitleFile = data.output.ass_file;

            // Show video generation controls
            const videoControls = document.getElementById('videoControls');
            videoControls.style.display = 'block';

            // Set up video generation button
            const generateVideoButton = document.getElementById('generateVideoButton');
            generateVideoButton.onclick = () => generateVideo(currentAudioId, subtitleFile);

        } catch (error) {
            console.error('Error:', error);
            alert('Failed to generate subtitles. Please try again.');
        } finally {
            subtitleLoading.style.display = 'none';
            generateSubtitlesButton.disabled = false;
        }
    });

    // Video Generation
    async function generateVideo(audioId, subtitleFile) {
        console.log('Starting video generation with:', { audioId, subtitleFile });
        const videoLoading = document.getElementById('videoLoading');
        const videoOutput = document.getElementById('videoOutput');
        const videoPreview = document.getElementById('videoPreview');
        const generateVideoButton = document.getElementById('generateVideoButton');

        videoLoading.style.display = 'block';
        videoOutput.style.display = 'none';
        videoOutput.textContent = 'Generating video...';
        videoPreview.style.display = 'none';
        generateVideoButton.disabled = true;

        // Clear previous content
        videoOutput.innerHTML = '';

        try {
            const response = await fetch('/api/v1/generate-video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    audio_id: audioId,
                    subtitle_id: subtitleFile,
                    background_color: 'green'
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to generate video');
            }

            // First, log response headers to check content type
            console.log('Response headers:', Object.fromEntries([...response.headers.entries()]));

            // Try to parse as JSON first
            try {
                const responseData = await response.json();
                console.log('Parsed JSON response:', responseData);
                return responseData;
            } catch (jsonError) {
                console.warn('Failed to parse as JSON, trying as text:', jsonError);

                // If JSON parse fails, try to get the raw text
                const responseText = await response.text();
                console.log('Raw response text:', responseText);

                // Try to parse the text as JSON (in case it's JSON but content-type was wrong)
                try {
                    const parsedData = JSON.parse(responseText);
                    console.log('Successfully parsed text as JSON:', parsedData);
                    return parsedData;
                } catch (e) {
                    console.error('Failed to parse response text as JSON:', e);
                    throw new Error('Server returned non-JSON response: ' + responseText.substring(0, 200));
                }
            }

            // Show success message
            const successMsg = document.createElement('div');
            successMsg.textContent = 'Video generated successfully!';
            successMsg.style.marginBottom = '10px';
            videoOutput.appendChild(successMsg);
            videoOutput.style.display = 'block';

            // Debug: Log all top-level properties
            console.log('Response data keys:', Object.keys(responseData));

            // Debug: Log the complete response data structure
            console.log('Response data structure:', JSON.stringify(responseData, null, 2));

            // Get the video URL from the response with fallbacks
            const videoUrl = responseData.download_url ||
                (responseData.output && responseData.output.download_url) ||
                responseData.video_url;

            if (!videoUrl) {
                console.error('No video URL found in response. Available keys:', Object.keys(responseData));
                throw new Error('Failed to get video URL from server. Check browser console for details.');
            }

            // Determine if the file is MP4
            const isMP4 = responseData.format === 'mp4' || videoUrl.toLowerCase().endsWith('.mp4');

            console.log('Video details - URL:', videoUrl, 'Format:', isMP4 ? 'MP4' : 'MKV');

            // Set up video source for preview (use preview endpoint)
            videoPreview.innerHTML = ''; // Clear any existing sources
            const previewUrl = videoUrl.replace('/api/v1/download/', '/api/v1/preview/');
            const source = document.createElement('source');
            source.src = previewUrl;
            source.type = isMP4 ? 'video/mp4' : 'video/x-matroska';
            videoPreview.appendChild(source);
            // Show the video element after setting the source
            videoPreview.style.display = 'block';
            videoPreview.load();
            videoPreview.oncanplay = () => {
                videoPreview.style.display = 'block';
            };
            videoPreview.onerror = (e) => {
                console.error('Video failed to load:', e);
                videoOutput.innerHTML += '<div style="color:red;">Video failed to load. Check the server logs and network tab for details.</div>';
            };

            // Add download link
            const downloadLink = document.createElement('a');
            downloadLink.href = videoUrl;
            downloadLink.textContent = `Download ${isMP4 ? 'MP4' : 'MKV'} Video`;
            downloadLink.download = responseData.video_file || 'video_download' + (isMP4 ? '.mp4' : '.mkv');
            downloadLink.className = 'upload-button';
            downloadLink.style.display = 'inline-block';
            downloadLink.style.marginTop = '10px';

            // Add message about browser compatibility if needed
            if (!isMP4) {
                const compatibilityNote = document.createElement('p');
                compatibilityNote.textContent = 'Note: Your browser may not support MKV preview. The download will still work.';
                compatibilityNote.style.fontSize = '0.9em';
                compatibilityNote.style.color = '#666';
                compatibilityNote.style.marginTop = '10px';
                videoOutput.appendChild(compatibilityNote);

                // Add a note that we recommend downloading for best compatibility
                const downloadNote = document.createElement('p');
                downloadNote.textContent = 'For best compatibility, we recommend downloading the video.';
                downloadNote.style.fontSize = '0.9em';
                downloadNote.style.color = '#666';
                downloadNote.style.marginTop = '5px';
                videoOutput.appendChild(downloadNote);
            }

            // Clear any existing download link
            const existingLink = videoOutput.querySelector('a');
            if (existingLink) {
                videoOutput.removeChild(existingLink);
            }

            videoOutput.appendChild(document.createElement('br'));
            videoOutput.appendChild(downloadLink);

        } catch (error) {
            console.error('Error:', error);
            videoOutput.textContent = `Error: ${error.message}`;
            videoOutput.style.display = 'block';
        } finally {
            videoLoading.style.display = 'none';
            generateVideoButton.disabled = false;
        }
    }

    // Image Slides Generation (no input box, use latest .ass file in uploads)
    const generateImageSlidesButton = document.getElementById('generateImageSlidesButton');
    const imageSlidesStatus = document.getElementById('imageSlidesStatus');
    const imageSlidesOutput = document.getElementById('imageSlidesOutput');

    if (generateImageSlidesButton) {
        generateImageSlidesButton.addEventListener('click', async function () {
            imageSlidesStatus.textContent = 'Finding latest .ass file...';
            imageSlidesOutput.style.display = 'none';
            try {
                // Fetch the latest .ass file from the uploads folder (backend endpoint needed)
                const res = await fetch('/api/v1/latest-ass-file');
                const data = await res.json();
                if (!data.path) {
                    imageSlidesStatus.textContent = 'No .ass file found in uploads.';
                    return;
                }
                imageSlidesStatus.textContent = 'Generating image slides...';
                const response = await fetch('/api/v1/generate-image-slides', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ass_path: data.path })
                });
                const slidesData = await response.json();
                if (slidesData.status === 'success') {
                    imageSlidesStatus.textContent = 'Image slides generated!';
                    imageSlidesOutput.textContent = JSON.stringify(slidesData.slides, null, 2);
                    imageSlidesOutput.style.display = 'block';
                } else {
                    imageSlidesStatus.textContent = 'Error: ' + (slidesData.message || 'Unknown error');
                }
            } catch (err) {
                imageSlidesStatus.textContent = 'Error: ' + err.message;
            }
        });
    }

    // Slideshow Generation
    const generateSlideshowButton = document.getElementById('generateSlideshowButton');
    const slideshowStatus = document.getElementById('slideshowStatus');
    const slideshowOutput = document.getElementById('slideshowOutput');

    if (generateSlideshowButton) {
        generateSlideshowButton.addEventListener('click', async function () {
            slideshowStatus.textContent = 'Finding latest image slides JSON...';
            slideshowOutput.style.display = 'none';
            try {
                // Find the latest .image_slides.json file in uploads
                const res = await fetch('/api/v1/latest-image-slides-json');
                const data = await res.json();
                if (!data.path) {
                    slideshowStatus.textContent = 'No image slides JSON found in uploads.';
                    return;
                }
                slideshowStatus.textContent = 'Generating slideshow video...';
                const response = await fetch('/api/v1/generate-slideshow', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        slides_json: data.path,
                        images_dir: 'app/output/image_slides'
                    })
                });
                const result = await response.json();
                if (result.status === 'success') {
                    slideshowStatus.textContent = 'Slideshow video generated!';
                    slideshowOutput.innerHTML = `<a href="/api/v1/download/${result.output_path.split('/').pop()}" download>Download Slideshow Video</a>`;
                    slideshowOutput.style.display = 'block';
                } else {
                    slideshowStatus.textContent = 'Error: ' + (result.message || 'Unknown error');
                }
            } catch (err) {
                slideshowStatus.textContent = 'Error: ' + err.message;
            }
        });
    }
});
