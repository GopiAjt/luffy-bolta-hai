/**
 * UI component for video playback
 */

export class VideoPlayer {
    constructor(videoElementId, outputElementId) {
        this.videoElement = document.getElementById(videoElementId);
        this.outputElement = document.getElementById(outputElementId);
    }

    setSource(source) {
        if (this.videoElement) {
            this.videoElement.src = source;
        }
    }

    show() {
        if (this.videoElement) {
            this.videoElement.style.display = 'block';
        }
        if (this.outputElement) {
            this.outputElement.style.display = 'block';
        }
    }

    hide() {
        if (this.videoElement) {
            this.videoElement.style.display = 'none';
        }
        if (this.outputElement) {
            this.outputElement.style.display = 'none';
        }
    }

    setMessage(message, isError = false) {
        if (this.outputElement) {
            this.outputElement.textContent = message;
            this.outputElement.style.color = isError ? '#dc3545' : '';
        }
    }

    createDownloadLink(url, filename) {
        if (!this.outputElement) return null;

        // Remove existing download link if any
        const existingLink = this.outputElement.querySelector('a');
        if (existingLink) {
            this.outputElement.removeChild(existingLink);
        }

        // Create new download link
        const downloadLink = document.createElement('a');
        downloadLink.href = url;
        downloadLink.download = filename || 'download';
        downloadLink.textContent = 'Download Video';
        downloadLink.className = 'btn btn-primary mt-2';
        downloadLink.style.display = 'inline-block';

        // Add compatibility note
        const compatibilityNote = document.createElement('p');
        compatibilityNote.textContent = 'If the video doesn\'t play in the browser, try downloading it instead.';
        compatibilityNote.style.fontSize = '0.9em';
        compatibilityNote.style.color = '#666';
        compatibilityNote.style.marginTop = '10px';

        this.outputElement.appendChild(document.createElement('br'));
        this.outputElement.appendChild(downloadLink);
        this.outputElement.appendChild(document.createElement('br'));
        this.outputElement.appendChild(compatibilityNote);

        return downloadLink;
    }
}
