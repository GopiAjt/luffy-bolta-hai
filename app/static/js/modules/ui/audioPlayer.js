/**
 * UI component for audio playback
 */

export class AudioPlayer {
    constructor(audioElementId, nameElementId, durationElementId) {
        this.audioElement = document.getElementById(audioElementId);
        this.nameElement = document.getElementById(nameElementId);
        this.durationElement = document.getElementById(durationElementId);
    }

    setAudioSource(source) {
        if (this.audioElement) {
            this.audioElement.src = source;
        }
    }

    setName(name) {
        if (this.nameElement) {
            this.nameElement.textContent = name;
        }
    }

    setDuration(seconds) {
        if (this.durationElement) {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = Math.floor(seconds % 60);
            this.durationElement.textContent = `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
        }
    }

    play() {
        if (this.audioElement) {
            return this.audioElement.play();
        }
        return Promise.reject(new Error('Audio element not found'));
    }

    pause() {
        if (this.audioElement) {
            this.audioElement.pause();
        }
    }

    onTimeUpdate(callback) {
        if (this.audioElement) {
            this.audioElement.ontimeupdate = callback;
        }
    }

    getCurrentTime() {
        return this.audioElement ? this.audioElement.currentTime : 0;
    }

    getDuration() {
        return this.audioElement ? this.audioElement.duration : 0;
    }
}
