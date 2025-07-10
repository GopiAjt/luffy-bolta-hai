/**
 * UI component for showing loading indicators
 */

export class LoadingIndicator {
    constructor(elementId) {
        this.element = document.getElementById(elementId);
    }

    show() {
        if (this.element) {
            this.element.style.display = 'block';
        }
    }

    hide() {
        if (this.element) {
            this.element.style.display = 'none';
        }
    }

    setText(text) {
        if (this.element) {
            this.element.textContent = text;
        }
    }
}
