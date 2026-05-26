/**
 * Renders per-slide upload cards (search term + file picker + preview).
 */
export class ImageSlidesUploadPanel {
    constructor(containerId, { onUploadComplete, onStatusChange } = {}) {
        this.container = document.getElementById(containerId);
        this.onUploadComplete = onUploadComplete || (() => {});
        this.onStatusChange = onStatusChange || (() => {});
        this.audioId = null;
        this.slidesJson = null;
    }

    render(payload) {
        if (!this.container) {
            return;
        }

        this.audioId = payload.audio_id;
        this.slidesJson = payload.slides_json;
        const slides = payload.slides || [];
        const uploaded = payload.uploaded ?? 0;
        const total = payload.total ?? slides.length;

        this.container.style.display = 'block';
        this.container.innerHTML = `
            <div class="image-slides-upload-header mb-3">
                <p class="text-white mb-1">
                    <strong>Upload images</strong> — use each search term to find the right still, then choose a file.
                </p>
                <p class="text-white-50 small mb-0" id="imageSlidesUploadProgress">
                    ${uploaded} of ${total} slides have an image
                </p>
            </div>
            <div class="image-slides-upload-list" id="imageSlidesUploadList"></div>
        `;

        const list = this.container.querySelector('#imageSlidesUploadList');
        slides.forEach((slide) => {
            list.appendChild(this._createSlideCard(slide));
        });

        this.onStatusChange({ uploaded, total, complete: payload.complete });
    }

    _createSlideCard(slide) {
        const card = document.createElement('div');
        card.className = 'image-slide-card mb-3 p-3 rounded';
        card.dataset.slideIndex = String(slide.index);

        const hasImage = slide.has_image;
        const isAi = slide.visual_source === 'ai_generate';
        const sourceBadge = isAi
            ? '<span class="badge bg-info text-dark">AI generate</span>'
            : '<span class="badge bg-primary">Vivre / search</span>';
        const previewHtml = hasImage
            ? `<img src="${this._withCacheBust(slide.preview_url)}" alt="Slide ${slide.index + 1}" class="image-slide-preview" />`
            : `<div class="image-slide-preview image-slide-preview--empty">${isAi ? 'Generate with AI, then upload' : 'No image yet'}</div>`;

        const assetBlock = isAi
            ? ''
            : `<p class="text-white mb-1"><strong>Pick image for:</strong> <span class="image-slide-query">${this._escape(slide.image_search_query || '')}</span></p>
               ${this._vivreSuggestionsHtml(slide)}`;

        const aiBlock = isAi && slide.ai_image_prompt
            ? `<div class="ai-prompt-block mb-2">
                    <label class="form-label text-white small mb-1">AI image prompt (copy into Imagen / DALL·E / Midjourney)</label>
                    <textarea class="form-control form-control-sm ai-prompt-text" readonly rows="4">${this._escape(slide.ai_image_prompt)}</textarea>
                    <button type="button" class="btn btn-sm btn-outline-light mt-1 copy-ai-prompt-btn">Copy prompt</button>
               </div>`
            : isAi
              ? '<p class="text-warning small">AI slide — prompt missing; regenerate slides or upload manually.</p>'
              : '';

        card.innerHTML = `
            <div class="d-flex flex-wrap justify-content-between align-items-start gap-2 mb-2">
                <div>
                    <span class="badge bg-secondary me-2">Slide ${slide.index + 1}</span>
                    ${sourceBadge}
                    <span class="text-white-50 small ms-2">${slide.start_time} → ${slide.end_time}</span>
                </div>
                <span class="badge ${hasImage ? 'bg-success' : 'bg-warning text-dark'} slide-upload-badge">
                    ${hasImage ? 'Uploaded' : 'Needs image'}
                </span>
            </div>
            <p class="text-white-50 small mb-1">${this._escape(slide.subtitle_text || slide.summary || '')}</p>
            ${aiBlock}
            ${assetBlock}
            <div class="row g-3 align-items-center">
                <div class="col-md-5">${previewHtml}</div>
                <div class="col-md-7">
                    <label class="form-label text-white small mb-1">${isAi ? 'Upload AI-generated image' : 'Or upload your own (JPG, PNG, WebP)'}</label>
                    <input type="file" class="form-control form-control-sm slide-image-input" accept="image/jpeg,image/png,image/webp,image/gif" />
                    <div class="slide-upload-status small mt-2 text-white-50"></div>
                </div>
            </div>
        `;

        const copyBtn = card.querySelector('.copy-ai-prompt-btn');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                const ta = card.querySelector('.ai-prompt-text');
                if (ta) {
                    navigator.clipboard.writeText(ta.value).then(() => {
                        copyBtn.textContent = 'Copied!';
                        setTimeout(() => { copyBtn.textContent = 'Copy prompt'; }, 1500);
                    });
                }
            });
        }

        card.querySelectorAll('[data-vivre-relative]').forEach((btn) => {
            btn.addEventListener('click', () => {
                const relative = btn.getAttribute('data-vivre-relative');
                this._applyVivreAsset(card, slide.index, relative);
            });
        });

        const input = card.querySelector('.slide-image-input');
        input.addEventListener('change', (event) => this._handleFileSelected(card, slide.index, event.target.files[0]));

        return card;
    }

    _vivreSuggestionsHtml(slide) {
        if (slide.visual_source === 'ai_generate') {
            return '';
        }
        const suggestions = slide.vivre_suggestions || [];
        if (!suggestions.length) {
            return '';
        }
        const chips = suggestions.map((s) => {
            const kind = s.asset_kind || 'asset';
            const kindLabel = kind === 'symbol' ? 'Flag' : kind === 'location' ? 'Place' : 'Character';
            return `
                <button type="button" class="btn btn-sm btn-outline-light me-1 mb-1 vivre-suggest-btn"
                    data-vivre-relative="${this._escapeAttr(s.relative)}"
                    title="${this._escapeAttr(s.label)}">
                    <img src="${s.preview_url}" alt="" class="vivre-suggest-thumb" />
                    <span class="vivre-suggest-label">${this._escape(kindLabel)}: ${this._escape(s.label)}</span>
                </button>
            `;
        }).join('');
        return `
            <div class="vivre-suggestions mb-2">
                <span class="text-white-50 small d-block mb-1">From Vivre pack (flags &amp; places):</span>
                <div class="vivre-suggestions-row">${chips}</div>
            </div>
        `;
    }

    async _applyVivreAsset(card, slideIndex, vivreRelative) {
        const statusEl = card.querySelector('.slide-upload-status');
        statusEl.textContent = 'Applying Vivre image...';
        try {
            const { useVivreForSlide } = await import('../api/imageSlidesApi.js');
            const result = await useVivreForSlide(
                this.audioId,
                slideIndex,
                vivreRelative,
                this.slidesJson
            );
            this._refreshCardFromResult(card, slideIndex, result);
            this.onUploadComplete(result);
        } catch (error) {
            statusEl.textContent = error.message || 'Failed to apply Vivre image';
            statusEl.classList.add('text-danger');
        }
    }

    _refreshCardFromResult(card, slideIndex, result) {
        const updated = (result.slides_detail || []).find((s) => s.index === slideIndex);
        const statusEl = card.querySelector('.slide-upload-status');
        const badge = card.querySelector('.slide-upload-badge');
        if (updated?.preview_url) {
            const previewCol = card.querySelector('.col-md-5');
            previewCol.innerHTML = `<img src="${this._withCacheBust(updated.preview_url)}" alt="Slide ${slideIndex + 1}" class="image-slide-preview" />`;
        }
        if (badge) {
            badge.className = 'badge bg-success slide-upload-badge';
            badge.textContent = 'Uploaded';
        }
        statusEl.textContent = 'Saved from Vivre pack.';
        statusEl.classList.remove('text-danger');
        statusEl.classList.add('text-success');
        const progress = document.getElementById('imageSlidesUploadProgress');
        if (progress && result.upload_status) {
            const { uploaded, total } = result.upload_status;
            progress.textContent = `${uploaded} of ${total} slides have an image`;
            this.onStatusChange({
                uploaded,
                total,
                complete: result.upload_status.complete,
            });
        }
    }

    async _handleFileSelected(card, slideIndex, file) {
        if (!file || !this.audioId) {
            return;
        }

        const statusEl = card.querySelector('.slide-upload-status');
        const badge = card.querySelector('.slide-upload-badge');
        const input = card.querySelector('.slide-image-input');

        statusEl.textContent = 'Uploading...';
        input.disabled = true;

        try {
            const { uploadSlideImage } = await import('../api/imageSlidesApi.js');
            const result = await uploadSlideImage(
                this.audioId,
                slideIndex,
                file,
                this.slidesJson
            );

            this._refreshCardFromResult(card, slideIndex, result);
            this.onUploadComplete(result);
        } catch (error) {
            statusEl.textContent = error.message || 'Upload failed';
            statusEl.classList.add('text-danger');
        } finally {
            input.disabled = false;
        }
    }

    _withCacheBust(url) {
        if (!url) {
            return '';
        }
        const separator = url.includes('?') ? '&' : '?';
        return `${url}${separator}t=${Date.now()}`;
    }

    _escape(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    _escapeAttr(text) {
        return String(text || '')
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/</g, '&lt;');
    }
}
