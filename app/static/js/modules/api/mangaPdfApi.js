import { parseJsonResponse, handleApiError } from '../utils/responseParser.js';

export const uploadMangaPdf = async (file) => {
    const formData = new FormData();
    formData.append('pdf', file);

    const response = await fetch('/api/v1/upload-manga-pdf', {
        method: 'POST',
        body: formData
    });

    await handleApiError(response);
    return parseJsonResponse(response);
};

export const generateScriptFromPdf = async (pdfId, topic = '') => {
    const response = await fetch('/api/v1/generate-script-from-pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            pdf_id: pdfId,
            topic
        })
    });

    await handleApiError(response);
    return parseJsonResponse(response);
};

export const generatePdfSlides = async (pdfId, audioId) => {
    const response = await fetch('/api/v1/generate-pdf-slides', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            pdf_id: pdfId,
            audio_id: audioId
        })
    });

    await handleApiError(response);
    return parseJsonResponse(response);
};

export const generateMangaVideo = async (pdfId, topic = '', options = {}) => {
    const response = await fetch('/api/v1/generate-manga-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            pdf_id: pdfId,
            topic,
            language: options.language || 'English',
            subtitle_style: options.subtitleStyle || 'pro',
            quality_mode: options.qualityMode || 'pro'
        })
    });

    await handleApiError(response);
    return parseJsonResponse(response);
};

export const getMangaSession = async (pdfId) => {
    const response = await fetch(`/api/v1/manga-session/${pdfId}`);
    await handleApiError(response);
    return parseJsonResponse(response);
};
