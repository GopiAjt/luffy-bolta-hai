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
