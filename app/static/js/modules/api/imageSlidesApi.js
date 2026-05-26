import { parseJsonResponse, handleApiError } from '../utils/responseParser.js';

export const generateImageSlides = async (audioId, videoProfile = 'short_vertical') => {
    const response = await fetch('/api/v1/generate-image-slides', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            audio_id: audioId,
            ass_path: 'auto',
            video_profile: videoProfile,
        }),
    });
    await handleApiError(response);
    return parseJsonResponse(response);
};

export const fetchImageSlides = async (audioId, slidesJson = null) => {
    const params = new URLSearchParams({ audio_id: audioId });
    if (slidesJson) {
        params.set('slides_json', slidesJson);
    }
    const response = await fetch(`/api/v1/image-slides?${params.toString()}`);
    await handleApiError(response);
    return parseJsonResponse(response);
};

export const useVivreForSlide = async (audioId, slideIndex, vivreRelative, slidesJson = null) => {
    const response = await fetch('/api/v1/image-slides/use-vivre', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            audio_id: audioId,
            slide_index: slideIndex,
            vivre_relative: vivreRelative,
            slides_json: slidesJson,
        }),
    });
    await handleApiError(response);
    return parseJsonResponse(response);
};

export const uploadSlideImage = async (audioId, slideIndex, file, slidesJson = null) => {
    const formData = new FormData();
    formData.append('audio_id', audioId);
    formData.append('slide_index', String(slideIndex));
    formData.append('file', file);
    if (slidesJson) {
        formData.append('slides_json', slidesJson);
    }

    const response = await fetch('/api/v1/image-slides/upload', {
        method: 'POST',
        body: formData,
    });
    await handleApiError(response);
    return parseJsonResponse(response);
};
