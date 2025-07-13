import { parseJsonResponse, handleApiError } from '../utils/responseParser.js';

/**
 * API client for video operations
 */

export const generateVideo = async (audioId, subtitleFile, slidesJsonPath) => {
    const response = await fetch('/api/v1/generate-final-video', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            audio_id: audioId,
            subtitle_file: subtitleFile,
            slides_json: slidesJsonPath
        })
    });

    await handleApiError(response);
    return parseJsonResponse(response);
};

export const getVideoStatus = async (videoId) => {
    const response = await fetch(`/api/v1/video/${videoId}/status`);
    await handleApiError(response);
    return parseJsonResponse(response);
};

export const generateSlideshow = async (audioId) => {
    const response = await fetch('/api/v1/generate-image-slides', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            audio_id: audioId,
            // We'll let the server determine the latest ASS file
            ass_path: 'auto' 
        })
    });

    await handleApiError(response);
    return parseJsonResponse(response);
};
