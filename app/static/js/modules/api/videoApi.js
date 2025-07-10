import { parseJsonResponse, handleApiError } from '../utils/responseParser.js';

/**
 * API client for video operations
 */

export const generateVideo = async (audioId, subtitleFileId, slideshowVideoId) => {
    const response = await fetch('/api/v1/generate-video', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            audio_id: audioId,
            subtitle_id: subtitleFileId,
            slideshow_video_id: slideshowVideoId
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
    const response = await fetch('/api/v1/generate-slideshow', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ audio_id: audioId })
    });

    await handleApiError(response);
    return parseJsonResponse(response);
};
