import { parseJsonResponse, handleApiError } from '../utils/responseParser.js';

/**
 * API client for subtitle operations
 */

export const generateSubtitles = async (audioId, script) => {
    const response = await fetch('/api/v1/generate-subtitles', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            audio_id: audioId,
            script: script
        })
    });

    await handleApiError(response);
    return parseJsonResponse(response);
};

export const getLatestSubtitleFile = async () => {
    const response = await fetch('/api/v1/latest-ass-file');
    await handleApiError(response);
    return parseJsonResponse(response);
};
