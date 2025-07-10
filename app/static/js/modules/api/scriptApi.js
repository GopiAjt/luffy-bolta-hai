import { parseJsonResponse, handleApiError } from '../utils/responseParser.js';

/**
 * API client for script generation
 */

export const generateScript = async (prompt = "Generate a 30-60 second One Piece narration script") => {
    const response = await fetch('/api/v1/generate-script', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ script: prompt })
    });

    await handleApiError(response);
    return parseJsonResponse(response);
};
