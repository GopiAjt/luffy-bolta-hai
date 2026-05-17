import { parseJsonResponse, handleApiError } from '../utils/responseParser.js';

/**
 * API client for script generation
 */

export const generateScript = async (prompt = null) => {
    const payload = { language: 'hindi' };
    if (prompt && prompt.trim()) {
        payload.topic = prompt.trim();
    }

    const response = await fetch('/api/v1/generate-script', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    });

    await handleApiError(response);
    return parseJsonResponse(response);
};
