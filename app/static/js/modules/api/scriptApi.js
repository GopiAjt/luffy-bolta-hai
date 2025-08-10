import { parseJsonResponse, handleApiError } from '../utils/responseParser.js';

/**
 * API client for script generation
 * 
 * @returns {Promise<Object>} Script data with title, script, description, and tags
 * @example
 * {
 *   title: "🌟 ONE PIECE'S MOST UNDERRATED LEGENDS!",
 *   script: "The generated script text...",
 *   description: "Discover the most underrated legends in One Piece history...",
 *   tags: ["#OnePiece", "#Anime", "#Manga"]
 * }
 */
export const generateScript = async () => {
    const response = await fetch('/api/v1/generate-script', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    });

    await handleApiError(response);
    const data = await parseJsonResponse(response);
    
    if (!data || !data.output) {
        throw new Error('Invalid response format from server');
    }
    
    return data.output;
};
