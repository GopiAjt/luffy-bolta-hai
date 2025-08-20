import { parseJsonResponse, handleApiError } from '../utils/responseParser.js';

/**
 * API client for script generation
 */

/**
 * Generate a script with the specified style and optional custom prompt
 * @param {string} style - The style of script to generate. One of:
 *   - 'maximum_engagement' (default)
 *   - 'community_focused'
 *   - 'theory_specialist'
 *   - 'viral_formula'
 *   - 'psychological_warfare'
 * @param {string} [prompt] - Optional custom prompt to use instead of the style's default prompt
 * @returns {Promise<Object>} The generated script data
 */
export const generateScript = async (style = 'maximum_engagement', prompt = null) => {
    const requestBody = {};
    
    if (style) {
        requestBody.style = style;
    }
    
    if (prompt) {
        requestBody.script = prompt;
    }
    
    try {
        const response = await fetch('/api/v1/generate-script', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'Failed to generate script');
        }

        const data = await response.json();
        return data.output || data; // Handle both old and new response formats
    } catch (error) {
        console.error('Error generating script:', error);
        throw error;
    }
};

/**
 * Get the list of available script styles
 * @returns {Array<{value: string, label: string, description: string}>} List of style objects
 */
export const getScriptStyles = () => {
    return [
        {
            value: 'maximum_engagement',
            label: 'Maximum Engagement',
            description: 'High-energy, clickbait-style content optimized for maximum views and engagement'
        },
        {
            value: 'community_focused',
            label: 'Community Focused',
            description: 'Content designed to spark discussions and debates in the comments'
        },
        {
            value: 'theory_specialist',
            label: 'Theory Specialist',
            description: 'In-depth analysis and theories for hardcore One Piece fans'
        },
        {
            value: 'viral_formula',
            label: 'Viral Formula',
            description: 'Algorithm-optimized content designed to go viral'
        },
        {
            value: 'psychological_warfare',
            label: 'Psychological Warfare',
            description: 'Content that plays on psychological triggers for maximum watch time'
        }
    ];
};
