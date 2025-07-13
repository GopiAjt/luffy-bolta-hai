import { parseJsonResponse, handleApiError } from '../utils/responseParser.js';

/**
 * API client for subtitle operations
 */

export const generateSubtitles = async (audioId, script) => {
    console.log('=== Starting generateSubtitles API call ===');
    console.log('Audio ID:', audioId);
    console.log('Script length:', script ? script.length : 0);
    
    try {
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

        console.log('API Response status:', response.status, response.statusText);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('API Error response:', errorText);
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }
        
        const result = await parseJsonResponse(response);
        console.log('generateSubtitles successful, result:', result);
        return result;
    } catch (error) {
        console.error('Error in generateSubtitles:', error);
        throw error;
    }
};

export const getLatestSubtitleFile = async () => {
    console.log('=== getLatestSubtitleFile called ===');
    try {
        console.log('Fetching latest ASS file from /api/v1/latest-ass-file');
        const response = await fetch('/api/v1/latest-ass-file');
        console.log('Response status:', response.status, response.statusText);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Error response:', errorText);
            throw new Error(`Failed to get latest subtitle file: ${response.status} ${response.statusText}`);
        }
        
        const data = await parseJsonResponse(response);
        console.log('Latest subtitle file data:', data);
        return data;
    } catch (error) {
        console.error('Error in getLatestSubtitleFile:', error);
        throw error;
    }
};

/**
 * Fetches the latest expressions file from the server
 * @returns {Promise<Object>} Object containing the path to the latest expressions file or null if not found
 */
/**
 * Fetches the latest expressions file from the server
 * @returns {Promise<Object>} Object containing the path to the latest expressions file or null if not found
 */
export const getLatestExpressionsFile = async () => {
    console.log('=== getLatestExpressionsFile called ===');
    try {
        console.log('Fetching latest expressions file from /api/v1/latest-expressions-file');
        const response = await fetch('/api/v1/latest-expressions-file');
        console.log('Response status:', response.status, response.statusText);
        
        if (!response.ok) {
            if (response.status === 404) {
                console.log('No expressions file found, returning null');
                return { path: null, exists: false };
            }
            const errorText = await response.text();
            console.error('Error response:', errorText);
            throw new Error(`Failed to get latest expressions file: ${response.status} ${response.statusText}`);
        }
        
        const data = await parseJsonResponse(response);
        console.log('Latest expressions file data:', data);
        
        // Handle different response formats
        if (data && data.path) {
            return { 
                path: data.path, 
                exists: data.exists || false,
                status: data.status || 'success'
            };
        } else if (data && data.status === 'not_found') {
            console.log('No expressions file found (not_found status)');
            return { path: null, exists: false, status: 'not_found' };
        } else if (typeof data === 'string') {
            // Handle case where response is just a path string
            return { 
                path: data, 
                exists: true,
                status: 'success'
            };
        }
        
        console.warn('Unexpected response format for expressions file:', data);
        return { path: null, exists: false, status: 'unknown_format' };
    } catch (error) {
        console.error('Error in getLatestExpressionsFile:', error);
        // Return null instead of throwing to allow video generation to continue without expressions
        return { 
            path: null, 
            exists: false, 
            status: 'error',
            error: error.message 
        };
    }
};
