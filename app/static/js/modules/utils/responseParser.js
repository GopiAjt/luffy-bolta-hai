/**
 * Utility functions for parsing API responses
 */

// Helper function to safely read response as text and optionally parse as JSON
const readResponse = async (response, asJson = true) => {
    // Clone the response so we can read it multiple times if needed
    const responseClone = response.clone();
    const text = await responseClone.text();
    
    if (!asJson) return text;
    
    try {
        return JSON.parse(text);
    } catch (e) {
        console.warn('Failed to parse as JSON, returning as text');
        return text;
    }
};

export const parseJsonResponse = async (response) => {
    try {
        // First try to parse as JSON
        const data = await readResponse(response, true);
        if (typeof data === 'string') {
            // If we got a string back, it means JSON parsing failed
            throw new Error('Response was not valid JSON');
        }
        return data;
    } catch (error) {
        console.warn('Error parsing JSON response:', error);
        // If we get here, either readResponse failed or JSON parsing failed
        const text = await readResponse(response, false);
        console.log('Raw response text:', text);
        
        // Try one more time to parse as JSON in case the error was transient
        try {
            const parsedData = JSON.parse(text);
            console.log('Successfully parsed text as JSON on second attempt');
            return parsedData;
        } catch (e) {
            console.error('Failed to parse response text as JSON:', e);
            throw new Error('Server returned non-JSON response: ' + (text || '').substring(0, 200));
        }
    }
};

export const handleApiError = async (response) => {
    if (!response.ok) {
        let errorMessage = 'An error occurred';
        try {
            // Clone the response before reading it
            const responseClone = response.clone();
            const errorData = await responseClone.json().catch(() => ({}));
            errorMessage = errorData.error || errorData.message || response.statusText || errorMessage;
        } catch (e) {
            console.warn('Error parsing error response:', e);
            try {
                const text = await response.text();
                errorMessage = text || errorMessage;
            } catch (textError) {
                console.error('Could not read error response:', textError);
            }
        }
        const error = new Error(errorMessage);
        error.status = response.status;
        throw error;
    }
    return response;
};
