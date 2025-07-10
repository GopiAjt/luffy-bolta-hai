/**
 * Utility functions for parsing API responses
 */

export const parseJsonResponse = async (response) => {
    try {
        return await response.json();
    } catch (jsonError) {
        console.warn('Failed to parse as JSON, trying as text:', jsonError);
        const responseText = await response.text();
        console.log('Raw response text:', responseText);
        
        // Try to parse the text as JSON (in case it's JSON but content-type was wrong)
        try {
            const parsedData = JSON.parse(responseText);
            console.log('Successfully parsed text as JSON:', parsedData);
            return parsedData;
        } catch (e) {
            console.error('Failed to parse response text as JSON:', e);
            throw new Error('Server returned non-JSON response: ' + responseText.substring(0, 200));
        }
    }
};

export const handleApiError = async (response) => {
    if (!response.ok) {
        let errorMessage = 'An error occurred';
        try {
            const errorData = await response.json();
            errorMessage = errorData.error || errorData.message || errorMessage;
        } catch (e) {
            const text = await response.text();
            errorMessage = text || errorMessage;
        }
        throw new Error(errorMessage);
    }
    return response;
};
