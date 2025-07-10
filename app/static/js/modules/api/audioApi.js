import { parseJsonResponse, handleApiError } from '../utils/responseParser.js';

/**
 * API client for audio operations
 */

export const uploadAudio = async (file, onProgress) => {
    const formData = new FormData();
    formData.append('audio', file);

    const xhr = new XMLHttpRequest();
    
    return new Promise((resolve, reject) => {
        xhr.upload.onprogress = (event) => {
            if (event.lengthComputable && onProgress) {
                const percentComplete = Math.round((event.loaded / event.total) * 100);
                onProgress(percentComplete);
            }
        };

        xhr.onload = async () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const response = new Response(xhr.responseText, {
                        status: xhr.status,
                        statusText: xhr.statusText,
                        headers: new Headers(xhr.getAllResponseHeaders())
                    });
                    const data = await parseJsonResponse(response);
                    resolve(data);
                } catch (error) {
                    reject(error);
                }
            } else {
                reject(new Error(`Upload failed with status ${xhr.status}`));
            }
        };

        xhr.onerror = () => {
            reject(new Error('Network error during upload'));
        };

        xhr.open('POST', '/api/v1/upload-audio', true);
        xhr.send(formData);
    });
};

export const getAudioMetadata = async (audioId) => {
    const response = await fetch(`/api/v1/audio/${audioId}/metadata`);
    await handleApiError(response);
    return parseJsonResponse(response);
};
