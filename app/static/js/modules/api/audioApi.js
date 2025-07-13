import { parseJsonResponse, handleApiError } from '../utils/responseParser.js';

/**
 * API client for audio operations
 */

export const uploadAudio = async (file, onProgress) => {
    console.log('Starting audio upload for file:', file.name, 'size:', file.size, 'type:', file.type);
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
                    // Parse the headers string into an object
                    const headers = new Headers();
                    const headersString = xhr.getAllResponseHeaders();
                    const headerPairs = headersString.trim().split(/[\r\n]+/);
                    
                    headerPairs.forEach(line => {
                        const parts = line.split(': ');
                        const header = parts.shift();
                        const value = parts.join(': ');
                        if (header) headers.append(header, value);
                    });
                    
                    const response = new Response(xhr.responseText, {
                        status: xhr.status,
                        statusText: xhr.statusText,
                        headers: headers
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
