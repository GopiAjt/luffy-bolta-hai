/**
 * Global error handling utilities
 */

/**
 * Shows an error message to the user
 * @param {HTMLElement} element - The element to display the error in
 * @param {Error|string} error - The error to display
 * @param {string} [defaultMessage='An error occurred'] - Default message if error is not an Error object
 */
export const showError = (element, error, defaultMessage = 'An error occurred') => {
    if (!element) {
        console.error('Error element not found');
        return;
    }

    let errorMessage = defaultMessage;
    
    if (error instanceof Error) {
        errorMessage = error.message;
    } else if (typeof error === 'string') {
        errorMessage = error;
    } else if (error && typeof error === 'object' && error.message) {
        errorMessage = error.message;
    }

    element.textContent = `Error: ${errorMessage}`;
    element.style.display = 'block';
    element.style.color = '#dc3545';
    
    // Log the full error to console for debugging
    if (error instanceof Error) {
        console.error(error);
    } else {
        console.error('Error:', error);
    }
};

/**
 * Wraps an async function with error handling
 * @param {Function} fn - The async function to wrap
 * @param {Object} options - Options
 * @param {HTMLElement} [options.errorElement] - Element to display errors in
 * @param {string} [options.errorMessage] - Default error message
 * @param {Function} [options.onError] - Custom error handler
 * @returns {Function} Wrapped function with error handling
 */
export const withErrorHandling = (fn, {
    errorElement,
    errorMessage = 'An error occurred',
    onError
} = {}) => {
    return async (...args) => {
        try {
            return await fn(...args);
        } catch (error) {
            if (errorElement) {
                showError(errorElement, error, errorMessage);
            }
            
            if (onError) {
                onError(error);
            }
            
            // Re-throw to allow further error handling if needed
            throw error;
        }
    };
};
