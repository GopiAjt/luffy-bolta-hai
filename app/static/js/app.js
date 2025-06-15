document.addEventListener('DOMContentLoaded', function() {
    const generateButton = document.getElementById('generateButton');
    const scriptOutput = document.getElementById('scriptOutput');
    const loading = document.getElementById('loading');

    generateButton.addEventListener('click', async function() {
        // Show loading state
        scriptOutput.style.display = 'none';
        loading.style.display = 'block';
        generateButton.disabled = true;

        try {
            // Make API call
            const response = await fetch('/api/v1/generate-script', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    output_name: "test_script"
                })
            });

            if (!response.ok) {
                throw new Error('Failed to generate script');
            }

            const data = await response.json();
            
            // Display the generated script
            scriptOutput.textContent = data.output.script;
            scriptOutput.style.display = 'block';
            loading.style.display = 'none';

        } catch (error) {
            scriptOutput.textContent = 'Error generating script. Please try again.';
            scriptOutput.style.display = 'block';
            loading.style.display = 'none';
            console.error('Error:', error);
        } finally {
            generateButton.disabled = false;
        }
    });
});
