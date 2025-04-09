const apiUrl = "http://localhost:8000"; // Change this to your API URL if needed

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("content-form");
    const promptInput = document.getElementById("prompt-input");
    const outputDiv = document.getElementById("output");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const prompt = promptInput.value;

        if (prompt) {
            outputDiv.innerHTML = "Generating content...";

            try {
                const response = await fetch(`${apiUrl}/generate?prompt=${encodeURIComponent(prompt)}`);
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }
                const data = await response.json();
                outputDiv.innerHTML = `<h3>Generated Content:</h3><p>${data.output}</p>`;
            } catch (error) {
                outputDiv.innerHTML = `<p>Error: ${error.message}</p>`;
            }
        } else {
            outputDiv.innerHTML = "<p>Please enter a prompt.</p>";
        }
    });
});