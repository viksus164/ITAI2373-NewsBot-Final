document.addEventListener("DOMContentLoaded", () => {
    const textarea = document.getElementById("text");
    const charCount = document.getElementById("charCount");
    const form = document.getElementById("analysisForm");
    const submitButton = document.getElementById("submitButton");

    if (textarea && charCount) {
        const updateCount = () => {
            const count = textarea.value.length;
            charCount.textContent = count;
            charCount.style.color = count >= 20 ? "#15803d" : "#b91c1c";
        };
        textarea.addEventListener("input", updateCount);
        updateCount();
    }

    if (form && submitButton) {
        form.addEventListener("submit", () => {
            submitButton.disabled = true;
            submitButton.textContent = "Analyzing... first run may take a moment";
        });
    }
});

function downloadResults() {
    if (typeof analysisResults === "undefined") {
        alert("No analysis results are available to download.");
        return;
    }

    const payload = {
        timestamp: new Date().toISOString(),
        original_text: originalText,
        analysis: analysisResults,
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
    });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "newsbot_web_analysis.json";
    link.click();
    URL.revokeObjectURL(link.href);
}
