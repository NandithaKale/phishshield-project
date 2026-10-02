const API_URL = "http://127.0.0.1:5000/predict";

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

    if (message.type !== "CHECK_URL") {
        return;
    }

    fetch(API_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            url: message.url
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Backend returned HTTP " + response.status);
        }

        return response.json();
    })
    .then(data => {
        sendResponse({
            success: true,
            data: data
        });
    })
    .catch(error => {
        console.error("PhishShield backend error:", error);

        sendResponse({
            success: false,
            error: error.message
        });
    });

    return true;
});