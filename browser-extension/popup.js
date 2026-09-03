const API_URL = "http://127.0.0.1:5000/predict";

let currentUrl = "";

const currentUrlElement = document.getElementById("currentUrl");
const checkButton = document.getElementById("checkButton");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("errorBox");
const result = document.getElementById("result");
const verdictCard = document.getElementById("verdictCard");
const verdictIcon = document.getElementById("verdictIcon");
const verdictLabel = document.getElementById("verdictLabel");
const confidence = document.getElementById("confidence");
const explanations = document.getElementById("explanations");


function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}


function clearError() {
    errorBox.textContent = "";
    errorBox.classList.add("hidden");
}


function setLoading(isLoading) {
    if (isLoading) {
        loading.classList.remove("hidden");
        checkButton.disabled = true;
    } else {
        loading.classList.add("hidden");
        checkButton.disabled = false;
    }
}


function renderExplanations(items) {
    explanations.innerHTML = "";

    if (!items || items.length === 0) {
        explanations.innerHTML =
            '<div class="explanation-item">' +
            '<div class="explanation-reason">' +
            'No explanation data was returned.' +
            '</div>' +
            '</div>';

        return;
    }

    items.forEach(item => {
        const explanationItem = document.createElement("div");
        explanationItem.className = "explanation-item";

        const feature = document.createElement("div");
        feature.className = "explanation-feature";
        feature.textContent = item.feature;

        const reason = document.createElement("div");
        reason.className = "explanation-reason";
        reason.textContent = item.reason;

        const impact = document.createElement("div");
        impact.className = "explanation-impact";

        const impactValue = Number(item.impact);

        if (impactValue > 0) {
            impact.textContent =
                "Impact: +" + impactValue.toFixed(4);
        } else {
            impact.textContent =
                "Impact: " + impactValue.toFixed(4);
        }

        explanationItem.appendChild(feature);
        explanationItem.appendChild(reason);
        explanationItem.appendChild(impact);

        explanations.appendChild(explanationItem);
    });
}


function renderResult(data) {
    const isPhishing = Boolean(data.is_phishing);
    const percentage = Number(data.confidence) * 100;

    verdictCard.classList.remove("safe", "phishing");

    if (isPhishing) {
        verdictCard.classList.add("phishing");
        verdictIcon.textContent = "!";
        verdictLabel.textContent = "PHISHING DETECTED";
    } else {
        verdictCard.classList.add("safe");
        verdictIcon.textContent = "OK";
        verdictLabel.textContent = "LIKELY SAFE";
    }

    confidence.textContent =
        "Confidence: " + percentage.toFixed(2) + "%";

    renderExplanations(data.explanation);

    result.classList.remove("hidden");
}


async function checkURL() {
    clearError();

    if (!currentUrl) {
        showError("Could not determine the current website URL.");
        return;
    }

    setLoading(true);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                url: currentUrl
            })
        });

        let data = {};

        try {
            data = await response.json();
        } catch (error) {
            throw new Error("Invalid response from the backend.");
        }

        if (!response.ok) {
            throw new Error(
                data.error ||
                "Backend returned HTTP " + response.status
            );
        }

        if (
            typeof data.is_phishing === "undefined" ||
            typeof data.confidence === "undefined"
        ) {
            throw new Error(
                "Backend response is missing prediction data."
            );
        }

        renderResult(data);

    } catch (error) {
        console.error("PhishShield error:", error);

        showError(
            error.message +
            " Make sure Flask is running with: python run.py"
        );
    } finally {
        setLoading(false);
    }
}


function getCurrentTabURL() {
    chrome.tabs.query(
        {
            active: true,
            currentWindow: true
        },
        function(tabs) {

            if (!tabs || tabs.length === 0) {
                showError("Could not access the current browser tab.");
                return;
            }

            const tab = tabs[0];

            if (!tab.url) {
                showError("Could not read the current website URL.");
                return;
            }

            currentUrl = tab.url;
            currentUrlElement.textContent = currentUrl;
        }
    );
}


checkButton.addEventListener("click", checkURL);

getCurrentTabURL();