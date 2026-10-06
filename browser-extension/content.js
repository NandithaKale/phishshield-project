document.addEventListener("click", function (event) {

    const link = event.target.closest("a");

    if (!link || !link.href) {
        return;
    }

    const url = link.href;

    // Only check HTTP/HTTPS links
    if (
        !url.startsWith("http://") &&
        !url.startsWith("https://")
    ) {
        return;
    }

    // Stop the browser from navigating immediately
    event.preventDefault();
    event.stopPropagation();

    chrome.runtime.sendMessage(
        {
            type: "CHECK_URL",
            url: url
        },
        function (response) {

            // Extension communication error
            if (chrome.runtime.lastError) {

                console.error(
                    "PhishShield extension error:",
                    chrome.runtime.lastError.message
                );

                // If checking fails, allow normal navigation
                window.location.href = url;
                return;
            }

            // Backend/checking failed
            if (!response || !response.success) {

                console.error(
                    "PhishShield could not check URL:",
                    response
                        ? response.error
                        : "No response"
                );

                // If checking fails, allow normal navigation
                window.location.href = url;
                return;
            }

            const data = response.data;

            // Phishing detected
            if (data.is_phishing) {

                showWarning(url, data);

            } else {

                // Safe URL → continue normally
                window.location.href = url;
            }
        }
    );
});


function showWarning(url, data) {

    // Remove an existing warning if there is one
    const existingWarning =
        document.getElementById("phishshield-warning");

    if (existingWarning) {
        existingWarning.remove();
    }

    const overlay = document.createElement("div");

    overlay.id = "phishshield-warning";

    const explanations = data.explanation || [];

    let xaiHTML = "";


    // -----------------------------
    // XAI EXPLANATIONS
    // -----------------------------

    if (explanations.length > 0) {

        xaiHTML = explanations
            .slice(0, 5)
            .map(function (item) {

                const impact = Number(item.impact);

                return `
                    <div style="
                        margin-bottom: 10px;
                        padding: 11px;
                        background: #0b1020;
                        border: 1px solid #25304a;
                        border-radius: 8px;
                        text-align: left;
                    ">

                        <div style="
                            font-size: 13px;
                            font-weight: bold;
                            color: #e8ecf7;
                        ">
                            ${escapeHtml(item.feature)}
                        </div>

                        <div style="
                            margin-top: 4px;
                            font-size: 11px;
                            line-height: 1.5;
                            color: #9aa7bf;
                        ">
                            ${escapeHtml(item.reason)}
                        </div>

                        <div style="
                            margin-top: 5px;
                            font-size: 11px;
                            font-weight: bold;
                            color: #f87171;
                        ">
                            Impact:
                            ${impact >= 0 ? "+" : ""}
                            ${impact.toFixed(4)}
                        </div>

                    </div>
                `;

            })
            .join("");

    } else {

        xaiHTML = `
            <div style="
                padding: 12px;
                background: #0b1020;
                border: 1px solid #25304a;
                border-radius: 8px;
                color: #9aa7bf;
                font-size: 12px;
            ">
                No XAI explanation data was returned.
            </div>
        `;
    }


    // -----------------------------
    // WARNING UI
    // -----------------------------

    overlay.innerHTML = `

        <div style="
            position: fixed;
            inset: 0;
            z-index: 2147483647;
            background: #0b1020;
            color: #e8ecf7;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: Arial, sans-serif;
            overflow-y: auto;
            padding: 30px 0;
        ">

            <div style="
                width: 90%;
                max-width: 560px;
                padding: 30px;
                background: #121a2d;
                border: 2px solid #dc2626;
                border-radius: 16px;
                text-align: center;
                box-shadow:
                    0 20px 60px rgba(0,0,0,0.5);
            ">

                <!-- Warning icon -->

                <div style="
                    font-size: 46px;
                    margin-bottom: 10px;
                ">
                    🚨
                </div>


                <!-- Title -->

                <h1 style="
                    margin: 0 0 10px;
                    color: #f87171;
                    font-size: 28px;
                ">
                    PHISHING DETECTED
                </h1>


                <!-- Description -->

                <p style="
                    color: #aeb9cf;
                    font-size: 14px;
                    line-height: 1.6;
                    margin-bottom: 18px;
                ">
                    PhishShield has detected suspicious
                    characteristics in this URL.
                </p>


                <!-- Suspicious URL -->

                <div style="
                    margin: 18px 0;
                    padding: 14px;
                    background: #0b1020;
                    border: 1px solid #25304a;
                    border-radius: 8px;
                    word-break: break-all;
                    text-align: left;
                    font-size: 12px;
                    color: #cbd5e1;
                ">
                    ${escapeHtml(url)}
                </div>


                <!-- Confidence -->

                <div style="
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 22px;
                ">
                    Confidence:
                    ${(Number(data.confidence) * 100).toFixed(2)}%
                </div>


                <!-- XAI -->

                <div style="
                    text-align: left;
                    margin-bottom: 20px;
                ">

                    <div style="
                        font-size: 15px;
                        font-weight: bold;
                        color: #e8ecf7;
                        margin-bottom: 10px;
                    ">
                        🧠 Why was this flagged?
                    </div>

                    ${xaiHTML}

                </div>


                <!-- BUTTONS -->

                <div>

                    <button
                        id="phishshield-back"
                        type="button"
                        style="
                            padding: 12px 22px;
                            margin-right: 8px;
                            border: none;
                            border-radius: 8px;
                            background: #2563eb;
                            color: white;
                            font-weight: bold;
                            cursor: pointer;
                        "
                    >
                        ← Go Back
                    </button>


                    <button
                        id="phishshield-continue"
                        type="button"
                        style="
                            padding: 12px 22px;
                            border: 1px solid #475569;
                            border-radius: 8px;
                            background: transparent;
                            color: #cbd5e1;
                            font-weight: bold;
                            cursor: pointer;
                        "
                    >
                        Continue Anyway
                    </button>

                </div>

            </div>

        </div>
    `;


    document.documentElement.appendChild(overlay);


    // -----------------------------
    // GO BACK BUTTON
    // -----------------------------

    document
        .getElementById("phishshield-back")
        .addEventListener("click", function () {

            // IMPORTANT:
            // Do NOT use window.history.back().
            //
            // The suspicious navigation was already blocked.
            // Removing the warning keeps the user safely
            // on the current page.

            overlay.remove();
        });


    // -----------------------------
    // CONTINUE ANYWAY BUTTON
    // -----------------------------

    document
        .getElementById("phishshield-continue")
        .addEventListener("click", function () {

            // User explicitly chooses to continue
            window.location.href = url;
        });
}


// -----------------------------
// HTML ESCAPING
// -----------------------------

function escapeHtml(value) {

    const div = document.createElement("div");

    div.textContent = value;

    return div.innerHTML;
}