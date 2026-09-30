// Browser UI controller for the web demo. It talks to the FastAPI routes in
// khmer_transliteration/web.py and does not affect the Windows TSF IME.
const input = document.querySelector("#romanInput");
const clearButton = document.querySelector("#clearButton");
const collectButton = document.querySelector("#collectButton");
const normalizedText = document.querySelector("#normalizedText");
const suggestionsEl = document.querySelector("#suggestions");
const emptyState = document.querySelector("#emptyState");
const countText = document.querySelector("#countText");
const outputText = document.querySelector("#outputText");
const modelStatus = document.querySelector("#modelStatus");
const collectStatus = document.querySelector("#collectStatus");

let activeRequest = 0;
let lastSelectedKhmer = "";

// Direct keyboard conversions for non-romanized keys in the browser UI.
const KHMER_DIGITS_BY_CODE = {
    Digit0: "០",
    Digit1: "១",
    Digit2: "២",
    Digit3: "៣",
    Digit4: "៤",
    Digit5: "៥",
    Digit6: "៦",
    Digit7: "៧",
    Digit8: "៨",
    Digit9: "៩",
    Numpad0: "០",
    Numpad1: "១",
    Numpad2: "២",
    Numpad3: "៣",
    Numpad4: "៤",
    Numpad5: "៥",
    Numpad6: "៦",
    Numpad7: "៧",
    Numpad8: "៨",
    Numpad9: "៩",
};

const KHMER_SHIFT_TOP_ROW_BY_CODE = {
    Digit1: "!",
    Digit2: "ៗ",
    Digit3: "\"",
    Digit4: "៛",
    Digit5: "%",
    Digit6: "៍",
    Digit7: "័",
    Digit8: "៏",
    Digit9: "(",
    Digit0: ")",
    Minus: "=",
    Equal: "៎",
};

const KHMER_DIRECT_KEYS_BY_CODE = {
    Period: "។",
    NumpadDecimal: "។",
};

function insertTextAtCursor(element, text) {
    // Preserve cursor position when inserting Khmer punctuation/digits.
    const start = element.selectionStart ?? element.value.length;
    const end = element.selectionEnd ?? element.value.length;
    element.value = `${element.value.slice(0, start)}${text}${element.value.slice(end)}`;
    const nextPosition = start + text.length;
    element.setSelectionRange(nextPosition, nextPosition);
}

function getKhmerKeyboardCharacter(event) {
    // Let browser/system shortcuts pass through untouched.
    if (event.ctrlKey || event.metaKey || event.altKey) {
        return null;
    }

    if (event.shiftKey && KHMER_SHIFT_TOP_ROW_BY_CODE[event.code]) {
        return KHMER_SHIFT_TOP_ROW_BY_CODE[event.code];
    }

    if (!event.shiftKey && KHMER_DIGITS_BY_CODE[event.code]) {
        return KHMER_DIGITS_BY_CODE[event.code];
    }

    if (!event.shiftKey && KHMER_DIRECT_KEYS_BY_CODE[event.code]) {
        return KHMER_DIRECT_KEYS_BY_CODE[event.code];
    }

    return null;
}

function handleKhmerKeyboardInput(event) {
    // Route Khmer number/punctuation keys into the selected text output.
    const khmerCharacter = getKhmerKeyboardCharacter(event);

    if (!khmerCharacter) {
        return;
    }

    event.preventDefault();

    if (event.currentTarget === outputText) {
        insertTextAtCursor(outputText, khmerCharacter);
        return;
    }

    insertTextAtCursor(outputText, khmerCharacter);
    input.value = "";
    renderSuggestions({ normalized: "", suggestions: [] });
}

function sourceLabel(source) {
    // Short label displayed on each suggestion card.
    if (source === "dictionary_exact") {
        return "exact";
    }

    if (source === "dictionary_completion") {
        return "completion";
    }

    if (source === "dictionary_compound") {
        return "compound";
    }

    if (source === "phrase_space") {
        return "phrase";
    }

    if (source === "dictionary_fuzzy") {
        return "fuzzy";
    }

    if (source.startsWith("rule_")) {
        return "rule";
    }

    if (source.startsWith("direct_")) {
        return "direct";
    }

    return source;
}

function sourceClass(source) {
    // CSS class controls the pill color for each suggestion source.
    if (source === "dictionary_exact") {
        return "exact";
    }

    if (source === "dictionary_completion") {
        return "completion";
    }

    if (source === "dictionary_compound") {
        return "completion";
    }

    if (source === "phrase_space") {
        return "completion";
    }

    if (source === "dictionary_fuzzy") {
        return "completion";
    }

    if (source.startsWith("rule_")) {
        return "rule";
    }

    if (source.startsWith("direct_")) {
        return "direct";
    }

    return "";
}

function formatScore(value) {
    if (value === undefined || value === null || value === "") {
        return null;
    }

    return Number(value).toFixed(4);
}

function renderSuggestions(data) {
    // Rebuild suggestion cards from the latest API response.
    const suggestions = data.suggestions || [];
    normalizedText.textContent = data.normalized || "empty";
    countText.textContent = `${suggestions.length} result${suggestions.length === 1 ? "" : "s"}`;
    suggestionsEl.innerHTML = "";
    emptyState.classList.toggle("visible", suggestions.length === 0);

    for (const suggestion of suggestions) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "suggestion";
        button.setAttribute("aria-label", `Insert ${suggestion.khmer}`);

        const mlScore = formatScore(suggestion.ml_score);
        const manualScore = formatScore(suggestion.score);
        const rankScore = formatScore(suggestion.rank_score);
        const datasetScore = formatScore(suggestion.dataset_match_score);
        const contextScore = formatScore(suggestion.previous_word_context_score);
        const compoundPairScore = formatScore(suggestion.compound_pair_context_score);
        const historyScore = formatScore(suggestion.user_history_score);
        const fuzzyBoost = formatScore(suggestion.high_confidence_fuzzy_boost);
        const compoundPenalty = formatScore(suggestion.compound_no_dataset_penalty);
        const sourceWeight = formatScore(suggestion.source_rank_weight);
        const source = sourceLabel(suggestion.source);
        const sourceCss = sourceClass(suggestion.source);

        button.innerHTML = `
            <span class="khmer">${suggestion.khmer}</span>
            <span class="meta">
                <span class="pill ${sourceCss}">${source}</span>
                ${suggestion.manual_label !== undefined ? `<span class="pill">label ${suggestion.manual_label}</span>` : ""}
                ${rankScore ? `<span class="pill">rank ${rankScore}</span>` : ""}
                ${suggestion.rank_reason ? `<span class="pill">${suggestion.rank_reason}</span>` : ""}
                ${sourceWeight ? `<span class="pill">source ${sourceWeight}</span>` : ""}
                ${datasetScore ? `<span class="pill">dataset ${datasetScore}${suggestion.dataset_romanized ? ` ${suggestion.dataset_romanized}` : ""}</span>` : ""}
                ${contextScore ? `<span class="pill">context ${contextScore}</span>` : ""}
                ${compoundPairScore ? `<span class="pill">pair ${compoundPairScore}</span>` : ""}
                ${historyScore ? `<span class="pill">history ${historyScore}</span>` : ""}
                ${fuzzyBoost ? `<span class="pill">fuzzy boost ${fuzzyBoost}</span>` : ""}
                ${compoundPenalty ? `<span class="pill">compound penalty ${compoundPenalty}</span>` : ""}
                ${mlScore ? `<span class="pill">ML ${mlScore}</span>` : ""}
                ${manualScore ? `<span class="pill">manual ${manualScore}</span>` : ""}
            </span>
        `;

        button.addEventListener("click", async () => {
            // Click commits the candidate, clears the romanized input, and
            // records history so future ranking can personalize results.
            const selectedInput = input.value.trim();
            const previousKhmer = lastSelectedKhmer;

            outputText.value += suggestion.khmer;
            lastSelectedKhmer = suggestion.khmer;
            input.value = "";
            renderSuggestions({ normalized: "", suggestions: [] });
            input.focus();

            if (selectedInput) {
                await recordSelection(
                    selectedInput,
                    suggestion.khmer,
                    previousKhmer,
                    suggestion.compound_segments || []
                );
            }
        });

        suggestionsEl.appendChild(button);
    }
}

async function recordSelection(query, khmer, previousKhmer, compoundSegments = []) {
    // Save personal selection and previous-word context through /api/select.
    try {
        await fetch("/api/select", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                q: query,
                khmer,
                previous_khmer: previousKhmer,
                compound_segments: compoundSegments,
            }),
        });
    } catch (error) {
        collectStatus.textContent = "Selection history could not be saved.";
    }
}

async function fetchSuggestions() {
    // Send the current romanized input to /api/suggest. activeRequest prevents
    // stale responses from overwriting newer typing results.
    const query = input.value.trim();
    const requestId = ++activeRequest;

    if (!query) {
        renderSuggestions({ normalized: "", suggestions: [] });
        return;
    }

    modelStatus.textContent = "Ranking...";

    try {
        const allowVowels = outputText.value.length > 0;
        const response = await fetch(
            `/api/suggest?q=${encodeURIComponent(query)}&allow_vowels=${allowVowels}&previous_word=${encodeURIComponent(lastSelectedKhmer)}`
        );

        if (!response.ok) {
            throw new Error(`Request failed: ${response.status}`);
        }

        const data = await response.json();

        if (requestId !== activeRequest) {
            return;
        }

        renderSuggestions(data);
        modelStatus.textContent = "Model ready";
    } catch (error) {
        if (requestId !== activeRequest) {
            return;
        }

        modelStatus.textContent = "API error";
        suggestionsEl.innerHTML = "";
        emptyState.textContent = "Could not load suggestions. Check the server console.";
        emptyState.classList.add("visible");
    }
}

input.addEventListener("input", fetchSuggestions);
input.addEventListener("keydown", handleKhmerKeyboardInput);
outputText.addEventListener("keydown", handleKhmerKeyboardInput);

clearButton.addEventListener("click", () => {
    input.value = "";
    renderSuggestions({ normalized: "", suggestions: [] });
    collectStatus.textContent = "";
    input.focus();
});

collectButton.addEventListener("click", async () => {
    // Confirm sends all generated candidates for this input to the review CSV.
    const query = input.value.trim();

    if (!query) {
        collectStatus.textContent = "Type a word before confirming.";
        input.focus();
        return;
    }

    collectStatus.textContent = "Sending candidates to review CSV...";

    try {
        const response = await fetch(`/api/collect?q=${encodeURIComponent(query)}`, {
            method: "POST",
        });

        if (!response.ok) {
            throw new Error(`Request failed: ${response.status}`);
        }

        const data = await response.json();
        collectStatus.textContent = `${data.message} Open data/label_data.csv to label them.`;
    } catch (error) {
        collectStatus.textContent = "Could not save candidates. Check the server console.";
    }
});

renderSuggestions({ normalized: "", suggestions: [] });
