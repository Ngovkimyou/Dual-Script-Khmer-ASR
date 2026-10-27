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
const asrStatus = document.querySelector("#asrStatus");
const recordButton = document.querySelector("#recordButton");
const stopRecordButton = document.querySelector("#stopRecordButton");
const audioFileInput = document.querySelector("#audioFileInput");
const asrKhmerOutput = document.querySelector("#asrKhmerOutput");
const asrRomanizedOutput = document.querySelector("#asrRomanizedOutput");

let activeRequest = 0;
let lastSelectedKhmer = "";
let recordingAudioContext = null;
let recordingStream = null;
let recordingSource = null;
let recordingProcessor = null;
let recordingSilenceGain = null;
let recordingChunks = [];
let isRecording = false;
let isAsrBusy = false;

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

function setAsrStatus(message, isError = false) {
    if (!asrStatus) {
        return;
    }
    asrStatus.textContent = message;
    asrStatus.classList.toggle("error", isError);
}

function setAsrControls() {
    if (!recordButton || !stopRecordButton || !audioFileInput) {
        return;
    }
    recordButton.disabled = isRecording || isAsrBusy;
    stopRecordButton.disabled = !isRecording;
    audioFileInput.disabled = isRecording || isAsrBusy;
}

function mergeAudioChunks(chunks) {
    const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
    const merged = new Float32Array(length);
    let offset = 0;
    for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
    }
    return merged;
}

function resampleAudio(samples, sourceRate, targetRate) {
    if (sourceRate === targetRate) {
        return samples;
    }

    const outputLength = Math.max(1, Math.round(samples.length * targetRate / sourceRate));
    const output = new Float32Array(outputLength);
    const ratio = sourceRate / targetRate;
    for (let index = 0; index < outputLength; index += 1) {
        const sourcePosition = index * ratio;
        const lower = Math.floor(sourcePosition);
        const upper = Math.min(lower + 1, samples.length - 1);
        const weight = sourcePosition - lower;
        output[index] = samples[lower] * (1 - weight) + samples[upper] * weight;
    }
    return output;
}

function writeWavString(view, offset, value) {
    for (let index = 0; index < value.length; index += 1) {
        view.setUint8(offset + index, value.charCodeAt(index));
    }
}

function encodeWav(samples, sampleRate) {
    const targetRate = 16000;
    const monoSamples = resampleAudio(samples, sampleRate, targetRate);
    const buffer = new ArrayBuffer(44 + monoSamples.length * 2);
    const view = new DataView(buffer);
    writeWavString(view, 0, "RIFF");
    view.setUint32(4, 36 + monoSamples.length * 2, true);
    writeWavString(view, 8, "WAVE");
    writeWavString(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, targetRate, true);
    view.setUint32(28, targetRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeWavString(view, 36, "data");
    view.setUint32(40, monoSamples.length * 2, true);

    let offset = 44;
    for (const sample of monoSamples) {
        const clipped = Math.max(-1, Math.min(1, sample));
        view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
        offset += 2;
    }
    return new Blob([buffer], { type: "audio/wav" });
}

async function sendAudioForTranscription(audioBlob) {
    isAsrBusy = true;
    setAsrControls();
    setAsrStatus("Transcribing...");

    try {
        const response = await fetch("/api/asr", {
            method: "POST",
            headers: { "Content-Type": "audio/wav" },
            body: audioBlob,
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || `Request failed: ${response.status}`);
        }
        asrKhmerOutput.value = data.khmer_text || "";
        asrRomanizedOutput.value = data.romanized_text || "";
        const latency = Number(data.latency_seconds);
        setAsrStatus(Number.isFinite(latency) ? `Done · ${latency.toFixed(2)}s` : "Done");
    } catch (error) {
        setAsrStatus(error.message || "ASR request failed.", true);
    } finally {
        isAsrBusy = false;
        setAsrControls();
    }
}

async function warmUpAsrModel() {
    if (isRecording || isAsrBusy) {
        return;
    }
    isAsrBusy = true;
    setAsrControls();
    setAsrStatus("Loading model...");
    try {
        const response = await fetch("/api/asr/warmup", { method: "POST" });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || `Request failed: ${response.status}`);
        }
        setAsrStatus(`Model ready · ${data.device}`);
    } catch (error) {
        setAsrStatus(error.message || "ASR model could not be loaded.", true);
    } finally {
        isAsrBusy = false;
        setAsrControls();
    }
}

async function startRecording() {
    if (isRecording || isAsrBusy) {
        return;
    }

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass || !navigator.mediaDevices?.getUserMedia) {
        setAsrStatus("This browser does not support microphone recording.", true);
        return;
    }

    try {
        recordingStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1 },
        });
        recordingAudioContext = new AudioContextClass();
        recordingSource = recordingAudioContext.createMediaStreamSource(recordingStream);
        recordingProcessor = recordingAudioContext.createScriptProcessor(4096, 1, 1);
        recordingSilenceGain = recordingAudioContext.createGain();
        recordingSilenceGain.gain.value = 0;
        recordingChunks = [];
        recordingProcessor.onaudioprocess = (event) => {
            if (isRecording) {
                recordingChunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
            }
        };
        recordingSource.connect(recordingProcessor);
        recordingProcessor.connect(recordingSilenceGain);
        recordingSilenceGain.connect(recordingAudioContext.destination);
        isRecording = true;
        setAsrStatus("Recording...");
        setAsrControls();
    } catch (error) {
        if (recordingStream) {
            recordingStream.getTracks().forEach((track) => track.stop());
        }
        setAsrStatus(error.message || "Microphone permission was denied.", true);
    }
}

async function stopRecording() {
    if (!isRecording) {
        return;
    }

    isRecording = false;
    const chunks = recordingChunks;
    const sourceRate = recordingAudioContext?.sampleRate || 44100;
    recordingProcessor?.disconnect();
    recordingSource?.disconnect();
    recordingSilenceGain?.disconnect();
    recordingStream?.getTracks().forEach((track) => track.stop());
    await recordingAudioContext?.close();
    recordingAudioContext = null;
    recordingProcessor = null;
    recordingSource = null;
    recordingSilenceGain = null;
    recordingStream = null;
    recordingChunks = [];
    setAsrControls();

    if (!chunks.length) {
        setAsrStatus("No audio was recorded.", true);
        return;
    }
    await sendAudioForTranscription(encodeWav(mergeAudioChunks(chunks), sourceRate));
}

function handleAudioFile(event) {
    const file = event.target.files?.[0];
    if (!file) {
        return;
    }
    if (!file.name.toLowerCase().endsWith(".wav")) {
        setAsrStatus("Please select a WAV file.", true);
        event.target.value = "";
        return;
    }
    sendAudioForTranscription(file);
    event.target.value = "";
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

recordButton?.addEventListener("click", startRecording);
stopRecordButton?.addEventListener("click", stopRecording);
audioFileInput?.addEventListener("change", handleAudioFile);
setAsrControls();
warmUpAsrModel();

renderSuggestions({ normalized: "", suggestions: [] });
