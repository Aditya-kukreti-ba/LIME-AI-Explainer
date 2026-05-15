"use strict";
/* ── LIME Explainer — frontend ── */

const API = "";   // same-origin Flask

// ════════════════════════════════════════════════════════
// Utility helpers
// ════════════════════════════════════════════════════════

function $(id)    { return document.getElementById(id); }
function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function showError(msg) {
  $("error-toast-msg").textContent = msg;
  show($("error-toast"));
  setTimeout(() => hide($("error-toast")), 8000);
}

function setLoading(spinner, btn, loading) {
  if (loading) {
    show(spinner); btn.disabled = true;
    btn.lastChild.textContent = " Running…";
  } else {
    hide(spinner); btn.disabled = false;
    btn.lastChild.textContent = btn.dataset.label;
  }
}

// ════════════════════════════════════════════════════════
// Tab switching
// ════════════════════════════════════════════════════════

// Tab switching is handled by router.js

// ════════════════════════════════════════════════════════
// TEXT TAB
// ════════════════════════════════════════════════════════

const MODEL_EXAMPLES = {
  sentiment:  "This product is absolutely fantastic! The quality is outstanding and I love every feature. Highly recommend to anyone looking for the best experience.",
  newsgroups: "The goalie made some unbelievable saves in last night's NHL game. The power play in the third period was a real turning point for the team.",
  openai:     "I've been using this app for 3 months and it completely transformed my workflow. The support team is incredibly responsive and helpful.",
};

document.querySelectorAll(".radio-card[data-value]").forEach(card => {
  if (!card.closest("#text-model-cards")) return;
  card.addEventListener("click", () => {
    document.querySelectorAll("#text-model-cards .radio-card").forEach(c => c.classList.remove("active"));
    card.classList.add("active");
    card.querySelector("input").checked = true;
    const val = card.dataset.value;
    if (val === "openai") show($("openai-config")); else hide($("openai-config"));
  });
});

$("load-example-btn").addEventListener("click", () => {
  const model = document.querySelector('input[name="text-model"]:checked')?.value || "sentiment";
  $("input-text").value = MODEL_EXAMPLES[model] || MODEL_EXAMPLES.sentiment;
});

const nSamplesSlider = $("n-samples-text");
const nSamplesVal    = $("n-samples-text-val");
nSamplesSlider.addEventListener("input", () => { nSamplesVal.textContent = nSamplesSlider.value; });

$("validate-key-btn").addEventListener("click", async () => {
  const key    = $("openai-key").value.trim();
  const status = $("key-status");
  if (!key) { status.textContent = "Enter a key first."; status.className = "key-status err"; return; }
  status.textContent = "Checking…"; status.className = "key-status";
  try {
    const res  = await fetch(`${API}/api/validate-key`, {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ api_key: key }),
    });
    const data = await res.json();
    status.textContent = data.message;
    status.className   = `key-status ${data.valid ? "ok" : "err"}`;
  } catch { status.textContent = "Network error."; status.className = "key-status err"; }
});

const explainTextBtn = $("explain-text-btn");
explainTextBtn.addEventListener("click", async () => {
  const text = $("input-text").value.trim();
  if (!text) { showError("Please enter some text to explain."); return; }
  const modelType  = document.querySelector('input[name="text-model"]:checked')?.value || "sentiment";
  const openaiKey  = $("openai-key").value.trim();
  const task       = $("task-desc").value.trim() || "sentiment analysis";
  const classNames = $("class-names").value.trim();
  const nSamples   = parseInt(nSamplesSlider.value, 10);
  if (modelType === "openai" && !openaiKey) { showError("Please enter your OpenAI API key."); return; }

  const spinner = $("text-spinner");
  setLoading(spinner, explainTextBtn, true);
  hide($("text-results")); show($("text-placeholder"));
  $("text-placeholder").querySelector("p").textContent = "Running LIME… this may take a moment.";
  try {
    const res  = await fetch(`${API}/api/explain/text`, {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ text, model_type: modelType, openai_key: openaiKey,
                             task, class_names: classNames || null, n_samples: nSamples }),
    });
    const data = await res.json();
    if (data.error) { showError(data.error); return; }
    renderTextResults(data);
  } catch (err) { showError(`Request failed: ${err.message}`); }
  finally {
    setLoading(spinner, explainTextBtn, false);
    $("text-placeholder").querySelector("p").textContent = "Run an explanation to see results";
  }
});

function renderTextResults(data) {
  const badgeRow = $("text-pred-badges");
  badgeRow.innerHTML = "";
  for (const [cls, prob] of Object.entries(data.class_probabilities)) {
    const badge = document.createElement("div");
    badge.className = `pred-badge ${cls === data.predicted_class ? "winner" : "loser"}`;
    badge.textContent = `${cls}  ${(prob * 100).toFixed(1)}%`;
    badgeRow.appendChild(badge);
  }
  $("text-lime-chart").src = `data:image/png;base64,${data.image}`;
  const expForClass = data.explanations?.[data.predicted_class] || [];
  const topFeatures = expForClass.slice().sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 12);
  $("text-feature-table").innerHTML = buildFeatureTable(topFeatures, `Predicted class: ${data.predicted_class}`);
  hide($("text-placeholder")); show($("text-results"));
}

// ════════════════════════════════════════════════════════
// TABULAR TAB
// ════════════════════════════════════════════════════════

let tabularModels   = [];
let currentTabModel = null;

async function loadTabularModels() {
  try {
    const res  = await fetch(`${API}/api/models/tabular`);
    const data = await res.json();
    tabularModels = data.models;
    const sel = $("tabular-model-select");
    tabularModels.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.id; opt.textContent = m.name;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => selectTabularModel(sel.value));
    selectTabularModel(tabularModels[0]?.id);
  } catch { showError("Failed to load tabular model list."); }
}

function selectTabularModel(modelId) {
  currentTabModel = tabularModels.find(m => m.id === modelId);
  if (!currentTabModel) return;
  $("tabular-model-desc").textContent = currentTabModel.description;
  buildFeatureInputs(currentTabModel);
}

function buildFeatureInputs(model) {
  const container = $("tabular-feature-inputs");
  container.innerHTML = "";
  model.features.forEach(fname => {
    const range  = model.feature_ranges?.[fname] || {};
    const min    = range.min     ?? 0;
    const max    = range.max     ?? 100;
    const defVal = range.default ?? ((min + max) / 2);
    const step   = range.step    ?? 0.01;
    const wrapper = document.createElement("div");
    wrapper.className = "feature-item";
    wrapper.innerHTML = `<label title="${fname}">${fname}</label>
      <input type="number" data-feature="${fname}" min="${min}" max="${max}" step="${step}" value="${defVal}" />`;
    container.appendChild(wrapper);
  });
}

function randomizeFeatures() {
  if (!currentTabModel) return;
  currentTabModel.features.forEach(fname => {
    const range = currentTabModel.feature_ranges?.[fname] || {};
    const min   = range.min  ?? 0;
    const max   = range.max  ?? 100;
    const step  = range.step ?? 0.01;
    const dec   = step < 1 ? (String(step).split(".")[1]?.length ?? 2) : 0;
    const val   = +(Math.random() * (max - min) + min).toFixed(dec);
    const input = document.querySelector(`[data-feature="${fname}"]`);
    if (input) input.value = val;
  });
}

$("tabular-randomize-btn").addEventListener("click", randomizeFeatures);

const explainTabBtn = $("explain-tabular-btn");
explainTabBtn.addEventListener("click", async () => {
  if (!currentTabModel) { showError("Select a model first."); return; }
  const features = {};
  let valid = true;
  currentTabModel.features.forEach(fname => {
    const input = document.querySelector(`[data-feature="${fname}"]`);
    const val   = parseFloat(input?.value);
    if (isNaN(val)) { showError(`Invalid value for feature: ${fname}`); valid = false; }
    else features[fname] = val;
  });
  if (!valid) return;

  const spinner = $("tabular-spinner");
  setLoading(spinner, explainTabBtn, true);
  hide($("tabular-results")); show($("tabular-placeholder"));
  $("tabular-placeholder").querySelector("p").textContent = "Running LIME… please wait.";
  try {
    const res  = await fetch(`${API}/api/explain/tabular`, {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ model_type: currentTabModel.id, features, n_samples: 500 }),
    });
    const data = await res.json();
    if (data.error) { showError(data.error); return; }
    renderTabularResults(data);
  } catch (err) { showError(`Request failed: ${err.message}`); }
  finally {
    setLoading(spinner, explainTabBtn, false);
    $("tabular-placeholder").querySelector("p").textContent = "Run an explanation to see results";
  }
});

function renderTabularResults(data) {
  const badgeRow = $("tabular-pred-badges");
  badgeRow.innerHTML = "";
  for (const [cls, prob] of Object.entries(data.class_probabilities)) {
    const badge = document.createElement("div");
    badge.className = `pred-badge ${cls === data.predicted_class ? "winner" : "loser"}`;
    badge.textContent = `${cls}  ${(prob * 100).toFixed(1)}%`;
    badgeRow.appendChild(badge);
  }
  $("tabular-lime-chart").src = `data:image/png;base64,${data.image}`;
  const fw = Array.isArray(data.explanations) ? data.explanations : [];
  $("tabular-feature-table").innerHTML = buildFeatureTable(fw, `Predicted class: ${data.predicted_class}`);
  hide($("tabular-placeholder")); show($("tabular-results"));
}

// ════════════════════════════════════════════════════════
// IMAGE TAB
// ════════════════════════════════════════════════════════

let cnnModels       = [];
let selectedCnnId   = "resnet50";
let selectedImageFile = null;

async function loadCnnModels() {
  try {
    const res  = await fetch(`${API}/api/models/cnn`);
    const data = await res.json();
    cnnModels = data.models;
    const container = $("cnn-model-cards");
    container.innerHTML = "";
    cnnModels.forEach((m, i) => {
      const icons = ["🧠", "⚡", "🔬"];
      const label = document.createElement("label");
      label.className = `radio-card ${i === 0 ? "active" : ""}`;
      label.dataset.value = m.id;
      label.innerHTML = `
        <input type="radio" name="cnn-model" value="${m.id}" ${i === 0 ? "checked" : ""} hidden />
        <div class="rc-icon">${icons[i] || "🤖"}</div>
        <div class="rc-title">${m.name.split("(")[0].trim()}</div>
        <div class="rc-desc">${m.description.split(".")[0]}</div>`;
      label.addEventListener("click", () => {
        document.querySelectorAll("#cnn-model-cards .radio-card").forEach(c => c.classList.remove("active"));
        label.classList.add("active");
        label.querySelector("input").checked = true;
        selectedCnnId = m.id;
      });
      container.appendChild(label);
    });
    if (cnnModels.length > 0) selectedCnnId = cnnModels[0].id;
  } catch { showError("Failed to load CNN model list."); }
}

// Drag-and-drop + click to browse
const dropZone   = $("drop-zone");
const imageInput = $("image-input");

dropZone.addEventListener("click", () => imageInput.click());

dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) setImageFile(file);
  else showError("Please drop a valid image file.");
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (file) setImageFile(file);
});

$("clear-image-btn").addEventListener("click", () => {
  selectedImageFile = null;
  imageInput.value  = "";
  hide($("image-preview-wrap"));
  show(dropZone);
});

function setImageFile(file) {
  selectedImageFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    $("image-preview").src = e.target.result;
    hide(dropZone);
    show($("image-preview-wrap"));
  };
  reader.readAsDataURL(file);
}

// Sliders
["n-samples-image", "num-features-image"].forEach(id => {
  const slider = $(id);
  const valEl  = $(`${id}-val`);
  slider.addEventListener("input", () => { valEl.textContent = slider.value; });
});

// Explain image
const explainImgBtn = $("explain-image-btn");
explainImgBtn.addEventListener("click", async () => {
  if (!selectedImageFile) { showError("Please upload an image first."); return; }

  const spinner    = $("image-spinner");
  const nSamples   = parseInt($("n-samples-image").value, 10);
  const numFeatures = parseInt($("num-features-image").value, 10);

  setLoading(spinner, explainImgBtn, true);
  hide($("image-results")); show($("image-placeholder"));
  $("image-placeholder").querySelector("p").textContent =
    `Running LIME with ${nSamples} samples — this takes ~20-40 seconds…`;

  try {
    const formData = new FormData();
    formData.append("image",        selectedImageFile);
    formData.append("model_type",   selectedCnnId);
    formData.append("n_samples",    nSamples);
    formData.append("num_features", numFeatures);

    const res  = await fetch(`${API}/api/explain/image`, { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) { showError(data.error); return; }
    renderImageResults(data);
  } catch (err) { showError(`Request failed: ${err.message}`); }
  finally {
    setLoading(spinner, explainImgBtn, false);
    $("image-placeholder").querySelector("p").textContent = "Upload an image and click Explain";
  }
});

function renderImageResults(data) {
  // Prediction badges — show top-5
  const badgeRow = $("image-pred-badges");
  badgeRow.innerHTML = "";
  (data.top5 || []).forEach((item, i) => {
    const badge = document.createElement("div");
    badge.className = `pred-badge ${i === 0 ? "winner" : "loser"}`;
    badge.textContent = `${item.class}  ${(item.prob * 100).toFixed(1)}%`;
    badgeRow.appendChild(badge);
  });

  // 4-panel chart
  $("image-lime-chart").src = `data:image/png;base64,${data.image}`;

  // Top-5 table with probability bars
  const top5 = data.top5 || [];
  const maxProb = top5.length > 0 ? top5[0].prob : 1;
  const rows = top5.map((item, i) => {
    const barWidth = ((item.prob / maxProb) * 100).toFixed(1);
    return `<tr class="${i === 0 ? "top-row" : ""}">
      <td>${i + 1}</td>
      <td>${item.class}</td>
      <td>${(item.prob * 100).toFixed(2)}%</td>
      <td class="prob-bar-cell">
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${barWidth}%"></div></div>
      </td>
    </tr>`;
  }).join("");

  $("image-top5-table").innerHTML = `
    <table class="top5-table">
      <thead><tr><th>#</th><th>Class (ImageNet)</th><th>Confidence</th><th>Bar</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p style="font-size:.75rem;color:var(--muted);margin-top:8px">
      Model: ${data.model_type} — ${top5.length} classes shown
    </p>`;

  hide($("image-placeholder")); show($("image-results"));
}

// ════════════════════════════════════════════════════════
// Shared helpers
// ════════════════════════════════════════════════════════

function buildFeatureTable(featureWeights, captionText) {
  if (!featureWeights || featureWeights.length === 0)
    return "<p style='color:var(--muted);font-size:.85rem'>No feature data available.</p>";
  const rows = featureWeights
    .slice().sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 12)
    .map(([feat, w]) => {
      const cls  = w >= 0 ? "weight-pos" : "weight-neg";
      const sign = w >= 0 ? "+" : "";
      const bar  = buildMiniBar(w, featureWeights);
      return `<tr>
        <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${feat}">${feat}</td>
        <td class="${cls}">${sign}${w.toFixed(4)}</td>
        <td>${bar}</td>
      </tr>`;
    }).join("");
  return `<table class="feature-table">
    <thead><tr><th>Feature / Word</th><th>Weight</th><th>Impact</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <p style="font-size:.75rem;color:var(--muted);margin-top:8px">${captionText}</p>`;
}

function buildMiniBar(w, allFW) {
  const maxAbs = Math.max(...allFW.map(([, v]) => Math.abs(v)), 0.0001);
  const pct    = Math.abs(w) / maxAbs * 100;
  const color  = w >= 0 ? "var(--accent)" : "var(--red)";
  return `<div style="background:var(--surface);border-radius:4px;overflow:hidden;height:10px;min-width:80px;max-width:160px">
    <div style="width:${pct.toFixed(1)}%;height:100%;background:${color};opacity:.8"></div></div>`;
}

// ════════════════════════════════════════════════════════
// Init
// ════════════════════════════════════════════════════════

loadTabularModels();
loadCnnModels();
