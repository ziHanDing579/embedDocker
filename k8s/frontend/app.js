// app.js
// Capture sentences, embed them, and plot each one by its cosine similarity to
// two sentences you choose: one for the X axis, one for the Y axis.

import { CONFIG } from "./config.js";
import { getEmbedding } from "./embeddings.js";
import { cosineSimilarity, clamp01 } from "./linalg.js";
import { Plot } from "./chart.js";

const state = {
  sentences: [], // { id, text, embedding }
  xAxisId: null, // id of the sentence defining the X axis
  yAxisId: null, // id of the sentence defining the Y axis
  points: [], // { id, text, x, y, isX, isY }
};

let seq = 0;
const nextId = () => `s${++seq}`;

const els = {
  form: document.getElementById("capture-form"),
  input: document.getElementById("sentence-input"),
  submit: document.getElementById("submit-btn"),
  list: document.getElementById("sentence-list"),
  count: document.getElementById("count"),
  mode: document.getElementById("mode"),
  xText: document.getElementById("x-text"),
  yText: document.getElementById("y-text"),
  status: document.getElementById("status"),
  svg: document.getElementById("plot"),
};

const plot = new Plot(els.svg);
els.mode.textContent = CONFIG.USE_MOCK ? "MOCK" : "LIVE";

const find = (id) => state.sentences.find((s) => s.id === id) || null;
const embOf = (id) => (find(id) ? find(id).embedding : null);
const textOf = (id) => (find(id) ? find(id).text : null);

// ---- Axis selection -------------------------------------------------------

function setXAxis(id) {
  state.xAxisId = id;
  recomputeAndRender();
  setStatus(`X axis set to “${truncate(textOf(id), 40)}”`);
}

function setYAxis(id) {
  state.yAxisId = id;
  recomputeAndRender();
  setStatus(`Y axis set to “${truncate(textOf(id), 40)}”`);
}

// ---- Projection -----------------------------------------------------------

function recomputeAndRender() {
  const xVec = embOf(state.xAxisId);
  const yVec = embOf(state.yAxisId);
  const ready = Boolean(xVec && yVec);

  state.points = ready
    ? state.sentences.map((s) => ({
        id: s.id,
        text: s.text,
        x: clamp01(cosineSimilarity(s.embedding, xVec)),
        y: clamp01(cosineSimilarity(s.embedding, yVec)),
        isX: s.id === state.xAxisId,
        isY: s.id === state.yAxisId,
      }))
    : [];

  plot.setAxisTitles(textOf(state.xAxisId), textOf(state.yAxisId));
  plot.render(state.points);
  renderList();
  renderReadout();
}

async function addSentence(text) {
  setBusy(true, "Embedding…");
  try {
    const embedding = await getEmbedding(text);
    const entry = { id: nextId(), text, embedding };
    state.sentences.push(entry);

    // Sensible defaults so there's an immediate plot: 1st sentence -> X, 2nd -> Y.
    // Everything is reassignable from the sidebar afterward.
    if (state.xAxisId === null) state.xAxisId = entry.id;
    else if (state.yAxisId === null) state.yAxisId = entry.id;

    recomputeAndRender();

    if (state.xAxisId && state.yAxisId) {
      setStatus(`Plotted “${truncate(text, 40)}”`);
    } else if (!state.yAxisId) {
      setStatus(`Added “${truncate(text, 28)}”. Now choose a Y sentence.`);
    } else {
      setStatus(`Added “${truncate(text, 28)}”. Now choose an X sentence.`);
    }
  } catch (err) {
    console.error(err);
    setStatus(
      CONFIG.USE_MOCK
        ? "Something went wrong generating the embedding."
        : "Couldn't reach the embedding endpoint. Check EMBEDDING_ENDPOINT in config.js, or set USE_MOCK to true.",
      true
    );
  } finally {
    setBusy(false);
  }
}

// ---- Sidebar + readout ----------------------------------------------------

function renderList() {
  els.count.textContent = String(state.sentences.length);
  els.list.replaceChildren();

  for (const s of state.sentences) {
    const p = state.points.find((pt) => pt.id === s.id);
    const isX = s.id === state.xAxisId;
    const isY = s.id === state.yAxisId;

    const row = document.createElement("div");
    row.className = "row" + (isX ? " is-x" : "") + (isY ? " is-y" : "");

    const txt = document.createElement("span");
    txt.className = "row-text";
    txt.textContent = s.text;

    const coord = document.createElement("span");
    coord.className = "row-coord";
    coord.textContent = p ? `${p.x.toFixed(2)}, ${p.y.toFixed(2)}` : "";

    const actions = document.createElement("div");
    actions.className = "row-actions";

    const bx = document.createElement("button");
    bx.type = "button";
    bx.className = "ax-btn ax-x" + (isX ? " active" : "");
    bx.textContent = "X";
    bx.title = "Use as X axis";
    bx.setAttribute("aria-pressed", String(isX));
    bx.addEventListener("click", () => setXAxis(s.id));

    const by = document.createElement("button");
    by.type = "button";
    by.className = "ax-btn ax-y" + (isY ? " active" : "");
    by.textContent = "Y";
    by.title = "Use as Y axis";
    by.setAttribute("aria-pressed", String(isY));
    by.addEventListener("click", () => setYAxis(s.id));

    actions.append(bx, by);
    row.append(txt, coord, actions);
    els.list.appendChild(row);
  }
}

function renderReadout() {
  const x = textOf(state.xAxisId);
  const y = textOf(state.yAxisId);
  els.xText.textContent = x || "choose a sentence";
  els.yText.textContent = y || "choose a sentence";
  els.xText.classList.toggle("unset", !x);
  els.yText.classList.toggle("unset", !y);
}

// ---- UI plumbing ----------------------------------------------------------

function setBusy(busy, label) {
  els.submit.disabled = busy;
  els.input.disabled = busy;
  els.submit.textContent = busy ? label || "…" : "Plot";
  if (!busy) els.input.focus();
}

function setStatus(msg, isError = false) {
  els.status.textContent = msg;
  els.status.classList.toggle("is-error", isError);
}

function truncate(str, n) {
  if (!str) return "";
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = els.input.value.trim();
  if (!text) return;
  els.input.value = "";
  addSentence(text);
});

// First paint.
recomputeAndRender();
setStatus(
  CONFIG.USE_MOCK
    ? "Running on the local mock. Set USE_MOCK to false in config.js for real embeddings."
    : "Add sentences, then assign one to X and one to Y."
);