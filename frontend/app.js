/**
 * LoCo Arena — Frontend Application
 *
 * Implements side-by-side A/B streaming arena between DeepSeek-R1-Distill-Qwen-7B
 * (Baseline) and LoCo-7B using Server-Sent Events (SSE).
 * Strictly conforms to CONTRACT.md.
 */

// 1. Configuration & API Base Detection
const params = new URLSearchParams(window.location.search);
const API_BASE = params.get("api") || window.LOCO_API_BASE || "/api";

// DOM Elements
const elModeBadge = document.getElementById("mode-badge");
const elModeText = document.getElementById("mode-text");
const elBaselineModelName = document.getElementById("baseline-model-name");
const elLocoModelName = document.getElementById("loco-model-name");

const elPresets = document.getElementById("presets");
const elPresetHint = document.getElementById("preset-hint");
const elPrompt = document.getElementById("prompt");
const elMaxTokens = document.getElementById("max-tokens");
const elCharCount = document.getElementById("char-count");
const elRunBtn = document.getElementById("run-btn");
const elStopBtn = document.getElementById("stop-btn");
const elGlobalError = document.getElementById("global-error");
const elSummary = document.getElementById("summary");
const elWinnerValue = document.getElementById("winner-value");

// Panels
const panels = {
  baseline: createPanelController("baseline", document.getElementById("panel-baseline")),
  loco: createPanelController("loco", document.getElementById("panel-loco")),
};

let activeEventSource = null;
let arenaStartTime = 0;
let rafId = null;
let locoFinishedAt = null;
let locoTokensAtFinish = 0;

/** Escape HTML characters safely */
function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/** Render text with \boxed{...} replaced by formatted badge */
function formatContent(text) {
  const escaped = escapeHtml(text);
  // Match \boxed{...} including optional LaTeX wrappers
  return escaped.replace(/\\boxed\{([^{}]+)\}/g, (_, content) => {
    // Strip \text{...} wrapper if present
    const clean = content.replace(/^\\text\{([^{}]+)\}$/, "$1");
    return `<span class="boxed">${clean}</span>`;
  });
}

/** Panel state & DOM controller */
function createPanelController(role, rootEl) {
  const els = {
    root: rootEl,
    status: rootEl.querySelector('[data-field="status"]'),
    modelName: rootEl.querySelector('[data-field="model-name"]'),
    stopwatch: rootEl.querySelector('[data-field="stopwatch"]'),
    ttft: rootEl.querySelector('[data-field="ttft"]'),
    tps: rootEl.querySelector('[data-field="tps"]'),
    thinkTokens: rootEl.querySelector('[data-field="think-tokens"]'),
    answerTokens: rootEl.querySelector('[data-field="answer-tokens"]'),
    totalTokens: rootEl.querySelector('[data-field="total-tokens"]'),
    barThink: rootEl.querySelector('[data-field="bar-think"]'),
    barAnswer: rootEl.querySelector('[data-field="bar-answer"]'),
    thinkText: rootEl.querySelector('[data-field="think-text"]'),
    answerText: rootEl.querySelector('[data-field="answer-text"]'),
    error: rootEl.querySelector('[data-field="error"]'),
    // Race elements
    finishedBanner: rootEl.querySelector(".finished-banner"),
    finishedTime: rootEl.querySelector('[data-field="finished-time"]'),
    deliberatingBanner: rootEl.querySelector(".deliberating-banner"),
    extraTokens: rootEl.querySelector('[data-field="extra-tokens"]'),
    extraTime: rootEl.querySelector('[data-field="extra-time"]'),
  };

  let state = {
    status: "idle",
    thinkBuf: "",
    answerBuf: "",
    thinkTokens: 0,
    answerTokens: 0,
    totalTokens: 0,
    finishedAtMs: null,
    metrics: null,
  };

  function setStatus(s) {
    state.status = s;
    els.root.setAttribute("data-status", s);
    if (els.status) els.status.textContent = s;
  }

  function reset() {
    state = {
      status: "idle",
      thinkBuf: "",
      answerBuf: "",
      thinkTokens: 0,
      answerTokens: 0,
      totalTokens: 0,
      finishedAtMs: null,
      metrics: null,
    };
    setStatus("idle");
    els.root.classList.remove("is-winner", "is-deliberating", "is-late");
    if (els.stopwatch) els.stopwatch.innerHTML = '0.000<span class="unit">s</span>';
    if (els.ttft) els.ttft.textContent = "—";
    if (els.tps) els.tps.textContent = "—";
    if (els.thinkTokens) els.thinkTokens.textContent = "0";
    if (els.answerTokens) els.answerTokens.textContent = "0";
    if (els.totalTokens) els.totalTokens.textContent = "0 tok";
    if (els.barThink) els.barThink.style.width = "0%";
    if (els.barAnswer) els.barAnswer.style.width = "0%";
    if (els.thinkText) els.thinkText.innerHTML = "";
    if (els.answerText) els.answerText.innerHTML = "";
    if (els.error) {
      els.error.textContent = "";
      els.error.classList.remove("show");
    }
  }

  function appendText(phase, text) {
    if (phase === "think") {
      state.thinkBuf += text;
      if (els.thinkText) {
        els.thinkText.innerHTML = formatContent(state.thinkBuf);
        els.thinkText.scrollTop = els.thinkText.scrollHeight;
      }
    } else {
      state.answerBuf += text;
      if (els.answerText) {
        els.answerText.innerHTML = formatContent(state.answerBuf);
        els.answerText.scrollTop = els.answerText.scrollHeight;
      }
    }
  }

  function updateMetrics(metrics) {
    state.metrics = metrics;
    state.thinkTokens = metrics.think_tokens;
    state.answerTokens = metrics.answer_tokens;
    state.totalTokens = metrics.total_tokens;

    if (metrics.ttft_ms != null && els.ttft) {
      els.ttft.textContent = `${Math.round(metrics.ttft_ms)}ms`;
    }
    if (metrics.tokens_per_sec > 0 && els.tps) {
      els.tps.textContent = `${metrics.tokens_per_sec.toFixed(1)}`;
    }
    if (els.thinkTokens) els.thinkTokens.textContent = String(metrics.think_tokens);
    if (els.answerTokens) els.answerTokens.textContent = String(metrics.answer_tokens);
    if (els.totalTokens) els.totalTokens.textContent = `${metrics.total_tokens} tok`;
  }

  function showError(msg) {
    setStatus("error");
    if (els.error) {
      els.error.textContent = msg;
      els.error.classList.add("show");
    }
  }

  return {
    role,
    els,
    getState: () => state,
    setStatus,
    reset,
    appendText,
    updateMetrics,
    showError,
  };
}

/** Update stacked progress bars scaled against the max total tokens */
function updateProgressBars() {
  const bTokens = panels.baseline.getState().totalTokens;
  const lTokens = panels.loco.getState().totalTokens;
  const maxTotal = Math.max(1, bTokens, lTokens);

  for (const role of ["baseline", "loco"]) {
    const p = panels[role];
    const s = p.getState();
    const thinkPct = (s.thinkTokens / maxTotal) * 100;
    const answerPct = (s.answerTokens / maxTotal) * 100;

    if (p.els.barThink) p.els.barThink.style.width = `${thinkPct}%`;
    if (p.els.barAnswer) p.els.barAnswer.style.width = `${answerPct}%`;
  }
}

/** Stopwatch and live race animation loop */
function startLiveLoop() {
  function loop() {
    const now = performance.now();
    const elapsedSec = (now - arenaStartTime) / 1000;

    for (const role of ["baseline", "loco"]) {
      const p = panels[role];
      const s = p.getState();
      if (s.status === "thinking" || s.status === "answering" || s.status === "connecting") {
        if (p.els.stopwatch) {
          p.els.stopwatch.innerHTML = `${elapsedSec.toFixed(3)}<span class="unit">s</span>`;
        }
      }
    }

    // Baseline live overthinking deliberation counters if LoCo already finished
    if (locoFinishedAt !== null) {
      const bState = panels.baseline.getState();
      if (bState.status === "thinking" || bState.status === "answering") {
        const extraSec = (now - locoFinishedAt) / 1000;
        const extraTok = Math.max(0, bState.totalTokens - locoTokensAtFinish);

        if (panels.baseline.els.extraTime) {
          panels.baseline.els.extraTime.textContent = `+${extraSec.toFixed(1)}s`;
        }
        if (panels.baseline.els.extraTokens) {
          panels.baseline.els.extraTokens.textContent = `+${extraTok}`;
        }
      }
    }

    rafId = requestAnimationFrame(loop);
  }
  rafId = requestAnimationFrame(loop);
}

function stopLiveLoop() {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
}

/** Fetch Health & Presets */
async function init() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      elModeBadge.setAttribute("data-mode", data.mode);
      elModeText.textContent = data.mode;
      if (elBaselineModelName) elBaselineModelName.textContent = data.baseline_model;
      if (elLocoModelName) elLocoModelName.textContent = data.loco_model;
      if (panels.baseline.els.modelName) panels.baseline.els.modelName.textContent = data.baseline_model.split("/").pop();
      if (panels.loco.els.modelName) panels.loco.els.modelName.textContent = data.loco_model.split("/").pop();
    } else {
      elModeBadge.setAttribute("data-mode", "offline");
      elModeText.textContent = "offline";
    }
  } catch (err) {
    elModeBadge.setAttribute("data-mode", "offline");
    elModeText.textContent = "offline";
  }

  try {
    const res = await fetch(`${API_BASE}/presets`);
    if (res.ok) {
      const presets = await res.json();
      renderPresets(presets);
    }
  } catch (err) {
    elPresets.innerHTML = '<span class="text-xs text-rose-400">Failed to load presets</span>';
  }
}

/** Render preset chips */
function renderPresets(presets) {
  elPresets.innerHTML = "";
  presets.forEach((p, idx) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.setAttribute("role", "listitem");
    chip.setAttribute("aria-pressed", "false");
    chip.innerHTML = `<span class="cat" data-cat="${p.category}">${p.category}</span> <span>${escapeHtml(p.title)}</span>`;

    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
      chip.setAttribute("aria-pressed", "true");
      elPrompt.value = p.prompt;
      updateCharCount();
      elPresetHint.textContent = `💡 Overthinking reason: ${p.why_overthinks}`;
      elPresetHint.classList.remove("hidden");
      autoResizeTextarea();
    });

    elPresets.appendChild(chip);

    // Auto-select first preset
    if (idx === 0) {
      chip.click();
    }
  });
}

/** Auto-resize prompt textarea */
function autoResizeTextarea() {
  elPrompt.style.height = "auto";
  elPrompt.style.height = `${Math.min(320, elPrompt.scrollHeight)}px`;
}

function updateCharCount() {
  const len = elPrompt.value.length;
  elCharCount.textContent = `${len} / 8000`;
  elCharCount.classList.toggle("text-rose-400", len > 7900);
}

elPrompt.addEventListener("input", () => {
  updateCharCount();
  autoResizeTextarea();
});

// Shortcut: Ctrl/Cmd + Enter to run
window.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    if (!elRunBtn.disabled) runArena();
  }
});

elRunBtn.addEventListener("click", runArena);
elStopBtn.addEventListener("click", stopArena);

/** Run Arena Stream */
function runArena() {
  const prompt = elPrompt.value.trim();
  if (!prompt) return;

  const maxTokens = parseInt(elMaxTokens.value, 10) || 4096;

  // Reset UI
  elGlobalError.classList.add("hidden");
  elGlobalError.textContent = "";
  elSummary.classList.add("hidden");
  elSummary.classList.remove("reveal");
  document.body.classList.add("arena-race");

  panels.baseline.reset();
  panels.loco.reset();
  panels.baseline.setStatus("connecting");
  panels.loco.setStatus("connecting");

  elRunBtn.disabled = true;
  elStopBtn.disabled = false;

  locoFinishedAt = null;
  locoTokensAtFinish = 0;
  arenaStartTime = performance.now();
  startLiveLoop();

  const url = `${API_BASE}/arena/stream?prompt=${encodeURIComponent(prompt)}&max_tokens=${maxTokens}`;
  activeEventSource = new EventSource(url);

  activeEventSource.onmessage = (event) => {
    if (!event.data) return;
    try {
      const data = JSON.parse(event.data);
      handleArenaEvent(data);
    } catch (err) {
      console.error("Failed to parse SSE event:", err, event.data);
    }
  };

  activeEventSource.onerror = (err) => {
    console.error("SSE connection error:", err);
    // If stream ended abruptly
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }
    stopLiveLoop();
    elRunBtn.disabled = false;
    elStopBtn.disabled = true;
  };
}

/** Process SSE Events conforming to CONTRACT.md */
function handleArenaEvent(evt) {
  if (!evt || !evt.type) return;

  if (evt.type === "start") {
    const p = panels[evt.model];
    if (p) {
      p.setStatus("thinking");
      if (evt.model_name && p.els.modelName) {
        p.els.modelName.textContent = evt.model_name.split("/").pop();
      }
    }
    return;
  }

  if (evt.type === "token") {
    const p = panels[evt.model];
    if (p) {
      p.setStatus(evt.phase === "think" ? "thinking" : "answering");
      p.appendText(evt.phase, evt.text);
      if (evt.metrics) p.updateMetrics(evt.metrics);
      updateProgressBars();
    }
    return;
  }

  if (evt.type === "phase") {
    const p = panels[evt.model];
    if (p) {
      p.setStatus(evt.phase);
      if (evt.metrics) p.updateMetrics(evt.metrics);
      updateProgressBars();
    }
    return;
  }

  if (evt.type === "done") {
    const p = panels[evt.model];
    if (p) {
      p.setStatus("done");
      if (evt.metrics) {
        p.updateMetrics(evt.metrics);
        if (p.els.stopwatch) {
          p.els.stopwatch.innerHTML = `${(evt.metrics.elapsed_ms / 1000).toFixed(3)}<span class="unit">s</span>`;
        }
      }

      // Check if LoCo finished first to trigger dramatic deliberating banner
      if (evt.model === "loco") {
        locoFinishedAt = performance.now();
        locoTokensAtFinish = p.getState().totalTokens;
        p.els.root.classList.add("is-winner");
        if (p.els.finishedTime && evt.metrics) {
          p.els.finishedTime.textContent = `${(evt.metrics.elapsed_ms / 1000).toFixed(2)}s`;
        }

        const bState = panels.baseline.getState();
        if (bState.status === "thinking" || bState.status === "answering") {
          panels.baseline.els.root.classList.add("is-deliberating");
        }
      } else if (evt.model === "baseline") {
        if (locoFinishedAt !== null) {
          panels.baseline.els.root.classList.remove("is-deliberating");
          panels.baseline.els.root.classList.add("is-late");
        }
      }
      updateProgressBars();
    }
    return;
  }

  if (evt.type === "error") {
    const p = panels[evt.model];
    if (p) {
      p.showError(evt.message || "Unknown error occurred");
    }
    return;
  }

  if (evt.type === "arena_done") {
    // Stream fully finished
    if (activeEventSource) {
      activeEventSource.close();
      activeEventSource = null;
    }
    stopLiveLoop();
    elRunBtn.disabled = false;
    elStopBtn.disabled = true;
    document.body.classList.remove("arena-race");

    renderSummary(evt.summary);
  }
}

/** Stop stream manually */
function stopArena() {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  stopLiveLoop();
  elRunBtn.disabled = false;
  elStopBtn.disabled = true;
  document.body.classList.remove("arena-race");

  for (const role of ["baseline", "loco"]) {
    const p = panels[role];
    const s = p.getState();
    if (s.status === "thinking" || s.status === "answering" || s.status === "connecting") {
      p.setStatus("stopped");
    }
  }
}

/** Animate summary stat numbers on arena_done */
function renderSummary(summary) {
  if (!summary) return;

  elSummary.classList.remove("hidden");
  elSummary.classList.add("reveal");

  function animateVal(attr, targetVal) {
    const el = elSummary.querySelector(`[data-count="${attr}"]`);
    if (!el) return;
    if (targetVal == null || isNaN(targetVal)) {
      el.textContent = "—";
      return;
    }
    const sign = targetVal > 0 ? "-" : "+"; // savings means fewer tokens
    const absVal = Math.abs(targetVal);
    let start = 0;
    const duration = 600;
    const startTime = performance.now();

    function step(now) {
      const progress = Math.min(1, (now - startTime) / duration);
      const cur = (start + (absVal - start) * progress).toFixed(1);
      el.textContent = `${targetVal >= 0 ? "" : "-"}${cur}`;
      if (progress < 1) {
        requestAnimationFrame(step);
      }
    }
    requestAnimationFrame(step);
  }

  animateVal("think_token_savings_pct", summary.think_token_savings_pct);
  animateVal("total_token_savings_pct", summary.total_token_savings_pct);
  animateVal("time_savings_pct", summary.time_savings_pct);

  if (elWinnerValue) {
    elWinnerValue.setAttribute("data-winner", summary.winner || "none");
    if (summary.winner === "loco") {
      elWinnerValue.textContent = "LoCo";
    } else if (summary.winner === "baseline") {
      elWinnerValue.textContent = "Baseline";
    } else if (summary.winner === "tie") {
      elWinnerValue.textContent = "Tie";
    } else {
      elWinnerValue.textContent = "—";
    }
  }
}

// Start initialization on load
init();
