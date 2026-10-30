import { useMemo, useState } from "react";
import { Activity, ChevronDown, Download, Play, Plus, RotateCcw, X } from "lucide-react";
import SignalScope from "@/components/SignalScope";
import { MODEL_PRESETS, OFFLINE_MODEL, sampleModel } from "@/lib/modelSampler";
import type { LayerChoice, ModelRun, SampleSettings } from "@/types/eeg";

const ACCENTS = ["#00f48e", "#ffca34", "#00d5e9"];
const DEFAULT_SETTINGS: SampleSettings = {
  prompt: "The brain learns by making predictions about",
  maxNewTokens: 10,
  channels: 6,
  layer: "last",
  seed: 42,
  endpoint: "https://huggingface.co",
};

type Slot = { id: string; modelId: string };

function createRun(slot: Slot, settings: SampleSettings): ModelRun {
  return {
    id: slot.id,
    modelId: slot.modelId,
    resolvedModelId: slot.modelId,
    status: "queued",
    progress: 0,
    stage: "Queued",
    tokens: [],
    channels: [],
    generatedText: "",
    promptLength: 0,
    layerIndex: 0,
    layerCount: 0,
    hiddenSize: 0,
    source: "logits",
    loadMs: 0,
    sampleMs: 0,
    settings,
  };
}

function downloadFile(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function Home() {
  const [slots, setSlots] = useState<Slot[]>([
    { id: "A", modelId: OFFLINE_MODEL },
    { id: "B", modelId: "Xenova/distilgpt2" },
  ]);
  const [settings, setSettings] = useState<SampleSettings>(DEFAULT_SETTINGS);
  const [runs, setRuns] = useState<Record<string, ModelRun>>({});
  const [running, setRunning] = useState(false);
  const [activeChannel, setActiveChannel] = useState<number | "all">("all");
  const [alignMode, setAlignMode] = useState<"token" | "normalized">("normalized");

  const completedRuns = slots.map((slot) => runs[slot.id]).filter((run) => run?.status === "complete");
  const maxTokens = Math.max(1, ...completedRuns.map((run) => run.tokens.length));
  const sharedScale = useMemo(() => {
    const values = completedRuns.flatMap((run) =>
      run.channels.filter((channel) => activeChannel === "all" || channel.id === activeChannel).flatMap((channel) => channel.values),
    );
    return Math.max(0.05, ...values.map(Math.abs));
  }, [completedRuns, activeChannel]);

  function updateSetting<K extends keyof SampleSettings>(key: K, value: SampleSettings[K]) {
    setSettings((current) => ({ ...current, [key]: value }));
    if (key === "channels") setActiveChannel("all");
  }

  function updateModel(id: string, modelId: string) {
    setSlots((current) => current.map((slot) => slot.id === id ? { ...slot, modelId } : slot));
    setRuns((current) => {
      const next = { ...current };
      delete next[id];
      return next;
    });
  }

  function addModel() {
    if (slots.length >= 3) return;
    const id = ["A", "B", "C"].find((candidate) => !slots.some((slot) => slot.id === candidate));
    if (!id) return;
    setSlots((current) => [...current, { id, modelId: "onnx-community/SmolLM-135M-ONNX" }]);
  }

  function removeModel(id: string) {
    if (slots.length === 1) return;
    setSlots((current) => current.filter((slot) => slot.id !== id));
    setRuns((current) => {
      const next = { ...current };
      delete next[id];
      return next;
    });
  }

  async function startAcquisition() {
    if (running || !settings.prompt.trim()) return;
    setRunning(true);
    const snapshot = { ...settings, prompt: settings.prompt.trim() };
    setRuns(Object.fromEntries(slots.map((slot) => [slot.id, createRun(slot, snapshot)])));
    for (const slot of slots) {
      if (!slot.modelId.trim()) {
        setRuns((current) => ({
          ...current,
          [slot.id]: { ...createRun(slot, snapshot), status: "error", stage: "Acquisition failed", error: "Enter a Hugging Face model ID." },
        }));
        continue;
      }
      try {
        await sampleModel(slot.id, slot.modelId.trim(), snapshot, (patch) => {
          setRuns((current) => ({ ...current, [slot.id]: { ...current[slot.id], ...patch } as ModelRun }));
        });
      } catch {
        // The error state is rendered in the affected channel; other models still run.
      }
    }
    setRunning(false);
  }

  function resetRuns() {
    setRuns({});
    setRunning(false);
  }

  function exportJSON() {
    downloadFile("virtual-eeg-runs.json", JSON.stringify(completedRuns, (_, value) => typeof value === "bigint" ? Number(value) : value, 2), "application/json");
  }

  function exportCSV() {
    const rows = ["model_id,token_index,token,phase,channel,value_au"];
    for (const run of completedRuns) {
      for (const channel of run.channels) {
        channel.values.forEach((value, index) => {
          const token = run.tokens[index];
          rows.push([
            JSON.stringify(run.modelId),
            token.index,
            JSON.stringify(token.text),
            token.phase,
            channel.label.replace(" ", ""),
            value.toFixed(8),
          ].join(","));
        });
      }
    }
    downloadFile("virtual-eeg-samples.csv", rows.join("\n"), "text/csv;charset=utf-8");
  }

  return (
    <div className="lab-shell">
      <div className="scanlines" aria-hidden="true" />
      <header className="topbar">
        <a className="brand" href="#workbench" aria-label="Virtual EEG home">
          <Activity size={22} strokeWidth={1.6} aria-hidden="true" />
          <span>VIRTUAL—EEG</span>
          <small>ACTIVATION SCOPE</small>
        </a>
        <div className="topbar-readout">
          <span><i /> BROWSER / LOCAL</span>
          <span>ONNX RUNTIME</span>
          <span className="hide-small">NO CLOUD INFERENCE</span>
        </div>
      </header>

      <main id="workbench" className="workbench">
        <div className="display-column">
          <div className="display-toolbar">
            <div>
              <span className="section-index">01 / SIGNAL MONITOR</span>
              <h1>Model activity, made visible.</h1>
            </div>
            <div className="display-actions">
              <button type="button" onClick={exportCSV} disabled={!completedRuns.length}><Download size={15} />CSV</button>
              <button type="button" onClick={exportJSON} disabled={!completedRuns.length}><Download size={15} />JSON</button>
            </div>
          </div>

          <div className="channel-controls" aria-label="Display controls">
            <div className="channel-tabs">
              <button type="button" className={activeChannel === "all" ? "active" : ""} onClick={() => setActiveChannel("all")}>ALL CH</button>
              {Array.from({ length: settings.channels }, (_, index) => (
                <button key={index} type="button" className={activeChannel === index ? "active" : ""} onClick={() => setActiveChannel(index)}>
                  {String(index + 1).padStart(2, "0")}
                </button>
              ))}
            </div>
            <div className="alignment-toggle">
              <button type="button" className={alignMode === "normalized" ? "active" : ""} onClick={() => setAlignMode("normalized")}>NORMALIZED</button>
              <button type="button" className={alignMode === "token" ? "active" : ""} onClick={() => setAlignMode("token")}>TOKEN ALIGN</button>
            </div>
          </div>

          <div className="scope-frame">
            <div className="scope-frame-inner">
              {slots.map((slot, index) => (
                <SignalScope
                  key={slot.id}
                  run={runs[slot.id]}
                  modelId={slot.modelId || `MODEL ${slot.id}`}
                  accent={ACCENTS[index]}
                  activeChannel={activeChannel}
                  alignMode={alignMode}
                  sharedScale={sharedScale}
                  maxTokens={maxTokens}
                />
              ))}
            </div>
          </div>

          <section className="comparison-readout" aria-label="Run comparison">
            <div className="comparison-title"><span>02 / COMPARISON BUS</span><strong>{completedRuns.length} ACQUIRED</strong></div>
            {completedRuns.length ? completedRuns.map((run) => {
              const rms = run.channels.reduce((sum, channel) => sum + channel.stats.rms, 0) / run.channels.length;
              const delta = run.channels.reduce((sum, channel) => sum + channel.stats.meanDelta, 0) / run.channels.length;
              return (
                <div className="comparison-row" key={run.id}>
                  <span className="comparison-model"><i style={{ background: ACCENTS[run.id.charCodeAt(0) - 65] }} />{run.modelId}</span>
                  <span>RMS <strong>{rms.toFixed(3)}</strong></span>
                  <span>Δ/TOKEN <strong>{delta.toFixed(3)}</strong></span>
                  <span>LOAD <strong>{(run.loadMs / 1000).toFixed(1)}s</strong></span>
                </div>
              );
            }) : <p className="comparison-empty">Run at least one model to populate comparative measurements.</p>}
          </section>

          <details className="method-panel">
            <summary>ACQUISITION METHOD & LIMITS <ChevronDown size={16} /></summary>
            <div>
              <p><strong>What is measured.</strong> For every prompt and generated token, an explicit forward pass records the selected transformer layer’s hidden-state vector when the ONNX export provides it. Otherwise, the sampler records the next-token logit vector and labels the run LOGITS. Each vector is token-wise normalized and projected through a fixed, seeded Gaussian matrix into synthetic channels. Prompt-mean baseline removal centers each trace.</p>
              <p><strong>What it means.</strong> The traces are a deterministic, low-dimensional view of model activations in arbitrary units—not voltage, physiological EEG, or a literal account of reasoning. Different tokenizers and architectures can produce different sequence lengths. Greedy decoding makes repeated runs comparable.</p>
              <p><strong>Compatibility.</strong> Use small causal language models with ONNX weights and hidden-state outputs. Quantized downloads use q4 when available. Model files and inference stay in this browser; cache clearing removes them. The offline fixture has random weights and is for pipeline verification only.</p>
            </div>
          </details>
        </div>

        <aside className="control-panel">
          <div className="panel-heading"><span>CONTROL / 01</span><strong>ACQUISITION</strong></div>

          <section className="control-section model-section">
            <div className="control-label"><span>MODEL INPUTS</span><small>HF / ONNX</small></div>
            {slots.map((slot, index) => (
              <div className="model-input" key={slot.id}>
                <span className="slot-id" style={{ color: ACCENTS[index] }}>{slot.id}</span>
                <input
                  aria-label={`Model ${slot.id} Hugging Face ID`}
                  list="model-presets"
                  value={slot.modelId}
                  onChange={(event) => updateModel(slot.id, event.target.value)}
                  disabled={running}
                  spellCheck={false}
                />
                <button type="button" onClick={() => removeModel(slot.id)} disabled={running || slots.length === 1} aria-label={`Remove model ${slot.id}`}><X size={15} /></button>
              </div>
            ))}
            <datalist id="model-presets">
              {MODEL_PRESETS.map((preset) => <option key={preset.id} value={preset.id}>{preset.label} — {preset.note}</option>)}
            </datalist>
            <button type="button" className="text-button" onClick={addModel} disabled={running || slots.length >= 3}><Plus size={15} />ADD MODEL <span>{slots.length}/3</span></button>
          </section>

          <section className="control-section">
            <label className="control-label" htmlFor="prompt"><span>PROMPT SOURCE</span><small>{settings.prompt.length}/400</small></label>
            <textarea
              id="prompt"
              value={settings.prompt}
              onChange={(event) => updateSetting("prompt", event.target.value.slice(0, 400))}
              disabled={running}
              rows={4}
            />
          </section>

          <section className="control-section parameter-grid">
            <label>
              <span>NEW TOKENS</span>
              <select value={settings.maxNewTokens} onChange={(event) => updateSetting("maxNewTokens", Number(event.target.value))} disabled={running}>
                {[4, 8, 10, 12, 16, 20, 24].map((value) => <option key={value} value={value}>{value}</option>)}
              </select>
            </label>
            <label>
              <span>CHANNELS</span>
              <select value={settings.channels} onChange={(event) => updateSetting("channels", Number(event.target.value))} disabled={running}>
                {[4, 6, 8, 12].map((value) => <option key={value} value={value}>{value}</option>)}
              </select>
            </label>
            <label>
              <span>LAYER TAP</span>
              <select value={settings.layer} onChange={(event) => updateSetting("layer", event.target.value as LayerChoice)} disabled={running}>
                <option value="last">Final</option>
                <option value="middle">Middle</option>
                <option value="embedding">Embedding</option>
              </select>
            </label>
            <label>
              <span>PROJECTION SEED</span>
              <input type="number" min={0} max={9999} value={settings.seed} onChange={(event) => updateSetting("seed", Number(event.target.value))} disabled={running} />
            </label>
          </section>

          <details className="runtime-settings">
            <summary>RUNTIME ENDPOINT <ChevronDown size={15} /></summary>
            <label htmlFor="endpoint">Hugging Face / mirror URL</label>
            <input id="endpoint" value={settings.endpoint} onChange={(event) => updateSetting("endpoint", event.target.value)} disabled={running} spellCheck={false} />
          </details>

          <div className="acquisition-actions">
            <button type="button" className="run-button" onClick={startAcquisition} disabled={running || !settings.prompt.trim()}>
              <Play size={17} fill="currentColor" />{running ? "ACQUIRING…" : "START ACQUISITION"}
            </button>
            <button type="button" className="reset-button" onClick={resetRuns} disabled={running || !Object.keys(runs).length}><RotateCcw size={16} />RESET</button>
          </div>
          <p className="panel-note">Small models only. First download may take several minutes; subsequent runs reuse the browser cache.</p>
        </aside>
      </main>

      <footer className="footer-bar">
        <span>SYNTHETIC MODEL TELEMETRY · NOT BIOLOGICAL EEG</span>
        <span>V—EEG / LAB 01</span>
      </footer>
    </div>
  );
}
