# Virtual EEG Lab

A small browser-based workbench for comparing EEG-like activation traces from Hugging Face causal language models. It uses Transformers.js and ONNX Runtime Web; inference and model caching happen locally in the browser.

## Run

```bash
npm install
npm run fetch:offline  # Optional: download the offline verification fixture
npm run dev
```

Open `http://localhost:3000`. Online Hugging Face models do not require the fixture download. If Hugging Face is unreachable, set `HF_ENDPOINT` to a compatible mirror before running `fetch:offline`, or configure the mirror in the app’s Runtime endpoint panel.

```bash
npm run check   # TypeScript
npm run build   # Production frontend + server bundle
npm start       # Serve the production build
```

If a restricted environment fails on optional native ONNX install scripts, `npm install --ignore-scripts` is sufficient for this browser/WASM app.

## Use

1. Enter one to three Hugging Face model IDs, or choose a preset.
2. Set a prompt, generated-token count, channel count, layer tap, and projection seed.
3. Select **Start acquisition**. Models run sequentially to limit memory use.
4. Compare stacked traces, isolate channels, switch alignment, and export CSV or JSON.

Recommended starting points:

- `offline:tiny-gpt2` — random-weight GPT2 fixture for verifying the full pipeline without further network access; run `npm run fetch:offline` once after cloning.
- `Xenova/distilgpt2` — small DistilGPT2 ONNX checkpoint.
- `onnx-community/SmolLM-135M-ONNX` — small SmolLM ONNX checkpoint.

Online checkpoints must be causal LMs with Transformers.js-compatible ONNX weights. Online loads request q4 weights; the offline fixture uses fp32. The first WASM/model initialization can take several minutes. Later runs reuse the in-memory model and browser cache. A Hugging Face-compatible mirror can be set under **Runtime endpoint**.

## Sampling method

The sampler performs explicit greedy decoding. At each token step it runs a full forward pass and records:

- the selected layer's hidden-state vector when the ONNX export provides hidden states; otherwise
- the next-token logit vector, with the run explicitly labeled `LOGITS`.

Vectors are token-wise normalized, projected through a deterministic seeded Gaussian matrix, and baseline-centered against the prompt. The resulting channels use arbitrary units and a token-step axis. Token alignment uses the longest acquired sequence; normalized alignment stretches each sequence to the same display width.

These traces are synthetic model telemetry, not biological EEG, voltage, or a direct visualization of reasoning. Different tokenizers can produce different token counts. Full forward-pass decoding favors inspectability over speed and is intended for small models and short generations.

## Exports

- CSV: one row per model × token × channel.
- JSON: complete run settings, tokens, channel samples, source, dimensions, timing, and summary statistics.

No run history is stored on a server. Downloaded model files remain only in the browser cache until it is cleared.

## Main files

- `src/lib/modelSampler.ts` — model loading, greedy decoding, activation capture, projection.
- `src/components/SignalScope.tsx` — multichannel trace rendering and run readouts.
- `src/pages/Home.tsx` — acquisition controls, comparison, and exports.
- `scripts/fetch-offline-model.mjs` — downloads the complete offline verification fixture into `public/models/tiny-random-gpt2/`.
