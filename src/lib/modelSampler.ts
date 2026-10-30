import { AutoModelForCausalLM, AutoTokenizer, Tensor, env } from "@huggingface/transformers";
import type {
  EEGChannel,
  LayerChoice,
  ModelProgress,
  ModelRun,
  SampleSettings,
  TokenSample,
} from "@/types/eeg";

export const OFFLINE_MODEL = "offline:tiny-gpt2";
export const OFFLINE_MODEL_PATH = "/models/tiny-random-gpt2";

export const MODEL_PRESETS = [
  { id: OFFLINE_MODEL, label: "Tiny GPT2 · offline fixture", note: "Random weights · verification only" },
  { id: "Xenova/distilgpt2", label: "DistilGPT2", note: "82M parameters · ONNX" },
  { id: "onnx-community/SmolLM-135M-ONNX", label: "SmolLM 135M", note: "135M parameters · ONNX" },
];

type ProgressUpdate = (patch: Partial<ModelRun>) => void;
type LoadedModel = { model: any; tokenizer: any };

const modelCache = new Map<string, Promise<LoadedModel>>();

function resolvedId(modelId: string) {
  return modelId.trim() === OFFLINE_MODEL ? OFFLINE_MODEL_PATH : modelId.trim();
}

function cacheKey(modelId: string) {
  return `${resolvedId(modelId)}::${modelId.trim() === OFFLINE_MODEL ? "fp32" : "q4"}`;
}

async function loadModel(modelId: string, endpoint: string, onProgress: ProgressUpdate) {
  const key = cacheKey(modelId);
  const cached = modelCache.get(key);
  if (cached) {
    onProgress({ stage: "Using cached model", progress: 45 });
    return cached;
  }

  const promise = (async (): Promise<LoadedModel> => {
    const id = resolvedId(modelId);
    const isOffline = modelId.trim() === OFFLINE_MODEL;
    env.allowLocalModels = true;
    env.allowRemoteModels = true;
    env.useBrowserCache = true;
    if (!isOffline) env.remoteHost = endpoint.replace(/\/$/, "");

    onProgress({ status: "loading", stage: "Loading tokenizer", progress: 8 });
    const tokenizer = await AutoTokenizer.from_pretrained(id);
    onProgress({ status: "loading", stage: "Loading ONNX weights", progress: 15 });
    const files = new Map<string, number>();
    const model = await AutoModelForCausalLM.from_pretrained(id, {
      device: "wasm",
      dtype: isOffline ? "fp32" : "q4",
      progress_callback: (event: ModelProgress) => {
        if (event.file && typeof event.progress === "number") {
          files.set(event.file, Math.max(0, Math.min(100, event.progress)));
          const values = [...files.values()];
          const aggregate = values.reduce((sum, value) => sum + value, 0) / values.length;
          onProgress({
            status: "loading",
            stage: event.status === "done" ? "Preparing inference" : "Downloading ONNX weights",
            detail: event.file,
            progress: 15 + aggregate * 0.3,
          });
        }
      },
    });
    return { model, tokenizer };
  })();

  modelCache.set(key, promise);
  try {
    return await promise;
  } catch (error) {
    modelCache.delete(key);
    throw error;
  }
}

function layerIndex(choice: LayerChoice, count: number) {
  if (choice === "embedding") return 0;
  if (choice === "middle") return Math.floor((count - 1) / 2);
  return count - 1;
}

function tensorRow(tensor: any, row: number): number[] {
  const dims = tensor?.dims as number[] | undefined;
  if (!dims || dims.length !== 3 || dims[0] !== 1) {
    throw new Error(`Unexpected hidden-state shape: ${dims?.join("×") || "unknown"}`);
  }
  const width = dims[2];
  if (row < 0 || row >= dims[1]) throw new Error("Hidden-state token index is out of range");
  return Array.from(tensor.data.slice(row * width, (row + 1) * width), Number);
}

function normalizeVector(values: number[]) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const scale = Math.sqrt(variance) + 1e-6;
  return values.map((value) => (value - mean) / scale);
}

function random(seed: number) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number) {
  const u = Math.max(rand(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

function channelStats(values: number[]) {
  const rms = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length);
  const peakToPeak = Math.max(...values) - Math.min(...values);
  const deltas = values.slice(1).map((value, index) => Math.abs(value - values[index]));
  const meanDelta = deltas.length ? deltas.reduce((sum, value) => sum + value, 0) / deltas.length : 0;
  return { rms, peakToPeak, meanDelta };
}

function projectActivations(vectors: number[][], count: number, seed: number, promptLength: number): EEGChannel[] {
  if (!vectors.length) throw new Error("The model returned no token activations");
  const normalized = vectors.map(normalizeVector);
  const hiddenSize = normalized[0].length;
  return Array.from({ length: count }, (_, channel) => {
    const rand = random(seed + channel * 104729);
    const weights = Array.from({ length: hiddenSize }, () => gaussian(rand) / Math.sqrt(hiddenSize));
    const values = normalized.map((vector) =>
      vector.reduce((sum, value, index) => sum + value * weights[index], 0),
    );
    const baselineCount = Math.max(1, Math.min(promptLength, values.length));
    const baseline = values.slice(0, baselineCount).reduce((sum, value) => sum + value, 0) / baselineCount;
    const centered = values.map((value) => value - baseline);
    return {
      id: channel,
      label: `CH ${String(channel + 1).padStart(2, "0")}`,
      values: centered,
      stats: channelStats(centered),
    };
  });
}

function decodeToken(tokenizer: any, id: number) {
  const value = tokenizer.decode([id], { skip_special_tokens: false }) as string;
  return value.replace(/\n/g, "↵").replace(/\r/g, "").trim() || "␠";
}

function friendlyError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  if (/fetch failed|failed to fetch|network|timeout/i.test(message)) {
    return "Could not download the model. Check the Hugging Face endpoint, or run the offline tiny GPT2 fixture.";
  }
  if (/onnx|no valid model|model_file/i.test(message)) {
    return `${message} This demo needs an ONNX-compatible causal LM; try one of the presets.`;
  }
  return message;
}

export async function sampleModel(
  runId: string,
  modelId: string,
  settings: SampleSettings,
  onProgress: ProgressUpdate,
): Promise<ModelRun> {
  const started = performance.now();
  const base: ModelRun = {
    id: runId,
    modelId,
    resolvedModelId: resolvedId(modelId),
    status: "loading",
    progress: 2,
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

  try {
    const { model, tokenizer } = await loadModel(modelId, settings.endpoint, onProgress);
    const loadMs = performance.now() - started;
    onProgress({ status: "sampling", stage: "Tokenizing prompt", progress: 52, loadMs });
    const inputs = tokenizer(settings.prompt, { return_tensors: "pt" });
    const promptIds = (inputs.input_ids.tolist()[0] as Array<number | bigint>).map(Number);
    if (!promptIds.length) throw new Error("The prompt produced no tokens");
    if (promptIds.length > 128) throw new Error("Keep the prompt below 128 tokens for this small-model demo");

    onProgress({ status: "sampling", stage: "Reading prompt activations", progress: 60, loadMs });
    const sequenceIds = [...promptIds];
    const generatedIds: number[] = [];
    const vectors: number[][] = [];
    let source: "hidden_states" | "logits" = "logits";
    let selectedLayer = 0;
    let layerCount = 1;
    let currentOutput: any;
    const eosValue = model.generation_config?.eos_token_id;
    const eosIds = new Set((Array.isArray(eosValue) ? eosValue : [eosValue]).filter((id) => typeof id === "number"));

    for (let step = 0; step < settings.maxNewTokens; step += 1) {
      if (step > 0) {
        onProgress({
          status: "sampling",
          stage: "Generating token steps",
          detail: `Forward pass ${step}/${settings.maxNewTokens}`,
          progress: 62 + (step / settings.maxNewTokens) * 24,
          loadMs,
        });
        const input_ids = new Tensor("int64", BigInt64Array.from(sequenceIds.map(BigInt)), [1, sequenceIds.length]);
        const attention_mask = new Tensor("int64", BigInt64Array.from(sequenceIds.map(() => 1n)), [1, sequenceIds.length]);
        currentOutput = await model({ input_ids, attention_mask, output_hidden_states: true });
      } else {
        currentOutput = await model({ ...inputs, output_hidden_states: true });
      }

      const hidden = currentOutput.hidden_states as any[] | undefined;
      const useHidden = Boolean(hidden?.length);
      if (useHidden && hidden) {
        source = "hidden_states";
        layerCount = hidden.length;
        selectedLayer = layerIndex(settings.layer, layerCount);
      }
      if (step === 0) {
        if (!currentOutput.logits) throw new Error("The model returned neither logits nor usable hidden states");
        for (let index = 0; index < promptIds.length; index += 1) {
          vectors.push(tensorRow(useHidden && hidden ? hidden[selectedLayer] : currentOutput.logits, index));
        }
      }

      const lastRow = sequenceIds.length - 1;
      const predictionVector = tensorRow(useHidden && hidden ? hidden[selectedLayer] : currentOutput.logits, lastRow);
      vectors.push(predictionVector);
      const logits = tensorRow(currentOutput.logits, lastRow);
      let nextToken = 0;
      for (let index = 1; index < logits.length; index += 1) {
        if (logits[index] > logits[nextToken]) nextToken = index;
      }
      generatedIds.push(nextToken);
      sequenceIds.push(nextToken);
      if (eosIds.has(nextToken)) break;
    }

    onProgress({ status: "sampling", stage: "Projecting channels", progress: 90, loadMs });
    const tokens: TokenSample[] = [
      ...promptIds.map((id, index): TokenSample => ({
        index,
        id,
        text: decodeToken(tokenizer, id),
        phase: "prompt",
      })),
      ...generatedIds.map((id, index): TokenSample => ({
        index: promptIds.length + index,
        id,
        text: decodeToken(tokenizer, id),
        phase: "generated",
      })),
    ];
    if (tokens.length !== vectors.length) {
      throw new Error(`Token/activation mismatch: ${tokens.length} tokens but ${vectors.length} vectors`);
    }
    const channels = projectActivations(vectors, settings.channels, settings.seed, promptIds.length);
    const generatedText = tokenizer.decode(generatedIds, { skip_special_tokens: true });
    const result: ModelRun = {
      ...base,
      status: "complete",
      progress: 100,
      stage: "Acquisition complete",
      tokens,
      channels,
      generatedText,
      promptLength: promptIds.length,
      layerIndex: selectedLayer,
      layerCount,
      hiddenSize: vectors[0].length,
      source,
      loadMs,
      sampleMs: performance.now() - started - loadMs,
    };
    onProgress(result);
    return result;
  } catch (error) {
    const message = friendlyError(error);
    onProgress({ status: "error", stage: "Acquisition failed", error: message, progress: 0 });
    throw new Error(message);
  }
}
