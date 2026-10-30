export type LayerChoice = "last" | "middle" | "embedding";
export type RunStatus = "idle" | "queued" | "loading" | "sampling" | "complete" | "error";

export interface SampleSettings {
  prompt: string;
  maxNewTokens: number;
  channels: number;
  layer: LayerChoice;
  seed: number;
  endpoint: string;
}

export interface TokenSample {
  index: number;
  id: number;
  text: string;
  phase: "prompt" | "generated";
}

export interface ChannelStats {
  rms: number;
  peakToPeak: number;
  meanDelta: number;
}

export interface EEGChannel {
  id: number;
  label: string;
  values: number[];
  stats: ChannelStats;
}

export interface ModelRun {
  id: string;
  modelId: string;
  resolvedModelId: string;
  status: RunStatus;
  progress: number;
  stage: string;
  detail?: string;
  error?: string;
  tokens: TokenSample[];
  channels: EEGChannel[];
  generatedText: string;
  promptLength: number;
  layerIndex: number;
  layerCount: number;
  hiddenSize: number;
  source: "hidden_states" | "logits";
  loadMs: number;
  sampleMs: number;
  settings: SampleSettings;
}

export interface ModelProgress {
  status?: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}
