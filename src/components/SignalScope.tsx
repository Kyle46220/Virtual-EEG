import { AlertTriangle, LoaderCircle } from "lucide-react";
import type { ModelRun } from "@/types/eeg";

interface SignalScopeProps {
  run?: ModelRun;
  modelId: string;
  accent: string;
  activeChannel: number | "all";
  alignMode: "token" | "normalized";
  sharedScale: number;
  maxTokens: number;
}

const WIDTH = 800;
const HEIGHT = 310;
const LEFT = 54;
const RIGHT = 16;
const TOP = 14;
const BOTTOM = 34;
const PLOT_WIDTH = WIDTH - LEFT - RIGHT;
const PLOT_HEIGHT = HEIGHT - TOP - BOTTOM;

function pathFor(values: number[], lane: number, lanes: number, scale: number, alignMode: "token" | "normalized", maxTokens: number) {
  const laneHeight = PLOT_HEIGHT / lanes;
  const baseline = TOP + laneHeight * (lane + 0.5);
  const amplitude = laneHeight * 0.38;
  const denominator = alignMode === "token" ? Math.max(1, maxTokens - 1) : Math.max(1, values.length - 1);
  return values
    .map((value, index) => {
      const x = LEFT + (index / denominator) * PLOT_WIDTH;
      const y = baseline - (value / scale) * amplitude;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${Math.max(TOP + lane * laneHeight + 2, Math.min(TOP + (lane + 1) * laneHeight - 2, y)).toFixed(2)}`;
    })
    .join(" ");
}

export default function SignalScope({ run, modelId, accent, activeChannel, alignMode, sharedScale, maxTokens }: SignalScopeProps) {
  const complete = run?.status === "complete" && run.channels.length > 0;
  const channels = complete
    ? activeChannel === "all"
      ? run.channels
      : run.channels.filter((channel) => channel.id === activeChannel)
    : [];
  const axisDenominator = complete
    ? alignMode === "token" ? Math.max(1, maxTokens - 1) : Math.max(1, run.tokens.length - 1)
    : 1;
  const promptBoundary = complete && run.promptLength > 0
    ? LEFT + ((run.promptLength - 0.5) / axisDenominator) * PLOT_WIDTH
    : 0;
  const lastTokenX = complete ? LEFT + ((run.tokens.length - 1) / axisDenominator) * PLOT_WIDTH : LEFT;

  return (
    <section className="model-scope" aria-label={`Signal for ${modelId}`}>
      <header className="scope-header">
        <div className="scope-model">
          <span className="model-led" style={{ backgroundColor: accent }} />
          <div>
            <h2>{modelId}</h2>
            <p>{run?.resolvedModelId && run.resolvedModelId !== modelId ? run.resolvedModelId : "CAUSAL LM"} / {run?.source === "hidden_states" ? "HIDDEN STATES" : "LOGIT TELEMETRY"}</p>
          </div>
        </div>
        <div className="scope-status">
          <span>{run?.stage || "Standing by"}</span>
          <strong style={{ color: accent }}>{run?.status === "complete" ? "LOCKED" : run?.status === "error" ? "FAULT" : "CH—A"}</strong>
        </div>
      </header>

      {run?.status === "error" ? (
        <div className="scope-message scope-error" role="alert">
          <AlertTriangle size={22} aria-hidden="true" />
          <div><strong>Acquisition failed</strong><p>{run.error}</p></div>
        </div>
      ) : !complete ? (
        <div className="scope-message scope-waiting">
          {run?.status === "loading" || run?.status === "sampling" ? (
            <LoaderCircle className="spin" size={25} aria-hidden="true" />
          ) : (
            <span className="waiting-cross" aria-hidden="true">+</span>
          )}
          <div>
            <strong>{run?.stage || "Awaiting acquisition"}</strong>
            <p>{run?.detail || "Configure a model and start a run to capture token-step activations."}</p>
            {run && typeof run.progress === "number" && run.progress > 0 && (
              <div className="load-track" aria-label={`${Math.round(run.progress)} percent complete`}>
                <span style={{ width: `${run.progress}%`, backgroundColor: accent }} />
              </div>
            )}
          </div>
        </div>
      ) : (
        <>
          <div className="scope-canvas">
            <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`${channels.length} projected activation channels for ${modelId}`}>
              <defs>
                <clipPath id={`plot-${run.id}`}><rect x={LEFT} y={TOP} width={PLOT_WIDTH} height={PLOT_HEIGHT} /></clipPath>
              </defs>
              <rect className="plot-bg" x={LEFT} y={TOP} width={PLOT_WIDTH} height={PLOT_HEIGHT} />
              {[0, 1, 2, 3, 4, 5].map((tick) => {
                const x = LEFT + (tick / 5) * PLOT_WIDTH;
                return <line key={tick} className="grid-line" x1={x} x2={x} y1={TOP} y2={TOP + PLOT_HEIGHT} />;
              })}
              {channels.map((channel, lane) => {
                const laneHeight = PLOT_HEIGHT / channels.length;
                const y = TOP + laneHeight * (lane + 0.5);
                return (
                  <g key={channel.id}>
                    <line className="channel-baseline" x1={LEFT} x2={LEFT + PLOT_WIDTH} y1={y} y2={y} />
                    <text className="channel-label" x={8} y={y + 4}>{channel.label}</text>
                  </g>
                );
              })}
              {promptBoundary > LEFT && promptBoundary < LEFT + PLOT_WIDTH && (
                <g>
                  <line className="prompt-boundary" x1={promptBoundary} x2={promptBoundary} y1={TOP} y2={TOP + PLOT_HEIGHT} />
                  <text className="boundary-label" x={promptBoundary + 5} y={TOP + 12}>GENERATION</text>
                </g>
              )}
              <g clipPath={`url(#plot-${run.id})`}>
                {channels.map((channel, lane) => (
                  <path
                    key={channel.id}
                    d={pathFor(channel.values, lane, channels.length, sharedScale, alignMode, maxTokens)}
                    fill="none"
                    stroke={accent}
                    strokeWidth={activeChannel === "all" ? 1.35 : 1.8}
                    vectorEffect="non-scaling-stroke"
                    className="signal-path"
                  />
                ))}
              </g>
              <text className="axis-label" x={LEFT} y={HEIGHT - 10}>TOKEN 0</text>
              <text className="axis-label" x={alignMode === "normalized" ? LEFT + PLOT_WIDTH : lastTokenX} y={HEIGHT - 10} textAnchor="end">
                {alignMode === "normalized" ? "100% SEQUENCE" : `TOKEN ${run.tokens.length - 1}`}
              </text>
              <text className="axis-label" x={WIDTH - 4} y={TOP + 10} textAnchor="end">±{sharedScale.toFixed(2)} AU</text>
              {run.tokens.map((token, index) => {
                const x = LEFT + (index / Math.max(1, run.tokens.length - 1)) * PLOT_WIDTH;
                return (
                  <circle key={`${token.index}-${token.id}`} cx={x} cy={TOP + PLOT_HEIGHT + 4} r={5} fill="transparent">
                    <title>{`${token.index} · ${token.phase} · ${token.text}`}</title>
                  </circle>
                );
              })}
            </svg>
          </div>

          <div className="token-ribbon" aria-label="Sampled tokens">
            {run.tokens.map((token) => (
              <span key={`${token.index}-${token.id}`} className={token.phase} title={`${token.phase} token ${token.index}`}>
                <small>{token.index}</small>{token.text}
              </span>
            ))}
          </div>

          <div className="run-readout">
            <div><span>GENERATED</span><p>{run.generatedText || "—"}</p></div>
            <dl>
              <div><dt>SOURCE</dt><dd>{run.source === "hidden_states" ? "HIDDEN" : "LOGITS"}</dd></div>
              <div><dt>LAYER</dt><dd>{run.source === "hidden_states" ? `${run.layerIndex + 1}/${run.layerCount}` : "—"}</dd></div>
              <div><dt>WIDTH</dt><dd>{run.hiddenSize}</dd></div>
              <div><dt>SAMPLE</dt><dd>{(run.sampleMs / 1000).toFixed(1)}s</dd></div>
            </dl>
          </div>
        </>
      )}
    </section>
  );
}
