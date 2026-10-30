import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const endpoint = (process.env.HF_ENDPOINT || "https://huggingface.co").replace(/\/$/, "");
const repository = "onnx-community/tiny-random-gpt2-ONNX";
const destination = path.resolve("public/models/tiny-random-gpt2");
const files = [
  "config.json",
  "generation_config.json",
  "merges.txt",
  "onnx/model.onnx",
  "special_tokens_map.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "vocab.json",
];

for (const file of files) {
  const url = `${endpoint}/${repository}/resolve/main/${file}`;
  let lastError;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(90_000) });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const target = path.join(destination, file);
      await mkdir(path.dirname(target), { recursive: true });
      await writeFile(target, Buffer.from(await response.arrayBuffer()));
      console.log(`Downloaded ${file}`);
      lastError = undefined;
      break;
    } catch (error) {
      lastError = error;
      if (attempt < 3) console.warn(`Retrying ${file} (${attempt}/3)`);
    }
  }
  if (lastError) {
    console.error(`Failed to download ${file}: ${lastError.message}`);
    console.error("Check your connection or set HF_ENDPOINT to a Hugging Face-compatible mirror.");
    process.exit(1);
  }
}

console.log("Offline tiny GPT2 is ready. Select offline:tiny-gpt2 in the app.");
