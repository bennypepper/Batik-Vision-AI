import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(__dirname, "..");
const srcDir = join(webRoot, "node_modules", "onnxruntime-web", "dist");
const destDir = join(webRoot, "public", "ort");

if (!existsSync(srcDir)) {
  console.warn(
    `[copy-ort-wasm] Skipping: ${srcDir} not found (install dependencies first).`
  );
  process.exit(0);
}

mkdirSync(destDir, { recursive: true });

const files = readdirSync(srcDir).filter(
  (name) =>
    name.startsWith("ort-wasm-") &&
    (name.endsWith(".wasm") || name.endsWith(".mjs"))
);

if (files.length === 0) {
  console.error(`[copy-ort-wasm] No ort-wasm-*.wasm/.mjs files found in ${srcDir}`);
  process.exit(1);
}

for (const file of files) {
  cpSync(join(srcDir, file), join(destDir, file));
}

console.log(
  `[copy-ort-wasm] Copied ${files.length} ONNX Runtime WASM assets to public/ort/`
);
