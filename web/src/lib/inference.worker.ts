/// <reference lib="webworker" />

import * as ort from "onnxruntime-web";
import type {
  ExecutionProvider,
  ModelCalibrationConfig,
} from "@/types/batik";

const MODEL_CACHE_NAME = "batik-models-v1";
const DEFAULT_CONFIG: ModelCalibrationConfig = {
  input_size: 224,
  temperature: 1.0,
  ood_entropy_threshold: 0.85,
  ood_min_confidence: 0.25,
};

export interface WorkerInferRequest {
  type: "infer";
  requestId: number;
  modelUrl: string;
  inputBuffer: ArrayBuffer;
  batchSize: number;
  inputSize: number;
}

export interface WorkerConfigRequest {
  type: "get-config";
  requestId: number;
  modelUrl: string;
}

export type WorkerRequest = WorkerInferRequest | WorkerConfigRequest;

export type WorkerResponse =
  | {
      type: "progress";
      requestId: number;
      percent: number;
      stage: "downloading" | "initializing" | "inferring";
    }
  | {
      type: "config";
      requestId: number;
      config: ModelCalibrationConfig;
    }
  | {
      type: "result";
      requestId: number;
      rawOutputs: number[][];
      executionProvider: ExecutionProvider;
      config: ModelCalibrationConfig;
      inferenceTimeMs: number;
    }
  | {
      type: "error";
      requestId: number;
      message: string;
    };

let activeSession: ort.InferenceSession | null = null;
let activeModelUrl: string | null = null;
let activeProvider: ExecutionProvider = "wasm";
let activeConfig: ModelCalibrationConfig = { ...DEFAULT_CONFIG };
let configLoadedForUrl: string | null = null;

// Self-hosted ONNX Runtime WASM binaries copied into /public/ort/ at build time
ort.env.wasm.wasmPaths = "/ort/";
ort.env.wasm.numThreads = 1;

async function loadOptionalModelConfig(
  modelUrl: string
): Promise<ModelCalibrationConfig> {
  if (configLoadedForUrl === modelUrl) {
    return activeConfig;
  }

  try {
    const configUrl = modelUrl.includes("/")
      ? modelUrl.replace(/[^/]+$/, "model_config.json")
      : "/models/model_config.json";
    const res = await fetch(configUrl, { cache: "no-cache" });
    if (res.ok) {
      const json: unknown = await res.json();
      if (json && typeof json === "object") {
        const obj = json as Record<string, unknown>;
        activeConfig = {
          input_size:
            typeof obj.input_size === "number" &&
            Number.isFinite(obj.input_size) &&
            obj.input_size >= 64
              ? Math.round(obj.input_size)
              : DEFAULT_CONFIG.input_size,
          temperature:
            typeof obj.temperature === "number" &&
            Number.isFinite(obj.temperature) &&
            obj.temperature > 0
              ? obj.temperature
              : DEFAULT_CONFIG.temperature,
          ood_entropy_threshold:
            typeof obj.ood_entropy_threshold === "number" &&
            Number.isFinite(obj.ood_entropy_threshold)
              ? obj.ood_entropy_threshold
              : DEFAULT_CONFIG.ood_entropy_threshold,
          ood_min_confidence:
            typeof obj.ood_min_confidence === "number" &&
            Number.isFinite(obj.ood_min_confidence)
              ? obj.ood_min_confidence
              : DEFAULT_CONFIG.ood_min_confidence,
        };
      }
    }
  } catch {
    activeConfig = { ...DEFAULT_CONFIG };
  }

  configLoadedForUrl = modelUrl;
  return activeConfig;
}

async function fetchModelBytesWithProgress(
  modelUrl: string,
  requestId: number
): Promise<Uint8Array> {
  // Check Cache API first for instant reload
  if (typeof caches !== "undefined") {
    try {
      const cache = await caches.open(MODEL_CACHE_NAME);
      const cached = await cache.match(modelUrl);
      if (cached && cached.ok) {
        const buf = await cached.arrayBuffer();
        postWorkerMessage({
          type: "progress",
          requestId,
          percent: 100,
          stage: "downloading",
        });
        return new Uint8Array(buf);
      }
    } catch {
      // Cache API unavailable in private browsing or restricted context
    }
  }

  const response = await fetch(modelUrl);
  if (!response.ok) {
    throw new Error(
      `Failed to download ONNX model (${response.status} ${response.statusText}) from ${modelUrl}`
    );
  }

  const contentLengthHeader = response.headers.get("Content-Length");
  const totalBytes = contentLengthHeader
    ? parseInt(contentLengthHeader, 10)
    : 0;

  if (!response.body) {
    const buf = await response.arrayBuffer();
    postWorkerMessage({
      type: "progress",
      requestId,
      percent: 100,
      stage: "downloading",
    });
    return new Uint8Array(buf);
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;
  let lastPercent = -1;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loadedBytes += value.byteLength;
      if (totalBytes > 0) {
        const percent = Math.min(99, Math.round((loadedBytes / totalBytes) * 100));
        if (percent !== lastPercent) {
          lastPercent = percent;
          postWorkerMessage({
            type: "progress",
            requestId,
            percent,
            stage: "downloading",
          });
        }
      }
    }
  }

  const combined = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.byteLength;
  }

  postWorkerMessage({
    type: "progress",
    requestId,
    percent: 100,
    stage: "downloading",
  });

  // Store in Cache API for future sessions
  if (typeof caches !== "undefined") {
    try {
      const cache = await caches.open(MODEL_CACHE_NAME);
      await cache.put(
        modelUrl,
        new Response(combined.slice().buffer, {
          status: 200,
          headers: {
            "Content-Type": "application/octet-stream",
            "Content-Length": String(combined.byteLength),
          },
        })
      );
    } catch {
      // Ignore cache quota failures
    }
  }

  return combined;
}

async function createSessionWithFallback(
  modelBytes: Uint8Array
): Promise<{ session: ort.InferenceSession; provider: ExecutionProvider }> {
  const candidates: ExecutionProvider[] = [];
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    candidates.push("webgpu");
  }
  candidates.push("webgl", "wasm");

  let lastError: unknown = null;
  for (const provider of candidates) {
    try {
      const session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: [provider],
      });
      return { session, provider };
    } catch (err) {
      lastError = err;
    }
  }

  throw new Error(
    `Failed to initialize ONNX Runtime session across providers (${candidates.join(", ")}): ${
      lastError instanceof Error ? lastError.message : String(lastError)
    }`
  );
}

async function ensureSession(
  modelUrl: string,
  requestId: number
): Promise<ort.InferenceSession> {
  if (activeSession && activeModelUrl === modelUrl) {
    return activeSession;
  }

  if (activeSession) {
    try {
      await activeSession.release();
    } catch {
      // Ignore release errors on old session
    }
    activeSession = null;
    activeModelUrl = null;
  }

  await loadOptionalModelConfig(modelUrl);
  const modelBytes = await fetchModelBytesWithProgress(modelUrl, requestId);

  postWorkerMessage({
    type: "progress",
    requestId,
    percent: 100,
    stage: "initializing",
  });

  const { session, provider } = await createSessionWithFallback(modelBytes);
  activeSession = session;
  activeModelUrl = modelUrl;
  activeProvider = provider;
  return session;
}

function postWorkerMessage(msg: WorkerResponse) {
  self.postMessage(msg);
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const data = event.data;

  if (data.type === "get-config") {
    const config = await loadOptionalModelConfig(data.modelUrl);
    postWorkerMessage({
      type: "config",
      requestId: data.requestId,
      config,
    });
    return;
  }

  if (data.type === "infer") {
    const { requestId, modelUrl, inputBuffer, batchSize, inputSize } = data;
    let inputTensor: ort.Tensor | null = null;
    let outputTensor: ort.Tensor | null = null;

    try {
      const start = performance.now();
      const session = await ensureSession(modelUrl, requestId);

      postWorkerMessage({
        type: "progress",
        requestId,
        percent: 100,
        stage: "inferring",
      });

      const floatArr = new Float32Array(inputBuffer);
      inputTensor = new ort.Tensor("float32", floatArr, [
        batchSize,
        inputSize,
        inputSize,
        3,
      ]);

      const inputName = session.inputNames[0];
      const feeds: Record<string, ort.Tensor> = {
        [inputName]: inputTensor,
      };

      const results = await session.run(feeds);
      const outputName = session.outputNames[0];
      outputTensor = results[outputName];

      const rawFlat = Array.from(outputTensor.data as Float32Array);
      const numClasses =
        batchSize > 0 ? Math.floor(rawFlat.length / batchSize) : rawFlat.length;

      const rawOutputs: number[][] = [];
      for (let b = 0; b < batchSize; b++) {
        rawOutputs.push(
          rawFlat.slice(b * numClasses, (b + 1) * numClasses)
        );
      }

      const elapsed = Math.round(performance.now() - start);
      postWorkerMessage({
        type: "result",
        requestId,
        rawOutputs,
        executionProvider: activeProvider,
        config: activeConfig,
        inferenceTimeMs: elapsed,
      });
    } catch (err) {
      postWorkerMessage({
        type: "error",
        requestId,
        message:
          err instanceof Error
            ? err.message
            : "Unknown ONNX Web Worker inference error",
      });
    } finally {
      try {
        inputTensor?.dispose();
      } catch {
        // Ignore disposal error
      }
      try {
        outputTensor?.dispose();
      } catch {
        // Ignore disposal error
      }
    }
  }
};
