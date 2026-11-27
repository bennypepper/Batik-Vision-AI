import {
  BatikClass,
  ExecutionProvider,
  InferenceResult,
  ModelCalibrationConfig,
  PredictionResult,
  ScanRecord,
} from "@/types/batik";
import { getAllMotifs, getMotifById, getMotifBySlug } from "@/lib/labels-data";
import type { WorkerRequest, WorkerResponse } from "@/lib/inference.worker";

export const DEFAULT_MODEL_URL =
  process.env.NEXT_PUBLIC_MODEL_URL || "/models/batik_model_v3.onnx";

export const DEFAULT_MODEL_CONFIG: ModelCalibrationConfig = {
  input_size: 224,
  temperature: 1.0,
  ood_entropy_threshold: 0.85,
  ood_min_confidence: 0.25,
};

export class ModelLoadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ModelLoadError";
  }
}

export interface TileRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface RunInferenceOptions {
  modelUrl?: string;
  timeoutMs?: number;
  onProgress?: (
    percent: number,
    stage: "downloading" | "initializing" | "inferring"
  ) => void;
}

/**
 * Extracts 3 aspect-preserving 1:1 square tiles (1 full center square + 2 interior
 * 65% macro tiles) when min(w, h) >= 260, or 1 center square when smaller.
 * Never stretches rectangular crops non-uniformly.
 */
export function extractAspectPreservingTileRects(
  width: number,
  height: number
): TileRect[] {
  const safeW = Math.max(1, Math.floor(width));
  const safeH = Math.max(1, Math.floor(height));
  const minSide = Math.min(safeW, safeH);

  const centerTile: TileRect = {
    x: Math.floor((safeW - minSide) / 2),
    y: Math.floor((safeH - minSide) / 2),
    width: minSide,
    height: minSide,
  };

  if (minSide < 260) {
    return [centerTile];
  }

  const macroSize = Math.max(64, Math.round(minSide * 0.65));
  const maxX = Math.max(0, safeW - macroSize);
  const maxY = Math.max(0, safeH - macroSize);

  let tile1: TileRect;
  let tile2: TileRect;

  if (safeW > safeH * 1.15) {
    // Wide rectangle: sample left-center and right-center interior macro squares
    const y = Math.floor((safeH - macroSize) / 2);
    tile1 = {
      x: Math.min(maxX, Math.max(0, Math.round(safeW * 0.25 - macroSize / 2))),
      y,
      width: macroSize,
      height: macroSize,
    };
    tile2 = {
      x: Math.min(maxX, Math.max(0, Math.round(safeW * 0.75 - macroSize / 2))),
      y,
      width: macroSize,
      height: macroSize,
    };
  } else if (safeH > safeW * 1.15) {
    // Tall rectangle: sample upper-center and lower-center interior macro squares
    const x = Math.floor((safeW - macroSize) / 2);
    tile1 = {
      x,
      y: Math.min(maxY, Math.max(0, Math.round(safeH * 0.25 - macroSize / 2))),
      width: macroSize,
      height: macroSize,
    };
    tile2 = {
      x,
      y: Math.min(maxY, Math.max(0, Math.round(safeH * 0.75 - macroSize / 2))),
      width: macroSize,
      height: macroSize,
    };
  } else {
    // Near-square: sample two balanced interior diagonal macro squares
    const insetX = Math.max(0, Math.round((safeW - macroSize) * 0.25));
    const insetY = Math.max(0, Math.round((safeH - macroSize) * 0.25));
    tile1 = {
      x: Math.min(maxX, insetX),
      y: Math.min(maxY, insetY),
      width: macroSize,
      height: macroSize,
    };
    tile2 = {
      x: Math.min(maxX, Math.max(0, safeW - macroSize - insetX)),
      y: Math.min(maxY, Math.max(0, safeH - macroSize - insetY)),
      width: macroSize,
      height: macroSize,
    };
  }

  return [centerTile, tile1, tile2];
}

export function temperatureSoftmax(
  logits: number[],
  temperature: number = 1.0
): number[] {
  if (logits.length === 0) return [];
  const safeTemp = Math.max(0.05, temperature);
  const scaled = logits.map((z) => z / safeTemp);
  const maxVal = Math.max(...scaled);
  const exps = scaled.map((z) => Math.exp(z - maxVal));
  const sumExps = exps.reduce((acc, v) => acc + v, 0);
  if (sumExps <= 0 || !Number.isFinite(sumExps)) {
    return new Array(logits.length).fill(1 / logits.length);
  }
  return exps.map((v) => v / sumExps);
}

function normalizeTileScores(
  scores: number[],
  temperature: number
): number[] {
  if (scores.length === 0) return [];
  const sum = scores.reduce((a, b) => a + b, 0);
  const isAlreadyProb =
    Math.abs(sum - 1.0) < 0.05 &&
    scores.every((v) => v >= -0.001 && v <= 1.001);

  if (isAlreadyProb) {
    const clamped = scores.map((v) => Math.max(1e-9, v));
    if (Math.abs(temperature - 1.0) > 1e-3) {
      const logProbs = clamped.map((p) => Math.log(p));
      return temperatureSoftmax(logProbs, temperature);
    }
    const clampedSum = clamped.reduce((a, b) => a + b, 0);
    return clamped.map((v) => v / (clampedSum || 1));
  }

  return temperatureSoftmax(scores, temperature);
}

function computeNormalizedEntropy(probs: number[]): number {
  const n = probs.length;
  if (n <= 1) return 0;
  const maxEntropy = Math.log(n);
  let entropy = 0;
  for (const p of probs) {
    if (p > 1e-12) {
      entropy -= p * Math.log(p);
    }
  }
  return Math.max(0, Math.min(1, entropy / maxEntropy));
}

/**
 * Inverse-entropy weighted multi-patch voting + OOD detection.
 * Sharper fabric tiles (lower Shannon entropy) carry higher weight than noisy background tiles.
 */
export function aggregateTilePredictions(
  tileOutputs: number[][],
  config: ModelCalibrationConfig = DEFAULT_MODEL_CONFIG
): {
  probabilities: number[];
  normalizedEntropy: number;
  topConfidence: number;
  isOutOfDistribution: boolean;
} {
  if (tileOutputs.length === 0 || tileOutputs[0].length === 0) {
    return {
      probabilities: [],
      normalizedEntropy: 1,
      topConfidence: 0,
      isOutOfDistribution: true,
    };
  }

  const numClasses = tileOutputs[0].length;
  const weightedSum = new Array<number>(numClasses).fill(0);
  let totalWeight = 0;

  for (const rawTile of tileOutputs) {
    const probs = normalizeTileScores(rawTile, config.temperature);
    const hNorm = computeNormalizedEntropy(probs);
    // Inverse Shannon entropy weight (1 - normalizedEntropy), floored at 0.05
    const weight = Math.max(0.05, 1 - hNorm);
    totalWeight += weight;
    for (let i = 0; i < numClasses; i++) {
      weightedSum[i] += weight * (probs[i] ?? 0);
    }
  }

  const probabilities = weightedSum.map((v) => v / (totalWeight || 1));
  const normalizedEntropy = computeNormalizedEntropy(probabilities);
  const topConfidence = Math.max(...probabilities);
  const isOutOfDistribution =
    topConfidence < config.ood_min_confidence ||
    normalizedEntropy > config.ood_entropy_threshold;

  return {
    probabilities,
    normalizedEntropy,
    topConfidence,
    isOutOfDistribution,
  };
}

/**
 * Strict runtime validator for localStorage scan history records.
 * Allowlists class slugs against MOTIFS and strips any large imageThumbnail payloads.
 */
export function validateHistoryRecords(raw: unknown): ScanRecord[] {
  if (!Array.isArray(raw)) return [];

  const validRecords: ScanRecord[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const rec = item as Record<string, unknown>;

    if (
      typeof rec.id !== "string" ||
      rec.id.length === 0 ||
      typeof rec.timestamp !== "number" ||
      !Number.isFinite(rec.timestamp) ||
      typeof rec.inferenceTimeMs !== "number" ||
      !Number.isFinite(rec.inferenceTimeMs)
    ) {
      continue;
    }

    const topPred = validatePredictionItem(rec.topPrediction);
    if (!topPred) continue;

    const rawCandidates = Array.isArray(rec.topCandidates)
      ? rec.topCandidates
      : [rec.topPrediction];
    const topCandidates = rawCandidates
      .map(validatePredictionItem)
      .filter((p): p is PredictionResult => p !== null);

    validRecords.push({
      id: rec.id,
      timestamp: rec.timestamp,
      topPrediction: topPred,
      topCandidates: topCandidates.length > 0 ? topCandidates : [topPred],
      inferenceTimeMs: Math.max(0, Math.round(rec.inferenceTimeMs)),
    });
  }

  return validRecords.slice(0, 30);
}

function validatePredictionItem(raw: unknown): PredictionResult | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as Record<string, unknown>;
  if (
    typeof obj.rank !== "number" ||
    !Number.isFinite(obj.rank) ||
    typeof obj.score !== "number" ||
    !Number.isFinite(obj.score)
  ) {
    return null;
  }

  const classInfoObj =
    obj.classInfo && typeof obj.classInfo === "object"
      ? (obj.classInfo as Record<string, unknown>)
      : null;
  const slugCandidate =
    typeof classInfoObj?.slug === "string"
      ? classInfoObj.slug
      : typeof obj.label === "string"
      ? obj.label
      : "";

  const canonical = getMotifBySlug(slugCandidate);
  if (!canonical) return null;

  return {
    rank: Math.max(1, Math.round(obj.rank)),
    score: Math.max(0, Math.min(100, obj.score)),
    label: canonical.slug,
    classInfo: canonical,
  };
}

let workerInstance: Worker | null = null;
let nextRequestId = 1;
let inferenceQueue: Promise<void> = Promise.resolve();

function getInferenceWorker(): Worker {
  if (typeof window === "undefined" || typeof Worker === "undefined") {
    throw new ModelLoadError(
      "Web Workers are not available in this environment."
    );
  }
  if (!workerInstance) {
    workerInstance = new Worker(
      new URL("./inference.worker.ts", import.meta.url),
      { type: "module" }
    );
  }
  return workerInstance;
}

export function resetInferenceWorker(): void {
  if (workerInstance) {
    workerInstance.terminate();
    workerInstance = null;
  }
}

function requestWorker<T extends WorkerResponse>(
  payload: WorkerRequest,
  transfer: Transferable[],
  timeoutMs: number,
  onProgress?: RunInferenceOptions["onProgress"]
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const worker = getInferenceWorker();
    let settled = false;

    const cleanup = () => {
      clearTimeout(timer);
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
    };

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      cleanup();
      resetInferenceWorker();
      reject(
        new ModelLoadError(
          `ONNX inference timed out after ${Math.round(timeoutMs / 1000)}s.`
        )
      );
    }, timeoutMs);

    const handleMessage = (event: MessageEvent<WorkerResponse>) => {
      const msg = event.data;
      if (msg.requestId !== payload.requestId) return;

      if (msg.type === "progress") {
        onProgress?.(msg.percent, msg.stage);
        return;
      }

      if (settled) return;
      settled = true;
      cleanup();

      if (msg.type === "error") {
        reject(new ModelLoadError(msg.message));
      } else {
        resolve(msg as T);
      }
    };

    const handleError = (err: ErrorEvent) => {
      if (settled) return;
      settled = true;
      cleanup();
      resetInferenceWorker();
      reject(
        new ModelLoadError(
          err.message || "ONNX Web Worker crashed unexpectedly."
        )
      );
    };

    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.postMessage(payload, transfer);
  });
}

function buildMultiPatchFloat32Buffer(
  canvas: HTMLCanvasElement,
  inputSize: number
): { buffer: ArrayBuffer; batchSize: number } {
  const rects = extractAspectPreservingTileRects(canvas.width, canvas.height);
  const batchSize = rects.length;
  const floatArr = new Float32Array(batchSize * inputSize * inputSize * 3);

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = inputSize;
  tempCanvas.height = inputSize;
  const ctx = tempCanvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new ModelLoadError("Failed to create 2D canvas context for tiling.");
  }

  let p = 0;
  for (const rect of rects) {
    ctx.clearRect(0, 0, inputSize, inputSize);
    ctx.drawImage(
      canvas,
      rect.x,
      rect.y,
      rect.width,
      rect.height,
      0,
      0,
      inputSize,
      inputSize
    );
    const imgData = ctx.getImageData(0, 0, inputSize, inputSize).data;
    for (let i = 0; i < imgData.length; i += 4) {
      // EfficientNetV2S expects raw float32 RGB in [0, 255]
      floatArr[p++] = imgData[i];
      floatArr[p++] = imgData[i + 1];
      floatArr[p++] = imgData[i + 2];
    }
  }

  return { buffer: floatArr.buffer, batchSize };
}

function mapProbabilitiesToPredictions(
  probabilities: number[],
  motifs: BatikClass[]
): PredictionResult[] {
  const indexed = probabilities.map((prob, idx) => ({
    prob,
    classInfo: getMotifById(idx) || motifs[idx] || motifs[0],
  }));

  indexed.sort((a, b) => b.prob - a.prob);

  return indexed.slice(0, 5).map((item, index) => ({
    rank: index + 1,
    score: Math.round(item.prob * 1000) / 10,
    label: item.classInfo.slug,
    classInfo: item.classInfo,
  }));
}

export async function runInferenceOnCanvas(
  canvas: HTMLCanvasElement,
  options: RunInferenceOptions = {}
): Promise<InferenceResult> {
  const modelUrl = options.modelUrl || DEFAULT_MODEL_URL;
  const timeoutMs = options.timeoutMs ?? 60000;

  let result!: InferenceResult;
  let caughtError: unknown = null;

  // Serialize ONNX Web Worker calls to prevent overlapping runs
  const task = inferenceQueue.then(async () => {
    try {
      const motifs = getAllMotifs();

      const configReqId = nextRequestId++;
      const configResp = await requestWorker<
        Extract<WorkerResponse, { type: "config" }>
      >(
        { type: "get-config", requestId: configReqId, modelUrl },
        [],
        15000
      );
      const inputSize = configResp.config.input_size || DEFAULT_MODEL_CONFIG.input_size;

      const { buffer, batchSize } = buildMultiPatchFloat32Buffer(
        canvas,
        inputSize
      );

      const inferReqId = nextRequestId++;
      const workerResult = await requestWorker<
        Extract<WorkerResponse, { type: "result" }>
      >(
        {
          type: "infer",
          requestId: inferReqId,
          modelUrl,
          inputBuffer: buffer,
          batchSize,
          inputSize,
        },
        [buffer],
        timeoutMs,
        options.onProgress
      );

      const aggregated = aggregateTilePredictions(
        workerResult.rawOutputs,
        workerResult.config
      );
      const predictions = mapProbabilitiesToPredictions(
        aggregated.probabilities,
        motifs
      );

      result = {
        predictions,
        inferenceTimeMs: workerResult.inferenceTimeMs,
        executionProvider: workerResult.executionProvider as ExecutionProvider,
        isOutOfDistribution: aggregated.isOutOfDistribution,
        normalizedEntropy: aggregated.normalizedEntropy,
        topConfidence: aggregated.topConfidence,
      };
    } catch (err) {
      caughtError = err;
    }
  });

  inferenceQueue = task;
  await task;

  if (caughtError) {
    if (caughtError instanceof ModelLoadError) {
      throw caughtError;
    }
    throw new ModelLoadError(
      caughtError instanceof Error
        ? caughtError.message
        : "Failed to run ONNX model inference."
    );
  }

  return result;
}
