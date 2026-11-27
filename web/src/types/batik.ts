export interface BatikClass {
  id: number;
  slug: string;
  name_id: string;
  name_en: string;
  region: string;
  region_id?: string;
  region_en?: string;
  island: string;
  island_id?: string;
  island_en?: string;
  is_weak_class: boolean;
  description_id: string;
  description_en: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  label?: string;
}

export interface PredictionResult {
  rank: number;
  score: number;
  label: string;
  classInfo: BatikClass;
}

export type ExecutionProvider = "webgpu" | "webgl" | "wasm";

export interface ModelCalibrationConfig {
  input_size: number;
  temperature: number;
  ood_entropy_threshold: number;
  ood_min_confidence: number;
}

export interface InferenceResult {
  predictions: PredictionResult[];
  inferenceTimeMs: number;
  executionProvider: ExecutionProvider;
  isOutOfDistribution: boolean;
  normalizedEntropy: number;
  topConfidence: number;
}

export type AppLanguage = "id" | "en";

export interface ScanRecord {
  id: string;
  timestamp: number;
  imageThumbnail?: string;
  topPrediction: PredictionResult;
  topCandidates: PredictionResult[];
  inferenceTimeMs: number;
}
