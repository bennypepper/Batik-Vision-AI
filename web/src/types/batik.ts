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

export type AppLanguage = "id" | "en";

export interface ScanRecord {
  id: string;
  timestamp: number;
  imageThumbnail: string;
  topPrediction: PredictionResult;
  topCandidates: PredictionResult[];
  inferenceTimeMs: number;
}
