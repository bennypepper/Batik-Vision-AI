import { describe, it, expect } from "vitest";
import {
  extractAspectPreservingTileRects,
  temperatureSoftmax,
  aggregateTilePredictions,
  validateHistoryRecords,
  DEFAULT_MODEL_CONFIG,
} from "@/lib/classifier";
import { MOTIFS } from "@/lib/labels-data";

describe("Stage-2 Classifier Helpers & History Validation", () => {
  it("extracts 3 aspect-preserving square tiles for wide rectangular crops >= 260px", () => {
    const rects = extractAspectPreservingTileRects(800, 400);
    expect(rects).toHaveLength(3);
    for (const r of rects) {
      expect(r.width).toBe(r.height);
      expect(r.x).toBeGreaterThanOrEqual(0);
      expect(r.y).toBeGreaterThanOrEqual(0);
      expect(r.x + r.width).toBeLessThanOrEqual(800);
      expect(r.y + r.height).toBeLessThanOrEqual(400);
    }
    // First tile is the full center square (400x400)
    expect(rects[0].width).toBe(400);
    // Remaining tiles are 65% macro tiles (260x260)
    expect(rects[1].width).toBe(Math.round(400 * 0.65));
    expect(rects[2].width).toBe(Math.round(400 * 0.65));
  });

  it("extracts 1 aspect-preserving center square when min dimension < 260px", () => {
    const rects = extractAspectPreservingTileRects(200, 150);
    expect(rects).toHaveLength(1);
    expect(rects[0]).toEqual({ x: 25, y: 0, width: 150, height: 150 });
  });

  it("computes numerically stable temperatureSoftmax", () => {
    const logits = [1000, 1002, 1001];
    const probsT1 = temperatureSoftmax(logits, 1.0);
    const probsT2 = temperatureSoftmax(logits, 2.0);

    const sum1 = probsT1.reduce((a, b) => a + b, 0);
    const sum2 = probsT2.reduce((a, b) => a + b, 0);
    expect(sum1).toBeCloseTo(1.0, 6);
    expect(sum2).toBeCloseTo(1.0, 6);
    // Higher temperature softens the peak probability
    expect(probsT2[1]).toBeLessThan(probsT1[1]);
    expect(probsT1[1]).toBeGreaterThan(probsT1[2]);
    expect(probsT1[2]).toBeGreaterThan(probsT1[0]);
  });

  it("aggregates tiles using inverse-entropy weighting so sharp tiles dominate noisy ones", () => {
    const numClasses = 28;
    // Tile 1: very confident on class 5
    const sharpLogits = new Array(numClasses).fill(-2);
    sharpLogits[5] = 8;
    // Tile 2: completely flat / uniform noise slightly favoring class 2
    const noisyLogits = new Array(numClasses).fill(0);
    noisyLogits[2] = 0.05;

    const result = aggregateTilePredictions(
      [sharpLogits, noisyLogits],
      DEFAULT_MODEL_CONFIG
    );
    expect(result.probabilities).toHaveLength(numClasses);
    const topIdx = result.probabilities.indexOf(Math.max(...result.probabilities));
    expect(topIdx).toBe(5);
    expect(result.isOutOfDistribution).toBe(false);
    expect(result.topConfidence).toBeGreaterThan(0.5);
  });

  it("flags Out-Of-Distribution (OOD) on high-entropy / low-confidence predictions", () => {
    const numClasses = 28;
    const flatLogits = new Array(numClasses).fill(0);
    flatLogits[0] = 0.2;

    const result = aggregateTilePredictions([flatLogits], DEFAULT_MODEL_CONFIG);
    expect(result.isOutOfDistribution).toBe(true);
    expect(result.normalizedEntropy).toBeGreaterThan(
      DEFAULT_MODEL_CONFIG.ood_entropy_threshold
    );
  });

  it("validates localStorage history records strictly and strips invalid entries and thumbnails", () => {
    const validMotif = MOTIFS[0];
    const raw = [
      {
        id: "scan_123",
        timestamp: 1700000000000,
        imageThumbnail: "data:image/jpeg;base64," + "A".repeat(1000),
        inferenceTimeMs: 120,
        topPrediction: {
          rank: 1,
          score: 91.5,
          label: validMotif.slug,
          classInfo: validMotif,
        },
        topCandidates: [
          {
            rank: 1,
            score: 91.5,
            label: validMotif.slug,
            classInfo: validMotif,
          },
        ],
      },
      {
        id: "scan_bad_slug",
        timestamp: 1700000000001,
        inferenceTimeMs: 50,
        topPrediction: {
          rank: 1,
          score: 80,
          label: "../../etc/passwd",
          classInfo: { ...validMotif, slug: "../../etc/passwd" },
        },
        topCandidates: [],
      },
      null,
      "corrupted",
    ];

    const validated = validateHistoryRecords(raw);
    expect(validated).toHaveLength(1);
    expect(validated[0].id).toBe("scan_123");
    expect(validated[0].imageThumbnail).toBeUndefined();
    expect(validated[0].topPrediction.classInfo.slug).toBe(validMotif.slug);
    expect(validateHistoryRecords("not-an-array")).toEqual([]);
  });
});
