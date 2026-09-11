import { BoundingBox } from "@/types/batik";

export interface FabricDetectionResult extends BoundingBox {
  framingMode: "swatch" | "center";
  presets: {
    swatch: BoundingBox;
    center: BoundingBox;
  };
}

export async function detectFabricRegion(
  canvas: HTMLCanvasElement
): Promise<FabricDetectionResult> {
  return computeAdaptiveFabricSaliency(canvas);
}

export function getFabricPresets(
  w: number,
  h: number
): { swatch: BoundingBox; center: BoundingBox } {
  // Swatch preset: 96% of frame with 2% margin to avoid border artifacts
  const swatchMarginX = Math.max(2, Math.round(w * 0.02));
  const swatchMarginY = Math.max(2, Math.round(h * 0.02));
  const swatchBox: BoundingBox = {
    x: swatchMarginX,
    y: swatchMarginY,
    width: Math.max(20, w - swatchMarginX * 2),
    height: Math.max(20, h - swatchMarginY * 2),
    confidence: 0.95,
    label: "Full Fabric Swatch",
  };

  // Center focus preset: 52% square focused on chest/torso motif area
  const centerSize = Math.max(20, Math.round(Math.min(w, h) * 0.52));
  const centerX = Math.round((w - centerSize) / 2);
  let centerY = Math.round(h * 0.54 - centerSize / 2);
  centerY = Math.max(0, Math.min(h - centerSize, centerY));

  const centerBox: BoundingBox = {
    x: Math.max(0, centerX),
    y: Math.max(0, centerY),
    width: centerSize,
    height: centerSize,
    confidence: 0.92,
    label: "Garment Motif Focus",
  };

  return { swatch: swatchBox, center: centerBox };
}

function computeAdaptiveFabricSaliency(
  canvas: HTMLCanvasElement
): FabricDetectionResult {
  const w = canvas.width;
  const h = canvas.height;
  const presets = getFabricPresets(w, h);

  const ctx = canvas.getContext("2d");
  if (!ctx || w < 20 || h < 20) {
    return {
      ...presets.swatch,
      framingMode: "swatch",
      presets,
    };
  }

  try {
    const imgData = ctx.getImageData(0, 0, w, h);
    const data = imgData.data;

    // Sample perimeter borders (outer 5%) and center region
    const bw = Math.max(2, Math.round(w * 0.05));
    const bh = Math.max(2, Math.round(h * 0.05));
    const step = Math.max(1, Math.floor(Math.min(w, h) / 100));

    let borderWhiteCount = 0;
    let borderDarkCount = 0;
    let borderSampleCount = 0;

    const borderLumas: number[] = [];
    const centerLumas: number[] = [];

    // Sample top and bottom borders
    for (let y = 0; y < bh; y += step) {
      for (let x = 0; x < w; x += step) {
        // Top border
        const idxTop = (y * w + x) * 4;
        const rT = data[idxTop];
        const gT = data[idxTop + 1];
        const bT = data[idxTop + 2];
        const lumaT = 0.299 * rT + 0.587 * gT + 0.114 * bT;
        borderLumas.push(lumaT);
        if (rT > 215 && gT > 215 && bT > 215) borderWhiteCount++;
        if (rT < 35 && gT < 35 && bT < 35) borderDarkCount++;
        borderSampleCount++;

        // Bottom border
        const yBot = h - 1 - y;
        const idxBot = (yBot * w + x) * 4;
        const rB = data[idxBot];
        const gB = data[idxBot + 1];
        const bB = data[idxBot + 2];
        const lumaB = 0.299 * rB + 0.587 * gB + 0.114 * bB;
        borderLumas.push(lumaB);
        if (rB > 215 && gB > 215 && bB > 215) borderWhiteCount++;
        if (rB < 35 && gB < 35 && bB < 35) borderDarkCount++;
        borderSampleCount++;
      }
    }

    // Sample left and right borders
    for (let x = 0; x < bw; x += step) {
      for (let y = bh; y < h - bh; y += step) {
        // Left border
        const idxLeft = (y * w + x) * 4;
        const rL = data[idxLeft];
        const gL = data[idxLeft + 1];
        const bL = data[idxLeft + 2];
        const lumaL = 0.299 * rL + 0.587 * gL + 0.114 * bL;
        borderLumas.push(lumaL);
        if (rL > 215 && gL > 215 && bL > 215) borderWhiteCount++;
        if (rL < 35 && gL < 35 && bL < 35) borderDarkCount++;
        borderSampleCount++;

        // Right border
        const xRight = w - 1 - x;
        const idxRight = (y * w + xRight) * 4;
        const rR = data[idxRight];
        const gR = data[idxRight + 1];
        const bR = data[idxRight + 2];
        const lumaR = 0.299 * rR + 0.587 * gR + 0.114 * bR;
        borderLumas.push(lumaR);
        if (rR > 215 && gR > 215 && bR > 215) borderWhiteCount++;
        if (rR < 35 && gR < 35 && bR < 35) borderDarkCount++;
        borderSampleCount++;
      }
    }

    // Sample center 50% region
    const cStartX = Math.round(w * 0.25);
    const cEndX = Math.round(w * 0.75);
    const cStartY = Math.round(h * 0.25);
    const cEndY = Math.round(h * 0.75);

    for (let y = cStartY; y < cEndY; y += step * 2) {
      for (let x = cStartX; x < cEndX; x += step * 2) {
        const idx = (y * w + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        centerLumas.push(0.299 * r + 0.587 * g + 0.114 * b);
      }
    }

    const calcStd = (arr: number[]): number => {
      if (arr.length === 0) return 0;
      const mean = arr.reduce((acc, v) => acc + v, 0) / arr.length;
      const variance = arr.reduce((acc, v) => acc + (v - mean) ** 2, 0) / arr.length;
      return Math.sqrt(variance);
    };

    const borderStd = calcStd(borderLumas);
    const centerStd = calcStd(centerLumas);
    const varianceRatio = borderStd / Math.max(1.0, centerStd);

    const whiteBgRatio = borderSampleCount > 0 ? borderWhiteCount / borderSampleCount : 0;
    const darkBgRatio = borderSampleCount > 0 ? borderDarkCount / borderSampleCount : 0;

    // Decision: If borders have significant uniform backdrop (white/dark studio background)
    // or significantly lower variance than center, image is a garment/mannequin.
    // Otherwise, image is an edge-to-edge fabric swatch.
    const isGarment = whiteBgRatio > 0.25 || darkBgRatio > 0.25 || varianceRatio < 0.45;
    const framingMode: "swatch" | "center" = isGarment ? "center" : "swatch";
    const chosenBox = isGarment ? presets.center : presets.swatch;

    return {
      ...chosenBox,
      framingMode,
      presets,
    };
  } catch {
    // Graceful fallback to swatch preset
    return {
      ...presets.swatch,
      framingMode: "swatch",
      presets,
    };
  }
}
