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

    const lumaAt = (px: number, py: number): number => {
      const cx = Math.max(0, Math.min(w - 1, px));
      const cy = Math.max(0, Math.min(h - 1, py));
      const idx = (cy * w + cx) * 4;
      return 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
    };

    // Sample perimeter borders (outer 6%) and center 50% region
    const bw = Math.max(2, Math.round(w * 0.06));
    const bh = Math.max(2, Math.round(h * 0.06));
    const step = Math.max(1, Math.floor(Math.min(w, h) / 100));

    const borderGradients: number[] = [];
    const borderLumas: number[] = [];
    const centerGradients: number[] = [];
    const centerLumas: number[] = [];

    // 1. Top and Bottom border strips
    for (let y = 0; y < bh; y += step) {
      for (let x = 0; x < w; x += step) {
        // Top
        const lumaT = lumaAt(x, y);
        borderLumas.push(lumaT);
        const gxT = Math.abs(lumaAt(x + 1, y) - lumaAt(x - 1, y));
        const gyT = Math.abs(lumaAt(x, y + 1) - lumaAt(x, y - 1));
        borderGradients.push(gxT + gyT);

        // Bottom
        const yB = h - 1 - y;
        const lumaB = lumaAt(x, yB);
        borderLumas.push(lumaB);
        const gxB = Math.abs(lumaAt(x + 1, yB) - lumaAt(x - 1, yB));
        const gyB = Math.abs(lumaAt(x, yB + 1) - lumaAt(x, yB - 1));
        borderGradients.push(gxB + gyB);
      }
    }

    // 2. Left and Right border strips
    for (let x = 0; x < bw; x += step) {
      for (let y = bh; y < h - bh; y += step) {
        // Left
        const lumaL = lumaAt(x, y);
        borderLumas.push(lumaL);
        const gxL = Math.abs(lumaAt(x + 1, y) - lumaAt(x - 1, y));
        const gyL = Math.abs(lumaAt(x, y + 1) - lumaAt(x, y - 1));
        borderGradients.push(gxL + gyL);

        // Right
        const xR = w - 1 - x;
        const lumaR = lumaAt(xR, y);
        borderLumas.push(lumaR);
        const gxR = Math.abs(lumaAt(xR + 1, y) - lumaAt(xR - 1, y));
        const gyR = Math.abs(lumaAt(xR, y + 1) - lumaAt(xR, y - 1));
        borderGradients.push(gxR + gyR);
      }
    }

    // 3. Center 50% region
    const cStartX = Math.round(w * 0.25);
    const cEndX = Math.round(w * 0.75);
    const cStartY = Math.round(h * 0.25);
    const cEndY = Math.round(h * 0.75);

    for (let y = cStartY; y < cEndY; y += step * 2) {
      for (let x = cStartX; x < cEndX; x += step * 2) {
        const lumaC = lumaAt(x, y);
        centerLumas.push(lumaC);
        const gxC = Math.abs(lumaAt(x + 1, y) - lumaAt(x - 1, y));
        const gyC = Math.abs(lumaAt(x, y + 1) - lumaAt(x, y - 1));
        centerGradients.push(gxC + gyC);
      }
    }

    const calcMean = (arr: number[]): number => {
      if (arr.length === 0) return 0;
      return arr.reduce((acc, v) => acc + v, 0) / arr.length;
    };

    const calcStd = (arr: number[]): number => {
      if (arr.length === 0) return 0;
      const mean = calcMean(arr);
      const variance = arr.reduce((acc, v) => acc + (v - mean) ** 2, 0) / arr.length;
      return Math.sqrt(variance);
    };

    const meanBorderGrad = calcMean(borderGradients);
    const borderStd = calcStd(borderLumas);
    const centerStd = calcStd(centerLumas);
    const varianceRatio = borderStd / Math.max(1.0, centerStd);

    // Decision:
    // A scene/garment photo (mannequin, model, studio shot) has a dead, uniform backdrop
    // with virtually no border edge gradients (< 6.5) and low border-to-center variance ratio (< 0.42).
    // An edge-to-edge fabric swatch (regardless of whether dark latar ireng or light mori ground)
    // has continuous wax canting lines and pattern transitions crossing the borders.
    const isGarment = (meanBorderGrad < 6.5) || (varianceRatio < 0.42 && meanBorderGrad < 11.0);
    const framingMode: "swatch" | "center" = isGarment ? "center" : "swatch";
    const chosenBox = isGarment ? presets.center : presets.swatch;

    return {
      ...chosenBox,
      framingMode,
      presets,
    };
  } catch {
    // Graceful fallback to full swatch preset
    return {
      ...presets.swatch,
      framingMode: "swatch",
      presets,
    };
  }
}
