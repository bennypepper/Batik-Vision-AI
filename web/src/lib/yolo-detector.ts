import { BoundingBox } from "@/types/batik";

export type FabricLocalizationMethod = "adaptive-saliency" | "onnx-segmentation";

export interface FabricDetectionResult extends BoundingBox {
  framingMode: "swatch" | "center";
  method: FabricLocalizationMethod;
  presets: {
    swatch: BoundingBox;
    center: BoundingBox;
    adaptive: BoundingBox;
  };
}

const CANONICAL_MAX_SIDE = 256;
const GRID_SIZE = 6;

export async function detectFabricRegion(
  canvas: HTMLCanvasElement
): Promise<FabricDetectionResult> {
  const origW = canvas.width;
  const origH = canvas.height;
  const presets = getFabricPresets(origW, origH);

  if (origW < 20 || origH < 20) {
    const fallbackBox: BoundingBox = {
      ...presets.swatch,
      confidence: 0.5,
      label: "Full Fabric Swatch",
    };
    return {
      ...fallbackBox,
      framingMode: "swatch",
      method: "adaptive-saliency",
      presets: { ...presets, adaptive: fallbackBox },
    };
  }

  try {
    // Downscale to resolution-normalized 256px canonical grid for invariant saliency
    const scale = Math.min(1, CANONICAL_MAX_SIDE / Math.max(origW, origH));
    const sw = Math.max(24, Math.round(origW * scale));
    const sh = Math.max(24, Math.round(origH * scale));

    const offscreen = document.createElement("canvas");
    offscreen.width = sw;
    offscreen.height = sh;
    const ctx = offscreen.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      return {
        ...presets.swatch,
        framingMode: "swatch",
        method: "adaptive-saliency",
        presets: { ...presets, adaptive: presets.swatch },
      };
    }

    ctx.drawImage(canvas, 0, 0, sw, sh);
    const imgData = ctx.getImageData(0, 0, sw, sh);
    return computeAdaptiveSaliencyFromPixels(imgData.data, sw, sh, origW, origH);
  } catch {
    return {
      ...presets.swatch,
      framingMode: "swatch",
      method: "adaptive-saliency",
      presets: { ...presets, adaptive: presets.swatch },
    };
  }
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
    confidence: 1.0,
    label: "Full Fabric Swatch",
  };

  // Center focus preset: 50% square focused on torso/garment motif area
  const centerSize = Math.max(20, Math.round(Math.min(w, h) * 0.5));
  const centerX = Math.round((w - centerSize) / 2);
  let centerY = Math.round(h * 0.52 - centerSize / 2);
  centerY = Math.max(0, Math.min(h - centerSize, centerY));

  const centerBox: BoundingBox = {
    x: Math.max(0, centerX),
    y: Math.max(0, centerY),
    width: centerSize,
    height: centerSize,
    confidence: 1.0,
    label: "Garment Motif Focus",
  };

  return { swatch: swatchBox, center: centerBox };
}

/**
 * Resolution-normalized 6x6 grid Laplacian texture energy localizer.
 * Pure pixel-array helper so it can be unit-tested deterministically in Node/Vitest.
 */
export function computeAdaptiveSaliencyFromPixels(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  origW: number = w,
  origH: number = h
): FabricDetectionResult {
  const presets = getFabricPresets(origW, origH);

  if (w < 12 || h < 12 || rgba.length < w * h * 4) {
    const fallback: BoundingBox = { ...presets.swatch, confidence: 0.5 };
    return {
      ...fallback,
      framingMode: "swatch",
      method: "adaptive-saliency",
      presets: { ...presets, adaptive: fallback },
    };
  }

  // Convert RGBA to grayscale luminance
  const gray = new Float32Array(w * h);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    gray[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }

  // Compute 6x6 grid of Laplacian texture energy + luma std
  const cellEnergy = new Float32Array(GRID_SIZE * GRID_SIZE);
  const cellW = w / GRID_SIZE;
  const cellH = h / GRID_SIZE;

  for (let gy = 0; gy < GRID_SIZE; gy++) {
    const yStart = Math.max(1, Math.floor(gy * cellH));
    const yEnd = Math.min(h - 1, Math.ceil((gy + 1) * cellH));

    for (let gx = 0; gx < GRID_SIZE; gx++) {
      const xStart = Math.max(1, Math.floor(gx * cellW));
      const xEnd = Math.min(w - 1, Math.ceil((gx + 1) * cellW));

      let lapSum = 0;
      let lumaSum = 0;
      let lumaSqSum = 0;
      let count = 0;

      for (let y = yStart; y < yEnd; y++) {
        const rowOffset = y * w;
        for (let x = xStart; x < xEnd; x++) {
          const idx = rowOffset + x;
          const c = gray[idx];
          const lap = Math.abs(
            4 * c -
              gray[idx - 1] -
              gray[idx + 1] -
              gray[idx - w] -
              gray[idx + w]
          );
          lapSum += lap;
          lumaSum += c;
          lumaSqSum += c * c;
          count++;
        }
      }

      if (count > 0) {
        const meanLap = lapSum / count;
        const meanLuma = lumaSum / count;
        const variance = Math.max(0, lumaSqSum / count - meanLuma * meanLuma);
        const stdLuma = Math.sqrt(variance);
        cellEnergy[gy * GRID_SIZE + gx] = meanLap + 0.4 * stdLuma;
      }
    }
  }

  let maxEnergy = 0;
  let totalEnergy = 0;
  let borderEnergySum = 0;
  let borderCount = 0;

  for (let gy = 0; gy < GRID_SIZE; gy++) {
    for (let gx = 0; gx < GRID_SIZE; gx++) {
      const e = cellEnergy[gy * GRID_SIZE + gx];
      totalEnergy += e;
      if (e > maxEnergy) maxEnergy = e;
      if (
        gx === 0 ||
        gx === GRID_SIZE - 1 ||
        gy === 0 ||
        gy === GRID_SIZE - 1
      ) {
        borderEnergySum += e;
        borderCount++;
      }
    }
  }

  const meanAllEnergy = totalEnergy / (GRID_SIZE * GRID_SIZE);
  const meanBorderEnergy = borderCount > 0 ? borderEnergySum / borderCount : 0;

  // Flat edge-to-edge swatch: high texture energy continues across the outer border ring
  const isFullSwatch =
    maxEnergy < 4.0 ||
    (meanBorderEnergy >= maxEnergy * 0.52 && meanBorderEnergy >= 10.0);

  if (isFullSwatch) {
    const computedScore = clamp(
      0.5 + Math.min(0.46, meanAllEnergy / 140),
      0.25,
      0.96
    );
    const adaptiveSwatch: BoundingBox = {
      ...presets.swatch,
      confidence: Number(computedScore.toFixed(2)),
      label: "Full Fabric Swatch",
    };
    return {
      ...adaptiveSwatch,
      framingMode: "swatch",
      method: "adaptive-saliency",
      presets: {
        ...presets,
        adaptive: adaptiveSwatch,
      },
    };
  }

  // Garment / off-center fabric: find highest-energy contiguous 4-connected region
  const activeThreshold = Math.max(6.0, maxEnergy * 0.35);
  const visited = new Uint8Array(GRID_SIZE * GRID_SIZE);
  let bestComponent: number[] = [];
  let bestComponentEnergy = -1;

  for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
    if (visited[i] || cellEnergy[i] < activeThreshold) continue;
    const queue = [i];
    visited[i] = 1;
    const component: number[] = [];
    let compEnergy = 0;

    while (queue.length > 0) {
      const curr = queue.shift()!;
      component.push(curr);
      compEnergy += cellEnergy[curr];
      const cx = curr % GRID_SIZE;
      const cy = Math.floor(curr / GRID_SIZE);

      const neighbors = [
        [cx - 1, cy],
        [cx + 1, cy],
        [cx, cy - 1],
        [cx, cy + 1],
      ];
      for (const [nx, ny] of neighbors) {
        if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
          const nIdx = ny * GRID_SIZE + nx;
          if (!visited[nIdx] && cellEnergy[nIdx] >= activeThreshold) {
            visited[nIdx] = 1;
            queue.push(nIdx);
          }
        }
      }
    }

    if (compEnergy > bestComponentEnergy) {
      bestComponentEnergy = compEnergy;
      bestComponent = component;
    }
  }

  if (bestComponent.length === 0) {
    const fallbackCenter: BoundingBox = {
      ...presets.center,
      confidence: 0.45,
    };
    return {
      ...fallbackCenter,
      framingMode: "center",
      method: "adaptive-saliency",
      presets: {
        ...presets,
        adaptive: fallbackCenter,
      },
    };
  }

  let minGx = GRID_SIZE;
  let maxGx = 0;
  let minGy = GRID_SIZE;
  let maxGy = 0;
  for (const idx of bestComponent) {
    const gx = idx % GRID_SIZE;
    const gy = Math.floor(idx / GRID_SIZE);
    if (gx < minGx) minGx = gx;
    if (gx > maxGx) maxGx = gx;
    if (gy < minGy) minGy = gy;
    if (gy > maxGy) maxGy = gy;
  }

  const rawX = Math.floor((minGx / GRID_SIZE) * origW);
  const rawY = Math.floor((minGy / GRID_SIZE) * origH);
  const rawX2 = Math.ceil(((maxGx + 1) / GRID_SIZE) * origW);
  const rawY2 = Math.ceil(((maxGy + 1) / GRID_SIZE) * origH);

  const x = clamp(rawX, 0, Math.max(0, origW - 32));
  const y = clamp(rawY, 0, Math.max(0, origH - 32));
  const width = clamp(rawX2 - x, Math.min(64, origW - x), origW - x);
  const height = clamp(rawY2 - y, Math.min(64, origH - y), origH - y);

  const compMean = bestComponentEnergy / bestComponent.length;
  const bgCells = GRID_SIZE * GRID_SIZE - bestComponent.length;
  const bgMean =
    bgCells > 0 ? Math.max(0, (totalEnergy - bestComponentEnergy) / bgCells) : 0;
  const contrast = (compMean - bgMean) / Math.max(1, compMean);
  const confidence = Number(clamp(0.4 + contrast * 0.55, 0.35, 0.96).toFixed(2));

  const adaptiveBox: BoundingBox = {
    x,
    y,
    width,
    height,
    confidence,
    label: "Adaptive Fabric ROI",
  };

  return {
    ...adaptiveBox,
    framingMode: "center",
    method: "adaptive-saliency",
    presets: {
      ...presets,
      adaptive: adaptiveBox,
    },
  };
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}
