import { describe, it, expect } from "vitest";
import {
  computeAdaptiveSaliencyFromPixels,
  getFabricPresets,
} from "@/lib/yolo-detector";

function createSolidRgba(
  w: number,
  h: number,
  r = 220,
  g = 220,
  b = 220
): Uint8ClampedArray {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = 255;
  }
  return data;
}

function paintHighFrequencyPattern(
  data: Uint8ClampedArray,
  w: number,
  x0: number,
  y0: number,
  rw: number,
  rh: number
) {
  for (let y = y0; y < y0 + rh; y++) {
    for (let x = x0; x < x0 + rw; x++) {
      const idx = (y * w + x) * 4;
      const stripe = ((x + y) % 4 < 2) ? 230 : 30;
      const cross = (x % 3 === 0) ? 200 : 40;
      const val = (x + y) % 2 === 0 ? stripe : cross;
      data[idx] = val;
      data[idx + 1] = Math.max(0, val - 25);
      data[idx + 2] = Math.max(0, val - 50);
      data[idx + 3] = 255;
    }
  }
}

describe("Stage-1 Adaptive Fabric Saliency Localizer", () => {
  it("returns valid swatch and center presets within image bounds", () => {
    const presets = getFabricPresets(800, 600);
    expect(presets.swatch.x).toBeGreaterThanOrEqual(0);
    expect(presets.swatch.y).toBeGreaterThanOrEqual(0);
    expect(presets.swatch.x + presets.swatch.width).toBeLessThanOrEqual(800);
    expect(presets.swatch.y + presets.swatch.height).toBeLessThanOrEqual(600);
    expect(presets.swatch.width).toBeGreaterThan(700);

    expect(presets.center.width).toBe(presets.center.height);
    expect(presets.center.width).toBe(300);
    expect(presets.center.x + presets.center.width).toBeLessThanOrEqual(800);
    expect(presets.center.y + presets.center.height).toBeLessThanOrEqual(600);
  });

  it("classifies a full-frame high-texture batik swatch as swatch mode", () => {
    const w = 240;
    const h = 240;
    const pixels = createSolidRgba(w, h);
    paintHighFrequencyPattern(pixels, w, 0, 0, w, h);

    const res = computeAdaptiveSaliencyFromPixels(pixels, w, h, 800, 800);
    expect(res.framingMode).toBe("swatch");
    expect(res.method).toBe("adaptive-saliency");
    expect(res.width).toBeGreaterThan(700);
    expect(res.confidence).toBeGreaterThan(0);
    expect(res.confidence).toBeLessThanOrEqual(1);
  });

  it("localizes a centered garment surrounded by plain studio backdrop", () => {
    const w = 240;
    const h = 240;
    const pixels = createSolidRgba(w, h, 240, 240, 240);
    // Textured garment in the center 40%
    paintHighFrequencyPattern(pixels, w, 72, 72, 96, 96);

    const res = computeAdaptiveSaliencyFromPixels(pixels, w, h, 1000, 1000);
    expect(res.framingMode).toBe("center");
    expect(res.method).toBe("adaptive-saliency");
    // Box should be narrower than a full swatch
    expect(res.width).toBeLessThan(850);
    // Center of detected box should be near center (500, 500)
    const cx = res.x + res.width / 2;
    const cy = res.y + res.height / 2;
    expect(cx).toBeGreaterThan(300);
    expect(cx).toBeLessThan(700);
    expect(cy).toBeGreaterThan(300);
    expect(cy).toBeLessThan(700);
  });

  it("localizes an off-center garment in the right-lower quadrant", () => {
    const w = 240;
    const h = 240;
    const pixels = createSolidRgba(w, h, 245, 245, 245);
    // Textured garment positioned clearly in lower-right (x: 130..230, y: 120..220)
    paintHighFrequencyPattern(pixels, w, 130, 120, 100, 100);

    const res = computeAdaptiveSaliencyFromPixels(pixels, w, h, 1200, 1200);
    expect(res.framingMode).toBe("center");
    expect(res.method).toBe("adaptive-saliency");
    const cx = res.x + res.width / 2;
    const cy = res.y + res.height / 2;
    // Center of detected region must shift clearly into the right-lower quadrant (> 650)
    expect(cx).toBeGreaterThan(650);
    expect(cy).toBeGreaterThan(600);
    expect(res.x + res.width).toBeLessThanOrEqual(1200);
    expect(res.y + res.height).toBeLessThanOrEqual(1200);
  });
});
