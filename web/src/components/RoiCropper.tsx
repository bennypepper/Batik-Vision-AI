"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";
import { BoundingBox, AppLanguage } from "@/types/batik";
import { t } from "@/lib/i18n";
import { Crop, RotateCcw, Check, Sparkles } from "lucide-react";

interface RoiCropperProps {
  imageSrc: string;
  initialBox: BoundingBox;
  language: AppLanguage;
  onConfirmCrop: (croppedCanvas: HTMLCanvasElement) => void;
  onReset: () => void;
}

const MAX_CROP_SIDE = 1024;

export const RoiCropper: React.FC<RoiCropperProps> = ({
  imageSrc,
  initialBox,
  language,
  onConfirmCrop,
  onReset,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [naturalSize, setNaturalSize] = useState<{
    width: number;
    height: number;
  }>({ width: 0, height: 0 });
  const [displaySize, setDisplaySize] = useState<{
    width: number;
    height: number;
  }>({ width: 0, height: 0 });

  // Current crop box in natural image pixels
  const [box, setBox] = useState<BoundingBox>(initialBox);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState<string | null>(null);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({
    x: 0,
    y: 0,
  });

  // Update display dimensions
  const updateDisplaySize = useCallback(() => {
    if (imgRef.current && imgRef.current.clientWidth > 0) {
      setDisplaySize({
        width: imgRef.current.clientWidth,
        height: imgRef.current.clientHeight,
      });
    }
  }, []);

  const syncFromImage = useCallback(
    (img: HTMLImageElement) => {
      const nW = img.naturalWidth;
      const nH = img.naturalHeight;
      if (nW === 0 || nH === 0) return;

      setNaturalSize({ width: nW, height: nH });
      updateDisplaySize();

      if (!initialBox || initialBox.width === 0) {
        const size = Math.min(nW, nH) * 0.8;
        setBox({
          x: Math.round((nW - size) / 2),
          y: Math.round((nH - size) / 2),
          width: Math.round(size),
          height: Math.round(size),
          confidence: 0.9,
          label: "Batik Fabric ROI",
        });
      } else {
        setBox(initialBox);
      }
    },
    [initialBox, updateDisplaySize]
  );

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    syncFromImage(e.currentTarget);
  };

  useEffect(() => {
    if (imgRef.current && imgRef.current.complete && imgRef.current.naturalWidth > 0) {
      syncFromImage(imgRef.current);
    }
    window.addEventListener("resize", updateDisplaySize);
    return () => window.removeEventListener("resize", updateDisplaySize);
  }, [syncFromImage, updateDisplaySize]);

  const scale =
    naturalSize.width > 0 ? displaySize.width / naturalSize.width : 1;

  // Convert natural coords to display coords
  const displayBox = {
    x: box.x * scale,
    y: box.y * scale,
    width: box.width * scale,
    height: box.height * scale,
  };

  const handlePointerDown = (
    e: React.PointerEvent,
    handle?: "nw" | "se"
  ) => {
    e.preventDefault();
    if (handle) {
      setIsResizing(handle);
    } else {
      setIsDragging(true);
    }
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handlePointerMove = useCallback(
    (e: PointerEvent) => {
      if (!isDragging && !isResizing) return;
      if (scale === 0) return;

      const dx = (e.clientX - dragStart.x) / scale;
      const dy = (e.clientY - dragStart.y) / scale;

      setBox((prev) => {
        let newX = prev.x;
        let newY = prev.y;
        let newW = prev.width;
        let newH = prev.height;

        if (isDragging) {
          newX = Math.max(
            0,
            Math.min(naturalSize.width - prev.width, prev.x + dx)
          );
          newY = Math.max(
            0,
            Math.min(naturalSize.height - prev.height, prev.y + dy)
          );
        } else if (isResizing === "se") {
          const delta = Math.max(dx, dy);
          const size = Math.max(
            64,
            Math.min(
              naturalSize.width - prev.x,
              naturalSize.height - prev.y,
              prev.width + delta
            )
          );
          newW = size;
          newH = size;
        } else if (isResizing === "nw") {
          const delta = Math.min(dx, dy);
          const newSize = Math.max(64, prev.width - delta);
          if (
            prev.x + prev.width - newSize >= 0 &&
            prev.y + prev.height - newSize >= 0
          ) {
            newX = prev.x + prev.width - newSize;
            newY = prev.y + prev.height - newSize;
            newW = newSize;
            newH = newSize;
          }
        }

        return {
          ...prev,
          x: Math.round(newX),
          y: Math.round(newY),
          width: Math.round(newW),
          height: Math.round(newH),
        };
      });

      setDragStart({ x: e.clientX, y: e.clientY });
    },
    [isDragging, isResizing, scale, dragStart, naturalSize]
  );

  const handlePointerUp = useCallback(() => {
    setIsDragging(false);
    setIsResizing(null);
  }, []);

  useEffect(() => {
    if (isDragging || isResizing) {
      window.addEventListener("pointermove", handlePointerMove);
      window.addEventListener("pointerup", handlePointerUp);
      window.addEventListener("pointercancel", handlePointerUp);
      return () => {
        window.removeEventListener("pointermove", handlePointerMove);
        window.removeEventListener("pointerup", handlePointerUp);
        window.removeEventListener("pointercancel", handlePointerUp);
      };
    }
  }, [isDragging, isResizing, handlePointerMove, handlePointerUp]);

  const handleApplyCrop = () => {
    if (!imgRef.current) return;
    // Output at natural crop resolution capped at MAX_CROP_SIDE (never pre-downscale to 224x224)
    const srcW = Math.max(32, Math.round(box.width));
    const srcH = Math.max(32, Math.round(box.height));
    const downscale = Math.min(1, MAX_CROP_SIDE / Math.max(srcW, srcH));
    const outW = Math.max(32, Math.round(srcW * downscale));
    const outH = Math.max(32, Math.round(srcH * downscale));

    const canvas = document.createElement("canvas");
    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(
      imgRef.current,
      box.x,
      box.y,
      box.width,
      box.height,
      0,
      0,
      outW,
      outH
    );

    onConfirmCrop(canvas);
  };

  const handleAutoCropReset = () => {
    const size = Math.min(naturalSize.width, naturalSize.height) * 0.8;
    setBox({
      x: Math.round((naturalSize.width - size) / 2),
      y: Math.round((naturalSize.height - size) / 2),
      width: Math.round(size),
      height: Math.round(size),
      confidence: initialBox.confidence,
      label: "Batik Fabric ROI",
    });
  };

  return (
    <div className="w-full flex flex-col items-center">
      {/* Header instructions */}
      <div className="w-full mb-3 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-amber-300 flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-amber-400" />
            {t(language, "stage1_label")}
          </h3>
          <p className="text-xs text-stone-400">{t(language, "stage1_sub")}</p>
        </div>
        <div className="text-xs text-stone-400 font-mono bg-stone-900 px-2 py-1 rounded border border-stone-800">
          ROI: {box.width}x{box.height} px
        </div>
      </div>

      {/* Interactive Image & Cropper Area */}
      <div
        ref={containerRef}
        className="relative max-w-full max-h-[500px] overflow-hidden rounded-xl border border-stone-800 bg-stone-950 shadow-2xl select-none touch-none"
      >
        <img
          ref={imgRef}
          src={imageSrc}
          alt="Batik scan target"
          className="max-w-full max-h-[500px] object-contain block"
          onLoad={handleImageLoad}
          draggable={false}
        />

        {displaySize.width > 0 && (
          <>
            {/* Darkened overlay outside crop box */}
            <div
              className="absolute pointer-events-none inset-0"
              style={{
                boxShadow: `0 0 0 9999px rgba(0, 0, 0, 0.65)`,
                clipPath: `polygon(
                  0% 0%, 0% 100%, 100% 100%, 100% 0%,
                  ${displayBox.x}px 0%,
                  ${displayBox.x}px ${displayBox.y}px,
                  ${displayBox.x + displayBox.width}px ${displayBox.y}px,
                  ${displayBox.x + displayBox.width}px ${displayBox.y + displayBox.height}px,
                  ${displayBox.x}px ${displayBox.y + displayBox.height}px,
                  ${displayBox.x}px 0%
                )`,
              }}
            />

            {/* Draggable Crop Bounding Box */}
            <div
              className="absolute border-2 border-amber-400 shadow-[0_0_15px_rgba(245,158,11,0.5)] cursor-move transition-shadow touch-none"
              style={{
                left: `${displayBox.x}px`,
                top: `${displayBox.y}px`,
                width: `${displayBox.width}px`,
                height: `${displayBox.height}px`,
              }}
              onPointerDown={(e) => handlePointerDown(e)}
            >
              {/* Corner handles */}
              <div
                className="absolute -top-2.5 -left-2.5 w-5 h-5 bg-amber-400 border border-stone-950 rounded-sm cursor-nwse-resize shadow-md touch-none"
                onPointerDown={(e) => {
                  e.stopPropagation();
                  handlePointerDown(e, "nw");
                }}
              />
              <div
                className="absolute -bottom-2.5 -right-2.5 w-5 h-5 bg-amber-400 border border-stone-950 rounded-sm cursor-nwse-resize shadow-md touch-none"
                onPointerDown={(e) => {
                  e.stopPropagation();
                  handlePointerDown(e, "se");
                }}
              />

              {/* Grid guide lines */}
              <div className="absolute inset-0 grid grid-cols-3 grid-rows-3 pointer-events-none opacity-30">
                <div className="border-r border-b border-amber-300" />
                <div className="border-r border-b border-amber-300" />
                <div className="border-b border-amber-300" />
                <div className="border-r border-b border-amber-300" />
                <div className="border-r border-b border-amber-300" />
                <div className="border-b border-amber-300" />
                <div className="border-r border-amber-300" />
                <div className="border-r border-amber-300" />
                <div />
              </div>

              {/* Tag badge */}
              <div className="absolute top-1 left-1 bg-amber-600/90 text-stone-950 text-[10px] font-bold px-1.5 py-0.5 rounded shadow">
                {box.label || "Fabric ROI"}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Control Buttons */}
      <div className="w-full mt-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleAutoCropReset}
            className="px-3 py-2 text-xs font-medium text-stone-300 bg-stone-800 hover:bg-stone-700 rounded-lg border border-stone-700 transition flex items-center gap-1.5 cursor-pointer"
          >
            <Crop className="w-3.5 h-3.5 text-amber-400" />
            <span>{t(language, "btn_auto_crop")}</span>
          </button>
          <button
            type="button"
            onClick={onReset}
            className="px-3 py-2 text-xs font-medium text-stone-400 hover:text-stone-200 transition flex items-center gap-1 cursor-pointer bg-transparent border-none"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>{t(language, "btn_reset_crop")}</span>
          </button>
        </div>

        <button
          type="button"
          onClick={handleApplyCrop}
          className="px-5 py-2.5 bg-gradient-to-r from-amber-600 to-amber-700 hover:from-amber-500 hover:to-amber-600 text-stone-950 font-semibold text-sm rounded-lg shadow-lg shadow-amber-950/50 flex items-center gap-2 transition cursor-pointer border-none"
        >
          <Check className="w-4 h-4" />
          <span>{t(language, "btn_classify")}</span>
        </button>
      </div>
    </div>
  );
};
