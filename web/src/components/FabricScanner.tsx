"use client";

import React, { useState, useRef, useCallback } from "react";
import { AppLanguage, BoundingBox, PredictionResult } from "@/types/batik";
import { t } from "@/lib/i18n";
import { RoiCropper } from "./RoiCropper";
import { ClassificationResults } from "./ClassificationResults";
import { detectFabricRegion } from "@/lib/yolo-detector";
import { runInferenceOnCanvas } from "@/lib/classifier";
import { Upload, Camera, Loader2, AlertCircle, ArrowRight, Crop } from "lucide-react";

interface FabricScannerProps {
  language: AppLanguage;
  onSaveScanRecord: (record: {
    thumbnail: string;
    predictions: PredictionResult[];
    inferenceTimeMs: number;
  }) => void;
  onNavigateToAtlas?: (motifName: string) => void;
}

export const FabricScanner: React.FC<FabricScannerProps> = ({
  language,
  onSaveScanRecord,
  onNavigateToAtlas,
}) => {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [initialBox, setInitialBox] = useState<BoundingBox | null>(null);
  const [isManualCropping, setIsManualCropping] = useState(false);
  const [status, setStatus] = useState<"idle" | "detecting" | "cropping" | "classifying" | "done">("idle");
  const [predictions, setPredictions] = useState<PredictionResult[] | null>(null);
  const [croppedThumbnail, setCroppedThumbnail] = useState<string | null>(null);
  const [inferenceTimeMs, setInferenceTimeMs] = useState(0);
  const [isSaved, setIsSaved] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = async () => {
    try {
      setErrorMsg(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsCameraActive(true);
    } catch (err) {
      console.error("Camera access failed:", err);
      // Fallback: trigger file input if camera is not available or blocked
      fileInputRef.current?.click();
    }
  };

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
  }, []);

  const capturePhoto = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
    stopCamera();
    processUploadedImage(dataUrl);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
      if (ev.target?.result) {
        processUploadedImage(ev.target.result as string);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        if (ev.target?.result) {
          processUploadedImage(ev.target.result as string);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const processUploadedImage = async (dataUrl: string) => {
    setImageSrc(dataUrl);
    setPredictions(null);
    setIsSaved(false);
    setErrorMsg(null);
    setIsManualCropping(false);
    setStatus("detecting");

    const img = new Image();
    img.src = dataUrl;
    img.onload = async () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(img, 0, 0);
        // Stage 1: YOLO fabric ROI detection
        const detectedBox = await detectFabricRegion(canvas);
        setInitialBox(detectedBox);
      }
      setStatus("cropping");
    };
  };

  const executeClassificationOnBox = async (boxToCrop: BoundingBox) => {
    if (!imageSrc) return;
    setStatus("classifying");

    const img = new Image();
    img.src = imageSrc;
    img.onload = async () => {
      const canvas = document.createElement("canvas");
      canvas.width = boxToCrop.width;
      canvas.height = boxToCrop.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(
        img,
        boxToCrop.x,
        boxToCrop.y,
        boxToCrop.width,
        boxToCrop.height,
        0,
        0,
        boxToCrop.width,
        boxToCrop.height
      );

      const thumbData = canvas.toDataURL("image/jpeg", 0.9);
      setCroppedThumbnail(thumbData);

      try {
        // Stage 2: EfficientNetV2S CNN inference
        const result = await runInferenceOnCanvas(canvas);
        setPredictions(result.predictions);
        setInferenceTimeMs(result.inferenceTimeMs);
        setStatus("done");
      } catch (err: any) {
        console.error("Classification error:", err);
        setErrorMsg(err?.message || "Classification failed");
        setStatus("cropping");
      }
    };
  };

  const handleConfirmCrop = async (croppedCanvas: HTMLCanvasElement) => {
    setStatus("classifying");
    setIsManualCropping(false);
    const thumbData = croppedCanvas.toDataURL("image/jpeg", 0.9);
    setCroppedThumbnail(thumbData);

    try {
      // Stage 2: EfficientNetV2S CNN inference
      const result = await runInferenceOnCanvas(croppedCanvas);
      setPredictions(result.predictions);
      setInferenceTimeMs(result.inferenceTimeMs);
      setStatus("done");
    } catch (err: any) {
      console.error("Classification error:", err);
      setErrorMsg(err?.message || "Classification failed");
      setStatus("cropping");
    }
  };

  const handleScanAgain = () => {
    setImageSrc(null);
    setPredictions(null);
    setInitialBox(null);
    setIsManualCropping(false);
    setStatus("idle");
    stopCamera();
  };

  const handleSaveToHistory = () => {
    if (predictions && croppedThumbnail && !isSaved) {
      onSaveScanRecord({
        thumbnail: croppedThumbnail,
        predictions,
        inferenceTimeMs,
      });
      setIsSaved(true);
    }
  };

  return (
    <div className="w-full max-w-[1180px] mx-auto pb-12 px-4">
      {/* Hero Header */}
      <div className="text-center pt-8 pb-3 max-w-[760px] mx-auto">
        <h1 className="display text-2xl sm:text-3xl font-semibold text-[var(--on-canvas)] mb-2.5">
          {t(language, "scanner_title")}
        </h1>
        <p className="text-[13.5px] text-[var(--on-canvas-soft)] leading-relaxed">
          {t(language, "scanner_desc")}
        </p>
      </div>

      {/* Two-Stage Progress Indicator */}
      <div className="flex justify-center items-center gap-2.5 my-6 text-[12px] text-[var(--on-canvas-faint)]">
        <div
          className={`flex items-center gap-2 ${
            imageSrc ? "text-[var(--on-canvas)] font-semibold" : ""
          }`}
        >
          <span
            className={`w-5 h-5 rounded-full flex items-center justify-center text-[11px] border transition ${
              imageSrc
                ? "bg-[var(--soga)] border-[var(--soga)] text-white"
                : "border-[var(--canvas-line-strong)]"
            }`}
          >
            1
          </span>
          <span>{t(language, "step_yolo")}</span>
        </div>
        <ArrowRight className="w-3.5 h-3.5 text-[var(--canvas-line-strong)]" />
        <div
          className={`flex items-center gap-2 ${
            predictions ? "text-[var(--on-canvas)] font-semibold" : ""
          }`}
        >
          <span
            className={`w-5 h-5 rounded-full flex items-center justify-center text-[11px] border transition ${
              predictions
                ? "bg-[var(--soga)] border-[var(--soga)] text-white"
                : "border-[var(--canvas-line-strong)]"
            }`}
          >
            2
          </span>
          <span>{t(language, "step_classifier")}</span>
        </div>
      </div>

      {errorMsg && (
        <div className="max-w-[760px] mx-auto mb-4 p-3.5 rounded-lg bg-[rgba(122,46,34,0.15)] border border-[rgba(122,46,34,0.35)] text-red-300 text-xs flex items-center gap-2.5">
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Live Camera View */}
      {isCameraActive && (
        <div className="max-w-[760px] mx-auto bg-[var(--canvas-deep)] p-4 rounded-xl border border-[var(--canvas-line-strong)] space-y-4 text-center my-4">
          <div className="relative rounded-lg overflow-hidden bg-black max-w-lg mx-auto aspect-[4/3]">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 border-2 border-dashed border-[var(--soga)] m-8 rounded-md pointer-events-none" />
          </div>
          <div className="flex items-center justify-center gap-3">
            <button
              onClick={capturePhoto}
              className="px-5 py-2.5 bg-[var(--soga)] hover:bg-[var(--soga-deep)] text-white font-semibold text-xs rounded-md transition flex items-center gap-2 cursor-pointer border-none"
            >
              <Camera className="w-4 h-4" />
              <span>{t(language, "btn_capture")}</span>
            </button>
            <button
              onClick={stopCamera}
              className="px-4 py-2.5 bg-transparent border border-[var(--canvas-line-strong)] hover:bg-[rgba(237,230,216,0.05)] text-[var(--on-canvas-soft)] text-xs rounded-md transition cursor-pointer"
            >
              {t(language, "btn_stop_camera")}
            </button>
          </div>
        </div>
      )}

      {/* Dropzone (Idle State) */}
      {status === "idle" && !isCameraActive && (
        <div className="space-y-5">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="max-w-[760px] mx-auto bg-[var(--cloth)] text-[var(--on-cloth)] border-[1.5px] border-dashed border-[var(--soga)] rounded-[10px] py-11 px-5 text-center cursor-pointer transition hover:bg-[var(--cloth-deep)]"
          >
            <div className="w-[52px] h-[52px] rounded-full bg-[var(--soga-tint)] text-[var(--soga)] flex items-center justify-center mx-auto mb-4">
              <Upload className="w-5 h-5" />
            </div>
            <h4 className="text-[15px] font-semibold text-[var(--on-cloth)] mb-1.5">
              {t(language, "upload_prompt")}
            </h4>
            <p className="text-[12px] text-[var(--on-cloth-faint)]">
              {t(language, "upload_sub")}
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-center gap-3">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-transparent border border-[var(--canvas-line-strong)] text-[var(--on-canvas)] text-[13px] font-medium px-5 py-2.5 rounded-md hover:bg-[rgba(237,230,216,0.05)] flex items-center gap-2 cursor-pointer transition"
            >
              <Upload className="w-3.5 h-3.5 text-[var(--on-canvas-soft)]" />
              <span>{t(language, "btn_browse")}</span>
            </button>
            <button
              onClick={startCamera}
              className="bg-[var(--soga)] text-white text-[13px] font-semibold px-5 py-2.5 rounded-md hover:bg-[var(--soga-deep)] flex items-center gap-2 cursor-pointer transition border-none"
            >
              <Camera className="w-3.5 h-3.5" />
              <span>{t(language, "btn_camera")}</span>
            </button>
          </div>
        </div>
      )}

      {/* Detecting / Classifying Loader */}
      {(status === "detecting" || status === "classifying") && (
        <div className="max-w-[760px] mx-auto py-12 px-6 text-center space-y-3 bg-[var(--canvas-deep)] rounded-xl border border-[var(--canvas-line-strong)]">
          <Loader2 className="w-7 h-7 text-[var(--soga)] animate-spin mx-auto" />
          <p className="text-sm font-medium text-[var(--on-canvas)]">
            {status === "detecting"
              ? t(language, "status_detecting_fabric")
              : t(language, "status_classifying")}
          </p>
        </div>
      )}

      {/* Stage 1 Complete: Preview with YOLO Bounding Box Overlay */}
      {status === "cropping" && imageSrc && initialBox && !isManualCropping && (
        <div className="max-w-[760px] mx-auto space-y-4">
          <div className="relative rounded-lg overflow-hidden bg-stone-900 border border-[var(--cloth-line)]">
            <img
              src={imageSrc}
              alt="Uploaded batik preview"
              className="w-full h-auto block"
            />
            {/* Simulated / Real YOLO bounding box */}
            <div
              className="absolute border-2 border-dashed border-[var(--soga)] rounded pointer-events-none"
              style={{
                left: `${(initialBox.x / (initialBox.width + initialBox.x * 2 || 1)) * 100}%`,
                top: `${(initialBox.y / (initialBox.height + initialBox.y * 2 || 1)) * 100}%`,
                width: `${(initialBox.width / (initialBox.width + initialBox.x * 2 || 1)) * 100}%`,
                height: `${(initialBox.height / (initialBox.height + initialBox.y * 2 || 1)) * 100}%`,
              }}
            >
              <span className="absolute -top-6 left-0 bg-[var(--soga)] text-white text-[10.5px] font-semibold px-2 py-0.5 rounded-t whitespace-nowrap shadow">
                {t(language, "fabric_detected")} · {(initialBox.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
            <button
              onClick={handleScanAgain}
              className="text-xs text-[var(--on-canvas-soft)] hover:text-[var(--on-canvas)] bg-transparent border-none cursor-pointer"
            >
              {t(language, "scan_again")}
            </button>

            <div className="flex items-center gap-2.5">
              <button
                onClick={() => setIsManualCropping(true)}
                className="bg-transparent border border-[var(--canvas-line-strong)] text-[var(--on-canvas-soft)] hover:text-[var(--on-canvas)] text-xs font-medium px-3.5 py-2 rounded-md transition flex items-center gap-1.5 cursor-pointer"
              >
                <Crop className="w-3.5 h-3.5" />
                <span>{t(language, "stage1_label")}</span>
              </button>

              <button
                onClick={() => executeClassificationOnBox(initialBox)}
                className="bg-[var(--soga)] hover:bg-[var(--soga-deep)] text-white text-xs font-semibold px-4 py-2 rounded-md transition flex items-center gap-1.5 cursor-pointer border-none shadow-sm"
              >
                <span>{t(language, "btn_classify")}</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Manual Fine-Tuning ROI Cropper */}
      {status === "cropping" && imageSrc && initialBox && isManualCropping && (
        <RoiCropper
          imageSrc={imageSrc}
          initialBox={initialBox}
          language={language}
          onConfirmCrop={handleConfirmCrop}
          onReset={() => setIsManualCropping(false)}
        />
      )}

      {/* Results State (Stage 2 Complete) */}
      {status === "done" && predictions && croppedThumbnail && (
        <ClassificationResults
          predictions={predictions}
          inferenceTimeMs={inferenceTimeMs}
          croppedThumbnail={croppedThumbnail}
          language={language}
          onScanAgain={handleScanAgain}
          onSaveToHistory={handleSaveToHistory}
          onNavigateToAtlas={onNavigateToAtlas}
          isSaved={isSaved}
        />
      )}
    </div>
  );
};
