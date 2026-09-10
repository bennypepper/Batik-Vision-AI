"use client";

import React, { useState, useRef, useCallback } from "react";
import { AppLanguage, BoundingBox, PredictionResult } from "@/types/batik";
import { t } from "@/lib/i18n";
import { RoiCropper } from "./RoiCropper";
import { ClassificationResults } from "./ClassificationResults";
import { detectFabricRegion } from "@/lib/yolo-detector";
import { runInferenceOnCanvas } from "@/lib/classifier";
import { Upload, Camera, Loader2, Sparkles, AlertCircle } from "lucide-react";

interface FabricScannerProps {
  language: AppLanguage;
  onSaveScanRecord: (record: {
    thumbnail: string;
    predictions: PredictionResult[];
    inferenceTimeMs: number;
  }) => void;
}

export const FabricScanner: React.FC<FabricScannerProps> = ({
  language,
  onSaveScanRecord,
}) => {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [initialBox, setInitialBox] = useState<BoundingBox | null>(null);
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
      setErrorMsg(language === "id" ? "Gagal mengakses kamera. Periksa izin peramban." : "Failed to access camera. Check browser permissions.");
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

  const handleConfirmCrop = async (croppedCanvas: HTMLCanvasElement) => {
    setStatus("classifying");
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
    <div className="w-full max-w-4xl mx-auto space-y-6">
      {/* Intro Header */}
      {status === "idle" && (
        <div className="text-center space-y-2 py-4">
          <h2 className="text-2xl sm:text-3xl font-extrabold text-stone-100 tracking-tight">
            {t(language, "scanner_title")}
          </h2>
          <p className="text-sm text-stone-400 max-w-xl mx-auto">
            {t(language, "scanner_desc")}
          </p>
        </div>
      )}

      {errorMsg && (
        <div className="p-4 rounded-xl bg-red-950/50 border border-red-800/60 text-red-300 text-sm flex items-center gap-2.5">
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Input Selection State (Idle) */}
      {status === "idle" && !isCameraActive && (
        <div className="space-y-4">
          {/* Drag and Drop Zone */}
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-stone-700 hover:border-amber-500/80 bg-stone-900/50 hover:bg-stone-900/80 rounded-2xl p-10 text-center cursor-pointer transition-all duration-200 group flex flex-col items-center justify-center min-h-[260px]"
          >
            <div className="w-16 h-16 rounded-full bg-stone-800 group-hover:bg-amber-500/10 flex items-center justify-center text-stone-400 group-hover:text-amber-400 mb-4 transition border border-stone-700 group-hover:border-amber-500/30">
              <Upload className="w-7 h-7" />
            </div>
            <h3 className="text-base font-semibold text-stone-200 group-hover:text-amber-200 mb-1">
              {t(language, "upload_prompt")}
            </h3>
            <p className="text-xs text-stone-500">{t(language, "upload_sub")}</p>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-center gap-3">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2.5 rounded-xl bg-stone-800 hover:bg-stone-700 text-stone-200 text-sm font-semibold border border-stone-700 transition flex items-center gap-2 shadow-sm"
            >
              <Upload className="w-4 h-4 text-amber-400" />
              <span>{t(language, "btn_browse")}</span>
            </button>
            <button
              onClick={startCamera}
              className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-amber-600 to-amber-700 hover:from-amber-500 hover:to-amber-600 text-stone-950 text-sm font-semibold transition flex items-center gap-2 shadow-lg shadow-amber-950/40"
            >
              <Camera className="w-4 h-4" />
              <span>{t(language, "btn_camera")}</span>
            </button>
          </div>
        </div>
      )}

      {/* Live Camera View */}
      {isCameraActive && (
        <div className="bg-stone-900 p-4 rounded-2xl border border-stone-800 space-y-4 text-center">
          <div className="relative rounded-xl overflow-hidden bg-black max-w-lg mx-auto aspect-[4/3]">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full h-full object-cover"
            />
            {/* Guide overlay */}
            <div className="absolute inset-0 border-2 border-dashed border-amber-400/60 m-8 rounded-lg pointer-events-none" />
          </div>
          <div className="flex items-center justify-center gap-3">
            <button
              onClick={capturePhoto}
              className="px-6 py-2.5 bg-amber-600 hover:bg-amber-500 text-stone-950 font-bold text-sm rounded-xl shadow-lg transition flex items-center gap-2"
            >
              <Camera className="w-4 h-4" />
              <span>{t(language, "btn_capture")}</span>
            </button>
            <button
              onClick={stopCamera}
              className="px-4 py-2.5 bg-stone-800 hover:bg-stone-700 text-stone-300 text-sm rounded-xl transition"
            >
              {t(language, "btn_stop_camera")}
            </button>
          </div>
        </div>
      )}

      {/* Processing State Indicator */}
      {(status === "detecting" || status === "classifying") && (
        <div className="p-8 text-center space-y-3 bg-stone-900/60 rounded-2xl border border-stone-800">
          <Loader2 className="w-8 h-8 text-amber-500 animate-spin mx-auto" />
          <p className="text-sm font-medium text-stone-200">
            {status === "detecting"
              ? t(language, "status_detecting_fabric")
              : t(language, "status_classifying")}
          </p>
        </div>
      )}

      {/* Interactive ROI Cropper (Stage 1 Complete, ready to crop) */}
      {status === "cropping" && imageSrc && initialBox && (
        <RoiCropper
          imageSrc={imageSrc}
          initialBox={initialBox}
          language={language}
          onConfirmCrop={handleConfirmCrop}
          onReset={handleScanAgain}
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
          isSaved={isSaved}
        />
      )}
    </div>
  );
};
