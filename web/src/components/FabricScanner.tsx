"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  AppLanguage,
  BoundingBox,
  ExecutionProvider,
  PredictionResult,
} from "@/types/batik";
import { t } from "@/lib/i18n";
import { RoiCropper } from "./RoiCropper";
import { ClassificationResults } from "./ClassificationResults";
import { detectFabricRegion, getFabricPresets } from "@/lib/yolo-detector";
import {
  runInferenceOnCanvas,
  resetInferenceWorker,
  ModelLoadError,
} from "@/lib/classifier";
import {
  Upload,
  Camera,
  Loader2,
  AlertCircle,
  ArrowRight,
  Crop,
  RotateCcw,
} from "lucide-react";

const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB
const MAX_IMAGE_SIDE = 1024;

interface FabricScannerProps {
  language: AppLanguage;
  onSaveScanRecord: (record: {
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
  const [naturalDimensions, setNaturalDimensions] = useState<{
    width: number;
    height: number;
  }>({ width: 0, height: 0 });
  const [framingMode, setFramingMode] = useState<"swatch" | "center">("swatch");
  const [presets, setPresets] = useState<{
    swatch: BoundingBox;
    center: BoundingBox;
    adaptive: BoundingBox;
  } | null>(null);
  const [isManualCropping, setIsManualCropping] = useState(false);
  const [status, setStatus] = useState<
    "idle" | "detecting" | "cropping" | "classifying" | "done"
  >("idle");
  const [predictions, setPredictions] = useState<PredictionResult[] | null>(
    null
  );
  const [croppedThumbnail, setCroppedThumbnail] = useState<string | null>(null);
  const [inferenceTimeMs, setInferenceTimeMs] = useState(0);
  const [executionProvider, setExecutionProvider] =
    useState<ExecutionProvider>("wasm");
  const [isOutOfDistribution, setIsOutOfDistribution] = useState(false);
  const [loadProgress, setLoadProgress] = useState<number>(0);
  const [loadStage, setLoadStage] = useState<
    "downloading" | "initializing" | "inferring"
  >("inferring");
  const [isSaved, setIsSaved] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isModelError, setIsModelError] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lastCropCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  }, []);

  // Attach MediaStream to <video> once the camera view mounts
  useEffect(() => {
    if (isCameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(() => {
        // Autoplay may be deferred until user interaction
      });
    }
  }, [isCameraActive]);

  // Always release hardware camera tracks on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const startCamera = async () => {
    try {
      setErrorMsg(null);
      setIsModelError(false);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      streamRef.current = stream;
      setIsCameraActive(true);
    } catch (err) {
      console.error("Camera access failed:", err);
      fileInputRef.current?.click();
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    stopCamera();
    processUploadedImage(dataUrl);
  };

  const validateAndReadFile = (file: File) => {
    setErrorMsg(null);
    setIsModelError(false);

    if (!file.type.startsWith("image/")) {
      setErrorMsg(t(language, "invalid_file_type"));
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setErrorMsg(t(language, "file_too_large"));
      return;
    }

    const reader = new FileReader();
    reader.onload = (ev) => {
      if (typeof ev.target?.result === "string") {
        processUploadedImage(ev.target.result);
      }
    };
    reader.onerror = () => {
      setErrorMsg(t(language, "image_load_failed"));
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // Reset value so uploading the same file twice still triggers onChange
    e.target.value = "";
    if (!file) return;
    validateAndReadFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      validateAndReadFile(file);
    }
  };

  const handleDropzoneKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  };

  const processUploadedImage = async (dataUrl: string) => {
    setPredictions(null);
    setIsSaved(false);
    setErrorMsg(null);
    setIsModelError(false);
    setIsManualCropping(false);
    setStatus("detecting");

    const img = new Image();
    let executed = false;

    const handleLoaded = async () => {
      if (executed) return;
      executed = true;

      const rawW = img.naturalWidth || 640;
      const rawH = img.naturalHeight || 640;

      // Downscale giant uploads to max 1024px side (prevents iOS Safari canvas crashes)
      const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(rawW, rawH));
      const normW = Math.max(32, Math.round(rawW * scale));
      const normH = Math.max(32, Math.round(rawH * scale));

      const canvas = document.createElement("canvas");
      canvas.width = normW;
      canvas.height = normH;
      const ctx = canvas.getContext("2d");

      let normalizedSrc = dataUrl;
      if (ctx) {
        ctx.drawImage(img, 0, 0, normW, normH);
        if (scale < 1) {
          normalizedSrc = canvas.toDataURL("image/jpeg", 0.92);
        }
        setImageSrc(normalizedSrc);
        setNaturalDimensions({ width: normW, height: normH });

        // Stage 1: Adaptive Texture Localization
        const detectedBox = await detectFabricRegion(canvas);
        setInitialBox(detectedBox);
        setFramingMode(detectedBox.framingMode);
        setPresets(detectedBox.presets);
      } else {
        setImageSrc(dataUrl);
        setNaturalDimensions({ width: rawW, height: rawH });
      }

      setStatus("cropping");
    };

    const handleError = () => {
      if (executed) return;
      executed = true;
      setErrorMsg(t(language, "image_load_failed"));
      setStatus("idle");
    };

    img.onload = handleLoaded;
    img.onerror = handleError;
    img.src = dataUrl;
    if (img.complete && img.naturalWidth > 0) {
      handleLoaded();
    }
  };

  const handleSelectPreset = (mode: "swatch" | "center") => {
    setFramingMode(mode);
    if (presets) {
      setInitialBox(mode === "center" ? presets.adaptive : presets.swatch);
    } else if (naturalDimensions.width > 0 && naturalDimensions.height > 0) {
      const p = getFabricPresets(
        naturalDimensions.width,
        naturalDimensions.height
      );
      setInitialBox(p[mode]);
    }
  };

  const classifyCanvas = async (canvas: HTMLCanvasElement) => {
    lastCropCanvasRef.current = canvas;
    setStatus("classifying");
    setErrorMsg(null);
    setIsModelError(false);
    setLoadProgress(0);
    setLoadStage("downloading");

    const thumbData = canvas.toDataURL("image/jpeg", 0.85);
    setCroppedThumbnail(thumbData);

    try {
      const result = await runInferenceOnCanvas(canvas, {
        onProgress: (percent, stage) => {
          setLoadProgress(percent);
          setLoadStage(stage);
        },
      });
      setPredictions(result.predictions);
      setInferenceTimeMs(result.inferenceTimeMs);
      setExecutionProvider(result.executionProvider);
      setIsOutOfDistribution(result.isOutOfDistribution);
      setStatus("done");

      // Persist lightweight scan record without full-res base64 thumbnail
      onSaveScanRecord({
        predictions: result.predictions,
        inferenceTimeMs: result.inferenceTimeMs,
      });
      setIsSaved(true);
    } catch (err: unknown) {
      console.error("Classification error:", err);
      setIsModelError(err instanceof ModelLoadError);
      setErrorMsg(
        err instanceof Error ? err.message : "Classification failed"
      );
      setStatus("cropping");
    }
  };

  const executeClassificationOnBox = async (boxToCrop?: BoundingBox | null) => {
    if (!imageSrc || status === "classifying") return;
    setStatus("classifying");

    const img = new Image();
    let executed = false;

    const runInference = async () => {
      if (executed) return;
      executed = true;

      const box: BoundingBox = boxToCrop || {
        x: 0,
        y: 0,
        width: img.naturalWidth || 480,
        height: img.naturalHeight || 480,
        label: "Fabric",
        confidence: 1.0,
      };

      const canvas = document.createElement("canvas");
      canvas.width = Math.max(32, Math.round(box.width));
      canvas.height = Math.max(32, Math.round(box.height));
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        setStatus("cropping");
        return;
      }

      ctx.drawImage(
        img,
        box.x,
        box.y,
        box.width,
        box.height,
        0,
        0,
        canvas.width,
        canvas.height
      );

      await classifyCanvas(canvas);
    };

    const handleError = () => {
      if (executed) return;
      executed = true;
      setErrorMsg(t(language, "image_load_failed"));
      setStatus("cropping");
    };

    img.onload = runInference;
    img.onerror = handleError;
    img.src = imageSrc;
    if (img.complete && img.naturalWidth > 0) {
      runInference();
    }
  };

  const handleConfirmCrop = async (croppedCanvas: HTMLCanvasElement) => {
    setIsManualCropping(false);
    await classifyCanvas(croppedCanvas);
  };

  const handleRetryModel = async () => {
    resetInferenceWorker();
    if (lastCropCanvasRef.current) {
      await classifyCanvas(lastCropCanvasRef.current);
    } else if (initialBox) {
      await executeClassificationOnBox(initialBox);
    }
  };

  const handleScanAgain = () => {
    setImageSrc(null);
    setPredictions(null);
    setInitialBox(null);
    setIsManualCropping(false);
    setErrorMsg(null);
    setIsModelError(false);
    lastCropCanvasRef.current = null;
    setStatus("idle");
    stopCamera();
  };

  const handleSaveToHistory = () => {
    if (predictions && !isSaved) {
      onSaveScanRecord({
        predictions,
        inferenceTimeMs,
      });
      setIsSaved(true);
    }
  };

  const loadSampleImage = async (path: string) => {
    try {
      setErrorMsg(null);
      setIsModelError(false);
      const res = await fetch(path);
      if (!res.ok) throw new Error("Sample image unavailable");
      const blob = await res.blob();
      const reader = new FileReader();
      reader.onload = (ev) => {
        if (typeof ev.target?.result === "string") {
          processUploadedImage(ev.target.result);
        }
      };
      reader.readAsDataURL(blob);
    } catch (err) {
      console.error("Failed to load sample image:", err);
      setErrorMsg(t(language, "image_load_failed"));
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto pb-16 px-4">
      {/* Hero Header */}
      <div className="text-center pt-4 pb-3 max-w-[800px] mx-auto">
        <h1 className="display text-2xl sm:text-4xl font-semibold text-[var(--on-canvas)] mb-2.5">
          {t(language, "scanner_title")}
        </h1>
        <p className="text-[14px] text-[var(--on-canvas-soft)] leading-relaxed">
          {t(language, "scanner_desc")}
        </p>
      </div>

      {/* Two-Stage Progress Indicator */}
      <div className="flex justify-center items-center gap-2.5 my-6 text-[12.5px] text-[var(--on-canvas-faint)]">
        <div
          className={`flex items-center gap-2 ${
            imageSrc ? "text-[var(--on-canvas)] font-semibold" : ""
          }`}
        >
          <span
            className={`w-5 h-5 rounded-full flex items-center justify-center text-[11px] border transition ${
              imageSrc
                ? "bg-[var(--soga)] border-[var(--soga)] text-white font-bold"
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
                ? "bg-[var(--soga)] border-[var(--soga)] text-white font-bold"
                : "border-[var(--canvas-line-strong)]"
            }`}
          >
            2
          </span>
          <span>{t(language, "step_classifier")}</span>
        </div>
      </div>

      {errorMsg && (
        <div
          role="alert"
          className="max-w-4xl mx-auto mb-4 p-4 rounded-xl bg-[var(--mahogany-tint)] border border-[var(--mahogany)] text-[var(--on-canvas)] text-xs flex flex-wrap items-center justify-between gap-3"
        >
          <div className="flex items-start gap-2.5 min-w-0 flex-1">
            <AlertCircle className="w-4 h-4 text-[var(--mahogany)] shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              {isModelError && (
                <div className="font-semibold text-[var(--mahogany)]">
                  {t(language, "model_error_title")}
                </div>
              )}
              <div className="text-[var(--on-canvas-soft)] break-words">
                {errorMsg}
              </div>
            </div>
          </div>
          {isModelError && imageSrc && (
            <button
              type="button"
              onClick={handleRetryModel}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-[var(--soga)] hover:bg-[var(--soga-deep)] text-white text-xs font-semibold transition cursor-pointer border-none shrink-0"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>{t(language, "model_error_retry")}</span>
            </button>
          )}
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
        <div className="max-w-4xl mx-auto bg-[var(--canvas-deep)] p-5 rounded-2xl border border-[var(--canvas-line-strong)] space-y-4 text-center my-4 shadow-md">
          <div className="relative rounded-xl overflow-hidden bg-black max-w-xl mx-auto aspect-[4/3]">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 border-2 border-dashed border-[var(--soga)] m-8 rounded-lg pointer-events-none" />
          </div>
          <div className="flex items-center justify-center gap-3">
            <button
              type="button"
              onClick={capturePhoto}
              className="px-5 py-2.5 bg-[var(--soga)] hover:bg-[var(--soga-deep)] text-white font-semibold text-xs rounded-md transition flex items-center gap-2 cursor-pointer border-none shadow"
            >
              <Camera className="w-4 h-4" />
              <span>{t(language, "btn_capture")}</span>
            </button>
            <button
              type="button"
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
        <div className="max-w-4xl mx-auto space-y-6">
          <div
            role="button"
            tabIndex={0}
            onKeyDown={handleDropzoneKeyDown}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="w-full bg-[var(--cloth)] text-[var(--on-cloth)] border-2 border-dashed border-[var(--soga)] rounded-2xl py-14 sm:py-16 px-6 text-center cursor-pointer transition hover:bg-[var(--cloth-deep)] focus:outline-none focus:ring-2 focus:ring-[var(--soga)] shadow-sm"
          >
            <div className="w-14 h-14 rounded-full bg-[var(--soga-tint)] text-[var(--soga)] flex items-center justify-center mx-auto mb-4 border border-[rgba(161,90,46,0.25)]">
              <Upload className="w-6 h-6" />
            </div>
            <h4 className="text-[17px] font-semibold text-[var(--on-cloth)] mb-2">
              {t(language, "upload_prompt")}
            </h4>
            <p className="text-[13px] text-[var(--on-cloth-faint)]">
              {t(language, "upload_sub")}
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-center gap-3.5">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="bg-transparent border border-[var(--canvas-line-strong)] text-[var(--on-canvas)] text-[13.5px] font-medium px-6 py-3 rounded-lg hover:bg-[rgba(237,230,216,0.05)] flex items-center gap-2 cursor-pointer transition"
            >
              <Upload className="w-4 h-4 text-[var(--on-canvas-soft)]" />
              <span>{t(language, "btn_browse")}</span>
            </button>
            <button
              type="button"
              onClick={startCamera}
              className="bg-[var(--soga)] text-white text-[13.5px] font-semibold px-6 py-3 rounded-lg hover:bg-[var(--soga-deep)] flex items-center gap-2 cursor-pointer transition border-none shadow"
            >
              <Camera className="w-4 h-4" />
              <span>{t(language, "btn_camera")}</span>
            </button>
          </div>

          {/* Quick Sample Motifs Bar */}
          <div className="pt-2 text-center">
            <span className="text-[12px] text-[var(--on-canvas-faint)] block mb-2.5">
              {language === "id"
                ? "Atau uji coba langsung dengan contoh motif berikut:"
                : "Or test immediately with these authentic sample motifs:"}
            </span>
            <div className="flex flex-wrap justify-center gap-2.5">
              {[
                {
                  name: "Megamendung",
                  path: "/motifs/Jawa_Barat_Megamendung.jpg",
                },
                { name: "Kawung", path: "/motifs/Yogyakarta_Kawung.jpg" },
                { name: "Parang Solo", path: "/motifs/Solo_Parang.jpg" },
                { name: "Ceplok", path: "/motifs/Ceplok.jpg" },
                { name: "Lasem", path: "/motifs/Lasem.jpg" },
              ].map((sample) => (
                <button
                  type="button"
                  key={sample.name}
                  onClick={() => loadSampleImage(sample.path)}
                  className="flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-[var(--canvas-line-strong)] hover:border-[var(--soga)] bg-[var(--canvas-deep)] hover:bg-[rgba(161,90,46,0.1)] text-[12px] text-[var(--on-canvas-soft)] hover:text-[var(--on-canvas)] transition cursor-pointer"
                >
                  <img
                    src={sample.path}
                    alt={sample.name}
                    className="w-4 h-4 rounded-full object-cover shrink-0"
                  />
                  <span>{sample.name}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Detecting / Classifying Loader with Real Download Progress */}
      {(status === "detecting" || status === "classifying") && (
        <div className="max-w-4xl mx-auto py-12 px-6 text-center space-y-3 bg-[var(--canvas-deep)] rounded-xl border border-[var(--canvas-line-strong)]">
          <Loader2 className="w-7 h-7 text-[var(--soga)] animate-spin mx-auto" />
          <p className="text-sm font-medium text-[var(--on-canvas)]">
            {status === "detecting"
              ? t(language, "status_detecting_fabric")
              : loadStage === "downloading"
              ? t(language, "status_downloading_model")
              : loadStage === "initializing"
              ? t(language, "status_loading_model")
              : t(language, "status_classifying")}
          </p>
          {status === "classifying" && loadStage === "downloading" && (
            <div className="max-w-xs mx-auto space-y-1.5 pt-1">
              <div className="w-full h-2 bg-[var(--cloth-deep)] rounded-full overflow-hidden border border-[var(--canvas-line-strong)]">
                <div
                  className="h-full bg-[var(--soga)] transition-all duration-200"
                  style={{ width: `${Math.max(4, loadProgress)}%` }}
                />
              </div>
              <div className="text-[11px] text-[var(--on-canvas-faint)] tnum">
                {loadProgress}%
              </div>
            </div>
          )}
        </div>
      )}

      {/* Stage 1 Complete: Preview with Adaptive Saliency Bounding Box Overlay */}
      {status === "cropping" && imageSrc && initialBox && !isManualCropping && (
        <div className="max-w-4xl mx-auto space-y-4">
          {/* Preset Framing Quick-Toggles */}
          <div className="flex flex-wrap items-center justify-between gap-3 p-2.5 rounded-xl bg-[var(--cloth)] border border-[var(--cloth-line)] shadow-sm">
            <div className="flex items-center gap-2">
              <span className="text-[12px] font-medium text-[var(--on-cloth-soft)] pl-1">
                {language === "id" ? "Bidikan Pola:" : "Pattern Framing:"}
              </span>
              <div className="inline-flex rounded-lg p-1 bg-[var(--cloth-deep)] border border-[var(--cloth-line)]">
                <button
                  type="button"
                  onClick={() => handleSelectPreset("swatch")}
                  className={`px-3 py-1 rounded-md text-[12px] font-medium transition cursor-pointer border-none flex items-center gap-1.5 ${
                    framingMode === "swatch"
                      ? "bg-[var(--soga)] text-white shadow-xs"
                      : "text-[var(--on-cloth-soft)] hover:text-[var(--on-cloth)] bg-transparent"
                  }`}
                >
                  <span>{t(language, "preset_full_swatch")}</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleSelectPreset("center")}
                  className={`px-3 py-1 rounded-md text-[12px] font-medium transition cursor-pointer border-none flex items-center gap-1.5 ${
                    framingMode === "center"
                      ? "bg-[var(--soga)] text-white shadow-xs"
                      : "text-[var(--on-cloth-soft)] hover:text-[var(--on-cloth)] bg-transparent"
                  }`}
                >
                  <span>{t(language, "preset_focus_center")}</span>
                </button>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setIsManualCropping(true)}
              className="bg-transparent border border-[var(--canvas-line-strong)] text-[var(--on-canvas)] hover:border-[var(--soga)] text-xs font-medium px-3 py-1.5 rounded-md transition flex items-center gap-1.5 cursor-pointer"
            >
              <Crop className="w-3.5 h-3.5 text-[var(--soga)]" />
              <span>{t(language, "btn_adjust_roi")}</span>
            </button>
          </div>

          <div className="relative rounded-lg overflow-hidden bg-stone-900 border border-[var(--cloth-line)]">
            <img
              src={imageSrc}
              alt="Uploaded batik preview"
              className="w-full h-auto block"
            />
            {/* Adaptive Saliency bounding box */}
            <div
              className="absolute border-2 border-dashed border-[var(--soga)] rounded pointer-events-none transition-all duration-300"
              style={{
                left: `${
                  (initialBox.x / (naturalDimensions.width || 1)) * 100
                }%`,
                top: `${
                  (initialBox.y / (naturalDimensions.height || 1)) * 100
                }%`,
                width: `${
                  (initialBox.width / (naturalDimensions.width || 1)) * 100
                }%`,
                height: `${
                  (initialBox.height / (naturalDimensions.height || 1)) * 100
                }%`,
              }}
            >
              <span className="absolute -top-6 left-0 bg-[var(--soga)] text-white text-[10.5px] font-semibold px-2 py-0.5 rounded-t whitespace-nowrap shadow">
                {t(language, "fabric_detected")} ·{" "}
                {(initialBox.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Contextual Framing Guidance Hint */}
          <div className="flex items-center gap-2.5 p-3 rounded-lg bg-[var(--cloth-deep)] border border-[var(--cloth-line)] text-[12.5px] text-[var(--on-cloth-soft)]">
            <Crop className="w-4 h-4 text-[var(--soga)] shrink-0" />
            <span>
              {framingMode === "swatch"
                ? t(language, "hint_swatch_detected")
                : t(language, "hint_garment_detected")}
            </span>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
            <button
              type="button"
              onClick={handleScanAgain}
              className="text-xs text-[var(--on-canvas-soft)] hover:text-[var(--on-canvas)] bg-transparent border-none cursor-pointer"
            >
              {t(language, "scan_again")}
            </button>

            <button
              type="button"
              onClick={() => executeClassificationOnBox(initialBox)}
              className="bg-[var(--soga)] hover:bg-[var(--soga-deep)] text-white text-xs font-semibold px-5 py-2.5 rounded-md transition flex items-center gap-2 cursor-pointer border-none shadow-sm"
            >
              <span>{t(language, "btn_classify")}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
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
          executionProvider={executionProvider}
          isOutOfDistribution={isOutOfDistribution}
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
