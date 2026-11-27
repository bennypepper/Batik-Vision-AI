"use client";

import React, { useState } from "react";
import {
  PredictionResult,
  AppLanguage,
  ExecutionProvider,
} from "@/types/batik";
import { t } from "@/lib/i18n";
import { getMotifRegion, getMotifIsland } from "@/lib/labels-data";
import {
  MapPin,
  ChevronDown,
  ChevronUp,
  Bookmark,
  BookmarkCheck,
  ArrowLeft,
  AlertTriangle,
  Cpu,
} from "lucide-react";

interface ClassificationResultsProps {
  predictions: PredictionResult[];
  inferenceTimeMs: number;
  executionProvider: ExecutionProvider;
  isOutOfDistribution: boolean;
  croppedThumbnail: string;
  language: AppLanguage;
  onScanAgain: () => void;
  onSaveToHistory: () => void;
  onNavigateToAtlas?: (motifName: string) => void;
  isSaved?: boolean;
}

export const ClassificationResults: React.FC<ClassificationResultsProps> = ({
  predictions,
  inferenceTimeMs,
  executionProvider,
  isOutOfDistribution,
  croppedThumbnail,
  language,
  onScanAgain,
  onSaveToHistory,
  onNavigateToAtlas,
  isSaved = false,
}) => {
  const top1 = predictions[0];
  const candidates = predictions.slice(1, 5);
  const [showCandidates, setShowCandidates] = useState(false);

  if (!top1) return null;

  const motifName =
    language === "id" ? top1.classInfo.name_id : top1.classInfo.name_en;
  const philosophy =
    language === "id"
      ? top1.classInfo.description_id
      : top1.classInfo.description_en;

  const providerLabel =
    executionProvider === "webgpu"
      ? t(language, "provider_webgpu")
      : executionProvider === "webgl"
      ? t(language, "provider_webgl")
      : t(language, "provider_wasm");

  return (
    <div className="w-full max-w-4xl mx-auto space-y-4 pb-12">
      {/* Top Utility Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-[var(--on-canvas-faint)] px-1">
        <button
          type="button"
          onClick={onScanAgain}
          className="flex items-center gap-1.5 text-[var(--on-canvas-soft)] hover:text-[var(--on-canvas)] transition cursor-pointer bg-transparent border-none"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>{t(language, "scan_again")}</span>
        </button>
        <div className="flex items-center gap-2.5">
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[var(--canvas-deep)] border border-[var(--canvas-line-strong)] text-[11px] font-medium text-[var(--on-canvas-soft)] tnum">
            <Cpu className="w-3 h-3 text-[var(--soga)]" />
            <span>{providerLabel}</span>
            <span>·</span>
            <span>{inferenceTimeMs}ms</span>
          </span>
          <button
            type="button"
            onClick={onSaveToHistory}
            disabled={isSaved}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border transition cursor-pointer ${
              isSaved
                ? "bg-[rgba(237,230,216,0.08)] text-[var(--on-canvas-soft)] border-[var(--canvas-line-strong)]"
                : "bg-[var(--soga)] text-white border-[var(--soga)] hover:bg-[var(--soga-deep)]"
            }`}
          >
            {isSaved ? (
              <>
                <BookmarkCheck className="w-3.5 h-3.5" />
                <span>{t(language, "saved_to_history")}</span>
              </>
            ) : (
              <>
                <Bookmark className="w-3.5 h-3.5" />
                <span>{t(language, "save_history")}</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Out-Of-Distribution / Low Confidence Warning Card */}
      {isOutOfDistribution && (
        <div
          role="alert"
          className="bg-amber-500/10 border border-amber-500/35 text-[var(--on-canvas)] rounded-xl p-4 flex items-start gap-3"
        >
          <AlertTriangle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="text-[13.5px] font-semibold text-amber-600 dark:text-amber-300">
              {t(language, "ood_warning_title")}
            </h4>
            <p className="text-[12.5px] text-[var(--on-canvas-soft)] leading-relaxed">
              {t(language, "ood_warning_desc")}
            </p>
          </div>
        </div>
      )}

      {/* Main Result Card on Warm Cloth */}
      <div className="bg-[var(--cloth)] text-[var(--on-cloth)] rounded-xl p-5 sm:p-7 border border-[var(--cloth-line)] shadow-sm flex flex-col sm:flex-row gap-5 items-start">
        {/* Swatch & Input Thumbnail */}
        <div className="flex sm:flex-col items-center gap-3 shrink-0">
          <div
            className="w-20 h-20 rounded-lg overflow-hidden border border-[var(--cloth-line-strong)] bg-stone-900 shadow-sm"
            title="Exemplar Pattern"
          >
            <img
              src={`/motifs/${top1.classInfo.slug}.jpg`}
              alt={motifName}
              className="w-full h-full object-cover"
            />
          </div>
          {croppedThumbnail && (
            <div
              className="w-20 h-20 rounded-lg overflow-hidden border border-[var(--cloth-line-strong)] bg-stone-900 shadow-sm"
              title="Cropped Input Fabric"
            >
              <img
                src={croppedThumbnail}
                alt={motifName}
                className="w-full h-full object-cover"
              />
            </div>
          )}
        </div>

        {/* Content Details */}
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline justify-between gap-3 mb-1">
            <h3 className="text-[17px] font-semibold text-[var(--on-cloth)] truncate">
              {motifName}
            </h3>
            <span className="text-[13px] font-semibold text-[var(--soga)] tnum shrink-0">
              {top1.score.toFixed(1)}% {t(language, "conf_confident")}
            </span>
          </div>

          <div className="text-[12px] text-[var(--on-cloth-faint)] flex items-center gap-1 mb-2">
            <MapPin className="w-3.5 h-3.5 text-[var(--on-cloth-faint)] shrink-0" />
            <span>
              {getMotifRegion(top1.classInfo, language)} ·{" "}
              {getMotifIsland(top1.classInfo, language)}
            </span>
          </div>

          <p className="text-[12.5px] text-[var(--on-cloth-soft)] leading-relaxed mb-3.5">
            {philosophy}
          </p>

          <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-[var(--cloth-line)]">
            <button
              type="button"
              onClick={() => onNavigateToAtlas?.(motifName)}
              className="text-[12.5px] text-[var(--soga)] font-semibold hover:underline cursor-pointer bg-transparent border-none p-0"
            >
              {t(language, "see_in_atlas")}
            </button>

            {candidates.length > 0 && (
              <button
                type="button"
                onClick={() => setShowCandidates(!showCandidates)}
                className="text-[12px] text-[var(--on-cloth-soft)] hover:text-[var(--on-cloth)] flex items-center gap-1 cursor-pointer bg-transparent border-none p-0"
              >
                <span>{t(language, "top_candidates")}</span>
                {showCandidates ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Collapsible Candidate Breakdown */}
      {showCandidates && candidates.length > 0 && (
        <div className="bg-[var(--cloth)] text-[var(--on-cloth)] rounded-lg p-4 border border-[var(--cloth-line)] space-y-2.5">
          <div className="text-xs font-semibold text-[var(--on-cloth-soft)] uppercase tracking-wider mb-1">
            {t(language, "top_candidates")}
          </div>
          {candidates.map((cand, idx) => {
            const candName =
              language === "id"
                ? cand.classInfo.name_id
                : cand.classInfo.name_en;
            return (
              <div key={cand.rank || idx} className="space-y-1">
                <div className="flex justify-between text-xs font-medium">
                  <span className="text-[var(--on-cloth)]">
                    {candName}
                    <span className="text-[var(--on-cloth-faint)] text-[11px] ml-1.5">
                      ({getMotifRegion(cand.classInfo, language)})
                    </span>
                  </span>
                  <span className="text-[var(--soga)] font-semibold tnum">
                    {cand.score.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-1.5 bg-[var(--cloth-deep)] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[var(--soga)] rounded-full transition-all duration-300"
                    style={{
                      width: `${Math.min(100, Math.max(2, cand.score))}%`,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
