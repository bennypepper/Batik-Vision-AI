"use client";

import React, { useState } from "react";
import { PredictionResult, AppLanguage } from "@/types/batik";
import { t } from "@/lib/i18n";
import { CheckCircle2, MapPin, Sparkles, BookOpen, Bookmark, BookmarkCheck, ArrowLeft } from "lucide-react";

interface ClassificationResultsProps {
  predictions: PredictionResult[];
  inferenceTimeMs: number;
  croppedThumbnail: string;
  language: AppLanguage;
  onScanAgain: () => void;
  onSaveToHistory: () => void;
  isSaved?: boolean;
}

export const ClassificationResults: React.FC<ClassificationResultsProps> = ({
  predictions,
  inferenceTimeMs,
  croppedThumbnail,
  language,
  onScanAgain,
  onSaveToHistory,
  isSaved = false,
}) => {
  const top1 = predictions[0];
  const candidates = predictions.slice(1, 5);
  const [activeTab, setActiveTab] = useState<"philosophy" | "candidates">("philosophy");

  if (!top1) return null;

  const motifName = language === "id" ? top1.classInfo.name_id : top1.classInfo.name_en;
  const philosophy = language === "id" ? top1.classInfo.description_id : top1.classInfo.description_en;

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 animate-fade-in">
      {/* Top Banner & Timing */}
      <div className="flex items-center justify-between">
        <button
          onClick={onScanAgain}
          className="text-xs text-stone-400 hover:text-amber-400 flex items-center gap-1.5 transition font-medium"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>{t(language, "scan_again")}</span>
        </button>

        <div className="flex items-center gap-2">
          <span className="text-xs text-stone-500 font-mono">
            Inference: {inferenceTimeMs}ms
          </span>
          <button
            onClick={onSaveToHistory}
            disabled={isSaved}
            className={`px-3 py-1 text-xs font-medium rounded-lg border flex items-center gap-1.5 transition ${
              isSaved
                ? "bg-emerald-950/40 text-emerald-400 border-emerald-800/60"
                : "bg-stone-800 text-stone-300 hover:text-amber-300 border-stone-700"
            }`}
          >
            {isSaved ? (
              <>
                <BookmarkCheck className="w-3.5 h-3.5 text-emerald-400" />
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

      {/* Top-1 Hero Prediction Card */}
      <div className="bg-stone-900/90 rounded-2xl border border-stone-800 p-6 md:p-8 shadow-2xl relative overflow-hidden">
        {/* Subtle decorative background glow */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-amber-500/5 rounded-full blur-3xl pointer-events-none" />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
          {/* Cropped Motif Thumbnail */}
          <div className="flex flex-col items-center">
            <div className="relative w-44 h-44 rounded-xl overflow-hidden border-2 border-amber-500/40 shadow-xl bg-stone-950">
              <img
                src={croppedThumbnail}
                alt={motifName}
                className="w-full h-full object-cover"
              />
              <div className="absolute bottom-0 inset-x-0 bg-stone-950/80 backdrop-blur-sm text-[10px] text-center py-1 text-stone-400 font-mono">
                224x224 CNN Input
              </div>
            </div>
            {top1.classInfo.is_weak_class && (
              <span className="mt-2.5 inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-950/80 text-amber-300 border border-amber-700/60 shadow">
                <Sparkles className="w-3 h-3 text-amber-400" />
                {t(language, "weak_class_badge")}
              </span>
            )}
          </div>

          {/* Core Motif Details & Confidence */}
          <div className="md:col-span-2 space-y-4">
            <div className="space-y-1">
              <div className="text-xs font-semibold tracking-wider uppercase text-amber-400 flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-amber-400" />
                {t(language, "top_prediction")}
              </div>
              <h2 className="text-2xl md:text-3xl font-extrabold text-stone-100 tracking-tight">
                {motifName}
              </h2>
              <p className="text-sm font-mono text-stone-400">
                slug: {top1.classInfo.slug}
              </p>
            </div>

            {/* Geographical Tags */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-stone-800 text-stone-300 border border-stone-700">
                <MapPin className="w-3.5 h-3.5 text-amber-400" />
                {top1.classInfo.region}
              </span>
              <span className="px-3 py-1 rounded-lg text-xs font-medium bg-stone-800/80 text-stone-400 border border-stone-700/60">
                {t(language, "island")}: {top1.classInfo.island}
              </span>
            </div>

            {/* Confidence Progress Bar */}
            <div className="space-y-1.5 pt-2">
              <div className="flex justify-between text-xs font-medium">
                <span className="text-stone-400">{t(language, "confidence")}</span>
                <span className="text-amber-400 font-mono font-bold text-sm">
                  {top1.score}%
                </span>
              </div>
              <div className="w-full h-3 bg-stone-950 rounded-full overflow-hidden border border-stone-800 p-0.5">
                <div
                  className="h-full bg-gradient-to-r from-amber-600 via-amber-500 to-amber-400 rounded-full transition-all duration-700 shadow-sm"
                  style={{ width: `${Math.min(100, Math.max(5, top1.score))}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Philosophy and Candidate Distribution Tabs */}
      <div className="bg-stone-900/80 rounded-2xl border border-stone-800 overflow-hidden shadow-xl">
        <div className="flex border-b border-stone-800">
          <button
            onClick={() => setActiveTab("philosophy")}
            className={`flex-1 py-3 px-4 text-sm font-semibold flex items-center justify-center gap-2 transition ${
              activeTab === "philosophy"
                ? "bg-amber-500/10 text-amber-400 border-b-2 border-amber-500"
                : "text-stone-400 hover:text-stone-200"
            }`}
          >
            <BookOpen className="w-4 h-4" />
            <span>{t(language, "philosophy")}</span>
          </button>
          <button
            onClick={() => setActiveTab("candidates")}
            className={`flex-1 py-3 px-4 text-sm font-semibold flex items-center justify-center gap-2 transition ${
              activeTab === "candidates"
                ? "bg-amber-500/10 text-amber-400 border-b-2 border-amber-500"
                : "text-stone-400 hover:text-stone-200"
            }`}
          >
            <Sparkles className="w-4 h-4" />
            <span>{t(language, "top_candidates")}</span>
          </button>
        </div>

        <div className="p-6 md:p-8">
          {activeTab === "philosophy" ? (
            <div className="prose prose-invert max-w-none space-y-3">
              <p className="text-sm md:text-base leading-relaxed text-stone-300">
                {philosophy}
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <p className="text-xs text-stone-400 mb-2">
                {t(language, "top_candidates")}
              </p>
              {predictions.map((c) => {
                const cName = language === "id" ? c.classInfo.name_id : c.classInfo.name_en;
                return (
                  <div key={c.rank} className="space-y-1">
                    <div className="flex justify-between text-xs font-medium">
                      <span className="text-stone-300">
                        <span className="font-mono text-stone-500 mr-2">#{c.rank}</span>
                        {cName}
                        <span className="text-stone-500 ml-1.5 text-[11px]">
                          ({c.classInfo.region})
                        </span>
                      </span>
                      <span className="text-stone-400 font-mono font-bold">
                        {c.score}%
                      </span>
                    </div>
                    <div className="w-full h-2 bg-stone-950 rounded-full overflow-hidden border border-stone-800">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          c.rank === 1
                            ? "bg-amber-500"
                            : c.rank === 2
                            ? "bg-amber-700/80"
                            : "bg-stone-700"
                        }`}
                        style={{ width: `${Math.min(100, Math.max(3, c.score))}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
