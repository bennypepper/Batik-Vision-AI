"use client";

import React from "react";
import { ScanRecord, AppLanguage } from "@/types/batik";
import { t } from "@/lib/i18n";

interface ScanHistoryProps {
  records: ScanRecord[];
  language: AppLanguage;
  onClearHistory: () => void;
  onNavigateToAtlas?: (motifName: string) => void;
}

// Fallback sample records matching the mockup if local storage is empty
const SAMPLE_RECORDS: Array<{
  id: string;
  name: string;
  region: string;
  timeStr: { id: string; en: string };
  confidence: number;
  swatchIdx: number;
  query: string;
}> = [
  {
    id: "sample_1",
    name: "Yogyakarta Kawung",
    region: "Yogyakarta",
    timeStr: { id: "Hari ini, 14:22 · Yogyakarta", en: "Today, 14:22 · Yogyakarta" },
    confidence: 92.4,
    swatchIdx: 2,
    query: "Kawung",
  },
  {
    id: "sample_2",
    name: "Solo Parang",
    region: "Solo / Surakarta",
    timeStr: { id: "Kemarin, 09:07 · Solo / Surakarta", en: "Yesterday, 09:07 · Solo / Surakarta" },
    confidence: 88.1,
    swatchIdx: 1,
    query: "Parang Solo",
  },
  {
    id: "sample_3",
    name: "Ceplok",
    region: "Jawa Tengah",
    timeStr: { id: "3 hari lalu, 18:45 · Jawa Tengah", en: "3 days ago, 18:45 · Central Java" },
    confidence: 95.0,
    swatchIdx: 3,
    query: "Ceplok",
  },
  {
    id: "sample_4",
    name: "Sogan",
    region: "Solo / Surakarta",
    timeStr: { id: "5 hari lalu, 11:30 · Solo / Surakarta", en: "5 days ago, 11:30 · Solo / Surakarta" },
    confidence: 61.7,
    swatchIdx: 5,
    query: "Sogan",
  },
];

export const ScanHistory: React.FC<ScanHistoryProps> = ({
  records,
  language,
  onClearHistory,
  onNavigateToAtlas,
}) => {
  const hasUserRecords = records.length > 0;
  const count = hasUserRecords ? records.length : SAMPLE_RECORDS.length;

  return (
    <div className="w-full max-w-[760px] mx-auto pb-16 px-4">
      {/* Hero Header */}
      <div className="text-center pt-8 pb-4">
        <h1 className="display text-2xl sm:text-3xl font-semibold text-[var(--on-canvas)] mb-2.5">
          {t(language, "history_title")}
        </h1>
        <p className="text-[13.5px] text-[var(--on-canvas-soft)] max-w-[620px] mx-auto leading-relaxed">
          {t(language, "history_desc")}
        </p>
      </div>

      {/* Header Bar */}
      <div className="flex items-center justify-between mt-6 mb-3">
        <h3 className="text-[14.5px] font-semibold text-[var(--on-canvas)]">
          {count} {t(language, "saved_scans")}
        </h3>
        {hasUserRecords && (
          <button
            onClick={onClearHistory}
            className="text-[12px] text-[var(--on-canvas-soft)] hover:text-[var(--on-canvas)] bg-transparent border border-[var(--canvas-line-strong)] hover:bg-[rgba(237,230,216,0.05)] px-3 py-1.5 rounded-md transition cursor-pointer"
          >
            {t(language, "clear_history")}
          </button>
        )}
      </div>

      {/* History List */}
      <div className="bg-[var(--cloth)] text-[var(--on-cloth)] rounded-lg overflow-hidden border border-[var(--cloth-line)]">
        {hasUserRecords
          ? records.map((rec, index) => {
              const top = rec.topPrediction;
              const name = language === "id" ? top.classInfo.name_id : top.classInfo.name_en;
              const swatchClass = `swatch-${(top.classInfo.id % 5) + 1}`;
              const dateStr = new Date(rec.timestamp).toLocaleString(
                language === "id" ? "id-ID" : "en-US",
                { dateStyle: "short", timeStyle: "short" }
              );

              return (
                <div
                  key={rec.id || index}
                  className="flex items-center gap-3.5 px-4 sm:px-5 py-3.5 border-b border-[var(--cloth-line)] last:border-b-0 hover:bg-[rgba(36,31,22,0.02)] transition"
                >
                  <div className={`w-9 h-9 rounded-md shrink-0 ${swatchClass}`} />
                  <div className="min-w-0 flex-1">
                    <div className="text-[13.5px] font-semibold text-[var(--on-cloth)] truncate">
                      {name}
                    </div>
                    <div className="text-[11.5px] text-[var(--on-cloth-faint)] truncate">
                      {dateStr} · {top.classInfo.region}
                    </div>
                  </div>
                  <span className="text-[12.5px] font-semibold text-[var(--soga)] tnum ml-auto shrink-0">
                    {top.score.toFixed(1)}%
                  </span>
                  <button
                    onClick={() => onNavigateToAtlas?.(name)}
                    className="text-[12px] font-semibold text-[var(--soga)] hover:underline ml-3 shrink-0 bg-transparent border-none cursor-pointer"
                  >
                    {t(language, "view_detail_link")}
                  </button>
                </div>
              );
            })
          : SAMPLE_RECORDS.map((sample) => (
              <div
                key={sample.id}
                className="flex items-center gap-3.5 px-4 sm:px-5 py-3.5 border-b border-[var(--cloth-line)] last:border-b-0 hover:bg-[rgba(36,31,22,0.02)] transition"
              >
                <div className={`w-9 h-9 rounded-md shrink-0 swatch-${sample.swatchIdx}`} />
                <div className="min-w-0 flex-1">
                  <div className="text-[13.5px] font-semibold text-[var(--on-cloth)] truncate">
                    {sample.name}
                  </div>
                  <div className="text-[11.5px] text-[var(--on-cloth-faint)] truncate">
                    {language === "id" ? sample.timeStr.id : sample.timeStr.en}
                  </div>
                </div>
                <span className="text-[12.5px] font-semibold text-[var(--soga)] tnum ml-auto shrink-0">
                  {sample.confidence.toFixed(1)}%
                </span>
                <button
                  onClick={() => onNavigateToAtlas?.(sample.query)}
                  className="text-[12px] font-semibold text-[var(--soga)] hover:underline ml-3 shrink-0 bg-transparent border-none cursor-pointer"
                >
                  {t(language, "view_detail_link")}
                </button>
              </div>
            ))}
      </div>
    </div>
  );
};
