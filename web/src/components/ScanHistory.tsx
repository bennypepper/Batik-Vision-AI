"use client";

import React from "react";
import { ScanRecord, AppLanguage } from "@/types/batik";
import { t } from "@/lib/i18n";
import { ArrowRight, Trash2, Clock } from "lucide-react";

interface ScanHistoryProps {
  records: ScanRecord[];
  language: AppLanguage;
  onClearHistory: () => void;
  onNavigateToAtlas?: (motifName: string) => void;
}

// Fallback sample records using authentic slugs
const SAMPLE_RECORDS: Array<{
  id: string;
  name: string;
  slug: string;
  region: string;
  timeStr: { id: string; en: string };
  confidence: number;
  query: string;
}> = [
  {
    id: "sample_1",
    name: "Yogyakarta Kawung",
    slug: "Yogyakarta_Kawung",
    region: "Yogyakarta",
    timeStr: { id: "Hari ini, 14:22 · Yogyakarta", en: "Today, 14:22 · Yogyakarta" },
    confidence: 92.4,
    query: "Kawung",
  },
  {
    id: "sample_2",
    name: "Solo Parang",
    slug: "Solo_Parang",
    region: "Solo / Surakarta",
    timeStr: { id: "Kemarin, 09:07 · Solo / Surakarta", en: "Yesterday, 09:07 · Solo / Surakarta" },
    confidence: 88.1,
    query: "Parang",
  },
  {
    id: "sample_3",
    name: "Ceplok",
    slug: "Ceplok",
    region: "Jawa Tengah",
    timeStr: { id: "3 hari lalu, 18:45 · Jawa Tengah", en: "3 days ago, 18:45 · Central Java" },
    confidence: 95.0,
    query: "Ceplok",
  },
  {
    id: "sample_4",
    name: "Sogan",
    slug: "Sogan",
    region: "Solo / Surakarta",
    timeStr: { id: "5 hari lalu, 11:30 · Solo / Surakarta", en: "5 days ago, 11:30 · Solo / Surakarta" },
    confidence: 61.7,
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
    <div className="w-full max-w-5xl mx-auto pb-16 px-4">
      {/* Hero Header */}
      <div className="text-center pt-4 pb-4">
        <h1 className="display text-2xl sm:text-4xl font-semibold text-[var(--on-canvas)] mb-2.5">
          {t(language, "history_title")}
        </h1>
        <p className="text-[14px] text-[var(--on-canvas-soft)] max-w-[640px] mx-auto leading-relaxed">
          {t(language, "history_desc")}
        </p>
      </div>

      {/* Header Bar */}
      <div className="flex items-center justify-between mt-6 mb-3 px-1">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-[var(--soga)]" />
          <h3 className="text-[14.5px] font-semibold text-[var(--on-canvas)]">
            {count} {t(language, "saved_scans")}
          </h3>
        </div>
        {hasUserRecords && (
          <button
            onClick={onClearHistory}
            className="text-[12px] text-[var(--on-canvas-soft)] hover:text-red-400 bg-transparent border border-[var(--canvas-line-strong)] hover:border-red-400/40 px-3 py-1.5 rounded-md transition flex items-center gap-1.5 cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>{t(language, "clear_history")}</span>
          </button>
        )}
      </div>

      {/* History List */}
      <div className="bg-[var(--cloth)] text-[var(--on-cloth)] rounded-xl overflow-hidden border border-[var(--cloth-line)] shadow-sm">
        {hasUserRecords
          ? records.map((rec, index) => {
              const top = rec.topPrediction;
              const name = language === "id" ? top.classInfo.name_id : top.classInfo.name_en;
              const dateStr = new Date(rec.timestamp).toLocaleString(
                language === "id" ? "id-ID" : "en-US",
                { dateStyle: "short", timeStyle: "short" }
              );

              return (
                <div
                  key={rec.id || index}
                  className="flex items-center gap-4 px-5 py-4 border-b border-[var(--cloth-line)] last:border-b-0 hover:bg-[rgba(36,31,22,0.02)] transition"
                >
                  {/* Swatch Image */}
                  <div className="w-12 h-12 rounded-lg overflow-hidden border border-[var(--cloth-line)] bg-stone-900 shrink-0">
                    <img
                      src={`/motifs/${top.classInfo.slug}.jpg`}
                      alt={name}
                      className="w-full h-full object-cover"
                    />
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="text-[14.5px] font-semibold text-[var(--on-cloth)] truncate">
                      {name}
                    </div>
                    <div className="text-[12px] text-[var(--on-cloth-faint)] truncate mt-0.5">
                      {dateStr} · {top.classInfo.region} ({top.classInfo.island})
                    </div>
                  </div>

                  <div className="text-[13.5px] font-semibold text-[var(--soga)] tnum shrink-0 text-right">
                    {top.score.toFixed(1)}%
                  </div>

                  <button
                    onClick={() => onNavigateToAtlas?.(name)}
                    className="text-[12px] font-medium text-[var(--soga)] hover:underline shrink-0 flex items-center gap-1 bg-transparent border-none cursor-pointer pl-2"
                  >
                    <span>{t(language, "view_detail_link")}</span>
                    <ArrowRight className="w-3 h-3" />
                  </button>
                </div>
              );
            })
          : SAMPLE_RECORDS.map((sample) => (
              <div
                key={sample.id}
                className="flex items-center gap-4 px-5 py-4 border-b border-[var(--cloth-line)] last:border-b-0 hover:bg-[rgba(36,31,22,0.02)] transition"
              >
                {/* Authentic Swatch */}
                <div className="w-12 h-12 rounded-lg overflow-hidden border border-[var(--cloth-line)] bg-stone-900 shrink-0">
                  <img
                    src={`/motifs/${sample.slug}.jpg`}
                    alt={sample.name}
                    className="w-full h-full object-cover"
                  />
                </div>

                <div className="min-w-0 flex-1">
                  <div className="text-[14.5px] font-semibold text-[var(--on-cloth)] truncate">
                    {sample.name}
                  </div>
                  <div className="text-[12px] text-[var(--on-cloth-faint)] truncate mt-0.5">
                    {language === "id" ? sample.timeStr.id : sample.timeStr.en}
                  </div>
                </div>

                <div className="text-[13.5px] font-semibold text-[var(--soga)] tnum shrink-0 text-right">
                  {sample.confidence.toFixed(1)}%
                </div>

                <button
                  onClick={() => onNavigateToAtlas?.(sample.query)}
                  className="text-[12px] font-medium text-[var(--soga)] hover:underline shrink-0 flex items-center gap-1 bg-transparent border-none cursor-pointer pl-2"
                >
                  <span>{t(language, "view_detail_link")}</span>
                  <ArrowRight className="w-3 h-3" />
                </button>
              </div>
            ))}
      </div>
    </div>
  );
};
