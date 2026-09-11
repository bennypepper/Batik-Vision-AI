"use client";

import React from "react";
import { ScanRecord, AppLanguage } from "@/types/batik";
import { t } from "@/lib/i18n";
import { getMotifRegion, getMotifIsland } from "@/lib/labels-data";
import { ArrowRight, Trash2, Clock } from "lucide-react";

interface ScanHistoryProps {
  records: ScanRecord[];
  language: AppLanguage;
  onClearHistory: () => void;
  onNavigateToAtlas?: (motifName: string) => void;
  onStartScan?: () => void;
}

export const ScanHistory: React.FC<ScanHistoryProps> = ({
  records,
  language,
  onClearHistory,
  onNavigateToAtlas,
  onStartScan,
}) => {
  const hasUserRecords = records.length > 0;

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

      {hasUserRecords ? (
        <>
          {/* Header Bar */}
          <div className="flex items-center justify-between mt-6 mb-3 px-1">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-[var(--soga)]" />
              <h3 className="text-[14.5px] font-semibold text-[var(--on-canvas)]">
                {records.length} {t(language, "saved_scans")}
              </h3>
            </div>
            <button
              onClick={onClearHistory}
              className="text-[12px] text-[var(--on-canvas-soft)] hover:text-red-400 bg-transparent border border-[var(--canvas-line-strong)] hover:border-red-400/40 px-3 py-1.5 rounded-md transition flex items-center gap-1.5 cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>{t(language, "clear_history")}</span>
            </button>
          </div>

          {/* History List */}
          <div className="bg-[var(--cloth)] text-[var(--on-cloth)] rounded-xl overflow-hidden border border-[var(--cloth-line)] shadow-sm">
            {records.map((rec, index) => {
              const top = rec.topPrediction;
              const name = language === "id" ? top.classInfo.name_id : top.classInfo.name_en;
              const dateStr = new Date(rec.timestamp).toLocaleString(
                language === "id" ? "id-ID" : "en-US",
                { dateStyle: "short", timeStyle: "short" }
              );

              return (
                <div
                  key={rec.id || index}
                  className="flex items-center gap-4 px-5 py-4 border-b border-[var(--cloth-line)] last:border-b-0 hover:bg-[var(--cloth-deep)] transition"
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
                      {dateStr} · {getMotifRegion(top.classInfo, language)} ({getMotifIsland(top.classInfo, language)})
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
            })}
          </div>
        </>
      ) : (
        /* Empty State */
        <div className="bg-[var(--cloth)] text-[var(--on-cloth)] rounded-xl p-10 border border-[var(--cloth-line)] text-center max-w-md mx-auto shadow-sm my-8">
          <div className="w-12 h-12 rounded-full bg-[var(--cloth-deep)] border border-[var(--cloth-line)] flex items-center justify-center mx-auto mb-3.5 text-[var(--soga)]">
            <Clock className="w-5 h-5" />
          </div>
          <h3 className="text-[15.5px] font-semibold text-[var(--on-cloth)] mb-1.5">
            {t(language, "no_scans")}
          </h3>
          <p className="text-[13px] text-[var(--on-cloth-soft)] leading-relaxed mb-5 max-w-xs mx-auto">
            {t(language, "no_scans_prompt")}
          </p>
          {onStartScan && (
            <button
              onClick={onStartScan}
              className="inline-flex items-center gap-2 bg-[var(--soga)] hover:bg-[var(--soga-deep)] text-white text-xs font-semibold px-4 py-2.5 rounded-md transition cursor-pointer border-none shadow-sm"
            >
              <span>{t(language, "btn_start_scan")}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      )}
    </div>
  );
};
