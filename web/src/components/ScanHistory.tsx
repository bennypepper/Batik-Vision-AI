"use client";

import React from "react";
import { ScanRecord, AppLanguage } from "@/types/batik";
import { t } from "@/lib/i18n";
import { Trash2, Calendar, MapPin } from "lucide-react";

interface ScanHistoryProps {
  records: ScanRecord[];
  language: AppLanguage;
  onClearHistory: () => void;
  onSelectRecord?: (record: ScanRecord) => void;
}

export const ScanHistory: React.FC<ScanHistoryProps> = ({
  records,
  language,
  onClearHistory,
  onSelectRecord,
}) => {
  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between py-2">
        <div>
          <h2 className="text-2xl font-extrabold text-stone-100 tracking-tight">
            {t(language, "history_title")}
          </h2>
          <p className="text-xs text-stone-400">{t(language, "history_desc")}</p>
        </div>
        {records.length > 0 && (
          <button
            onClick={onClearHistory}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-red-400 hover:text-red-300 hover:bg-red-950/40 border border-red-900/50 flex items-center gap-1.5 transition"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>{t(language, "clear_history")}</span>
          </button>
        )}
      </div>

      {/* History Items */}
      {records.length === 0 ? (
        <div className="text-center py-16 bg-stone-900/50 rounded-2xl border border-stone-800 space-y-2">
          <p className="text-sm text-stone-400">{t(language, "no_scans")}</p>
        </div>
      ) : (
        <div className="space-y-3">
          {records.map((rec) => {
            const top = rec.topPrediction;
            const name = language === "id" ? top.classInfo.name_id : top.classInfo.name_en;
            const dateStr = new Date(rec.timestamp).toLocaleString(
              language === "id" ? "id-ID" : "en-US",
              { dateStyle: "medium", timeStyle: "short" }
            );

            return (
              <div
                key={rec.id}
                onClick={() => onSelectRecord?.(rec)}
                className="bg-stone-900/80 hover:bg-stone-900 rounded-xl border border-stone-800 hover:border-amber-500/40 p-4 transition flex items-center gap-4 cursor-pointer group shadow-sm"
              >
                {/* Thumbnail */}
                <div className="w-16 h-16 rounded-lg overflow-hidden bg-stone-950 shrink-0 border border-stone-800">
                  <img
                    src={rec.imageThumbnail}
                    alt={name}
                    className="w-full h-full object-cover"
                  />
                </div>

                {/* Details */}
                <div className="flex-1 min-w-0 space-y-1">
                  <div className="flex items-center justify-between gap-2">
                    <h4 className="font-bold text-stone-200 group-hover:text-amber-300 transition text-sm truncate">
                      {name}
                    </h4>
                    <span className="text-xs font-mono font-bold text-amber-400 shrink-0">
                      {top.score}%
                    </span>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 text-xs text-stone-400">
                    <span className="inline-flex items-center gap-1">
                      <MapPin className="w-3 h-3 text-amber-400" />
                      {top.classInfo.region}
                    </span>
                    <span className="text-stone-600">•</span>
                    <span className="inline-flex items-center gap-1 text-stone-500">
                      <Calendar className="w-3 h-3" />
                      {dateStr}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
