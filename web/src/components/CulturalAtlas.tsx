"use client";

import React, { useState, useMemo } from "react";
import { BatikClass, AppLanguage } from "@/types/batik";
import { getAllMotifs, getAllIslands } from "@/lib/labels-data";
import { t } from "@/lib/i18n";
import { Search, MapPin, Sparkles, BookOpen, X } from "lucide-react";

interface CulturalAtlasProps {
  language: AppLanguage;
}

export const CulturalAtlas: React.FC<CulturalAtlasProps> = ({ language }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedIsland, setSelectedIsland] = useState<string>("ALL");
  const [activeModalMotif, setActiveModalMotif] = useState<BatikClass | null>(null);

  const allMotifs = useMemo(() => getAllMotifs(), []);
  const islands = useMemo(() => getAllIslands(), []);

  const filteredMotifs = useMemo(() => {
    return allMotifs.filter((m) => {
      // Island filter
      if (selectedIsland !== "ALL" && m.island.toLowerCase() !== selectedIsland.toLowerCase()) {
        return false;
      }
      // Query filter
      if (!searchQuery.trim()) return true;
      const q = searchQuery.toLowerCase();
      const name = language === "id" ? m.name_id : m.name_en;
      const desc = language === "id" ? m.description_id : m.description_en;
      return (
        name.toLowerCase().includes(q) ||
        m.region.toLowerCase().includes(q) ||
        m.island.toLowerCase().includes(q) ||
        desc.toLowerCase().includes(q) ||
        m.slug.toLowerCase().includes(q)
      );
    });
  }, [allMotifs, selectedIsland, searchQuery, language]);

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center space-y-2 py-2">
        <h2 className="text-2xl sm:text-3xl font-extrabold text-stone-100 tracking-tight">
          {t(language, "atlas_title")}
        </h2>
        <p className="text-sm text-stone-400 max-w-2xl mx-auto">
          {t(language, "atlas_desc")}
        </p>
      </div>

      {/* Search and Island Filter Bar */}
      <div className="space-y-3 bg-stone-900/80 p-4 rounded-2xl border border-stone-800 shadow-lg">
        {/* Search input */}
        <div className="relative">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-stone-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t(language, "search_placeholder")}
            className="w-full pl-10 pr-4 py-2.5 bg-stone-950/80 border border-stone-800 rounded-xl text-sm text-stone-200 placeholder-stone-500 focus:outline-none focus:border-amber-500/80 transition"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-stone-500 hover:text-stone-300"
            >
              Clear
            </button>
          )}
        </div>

        {/* Island filter pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs no-scrollbar">
          <button
            onClick={() => setSelectedIsland("ALL")}
            className={`px-3 py-1.5 rounded-lg font-medium whitespace-nowrap transition ${
              selectedIsland === "ALL"
                ? "bg-amber-600 text-stone-950 font-bold shadow"
                : "bg-stone-800 text-stone-400 hover:text-stone-200"
            }`}
          >
            {t(language, "all_islands")}
          </button>
          {islands.map((isl) => (
            <button
              key={isl}
              onClick={() => setSelectedIsland(isl)}
              className={`px-3 py-1.5 rounded-lg font-medium whitespace-nowrap transition ${
                selectedIsland === isl
                  ? "bg-amber-600 text-stone-950 font-bold shadow"
                  : "bg-stone-800 text-stone-400 hover:text-stone-200"
              }`}
            >
              {isl}
            </button>
          ))}
        </div>
      </div>

      {/* Result Count */}
      <div className="text-xs text-stone-400 flex items-center justify-between px-1">
        <span>
          {filteredMotifs.length} {t(language, "motifs_found")}
        </span>
        {selectedIsland !== "ALL" && (
          <span className="text-amber-400 font-mono">
            {selectedIsland}
          </span>
        )}
      </div>

      {/* Grid of Motif Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredMotifs.map((motif) => {
          const name = language === "id" ? motif.name_id : motif.name_en;
          const desc = language === "id" ? motif.description_id : motif.description_en;

          return (
            <div
              key={motif.id}
              onClick={() => setActiveModalMotif(motif)}
              className="bg-stone-900/80 hover:bg-stone-900 rounded-xl border border-stone-800 hover:border-amber-500/50 p-5 space-y-3 cursor-pointer transition-all duration-200 shadow-md hover:shadow-xl hover:shadow-amber-950/20 flex flex-col justify-between group"
            >
              <div className="space-y-2">
                <div className="flex items-start justify-between gap-2">
                  <h3 className="font-bold text-stone-200 group-hover:text-amber-300 transition text-base leading-snug">
                    {name}
                  </h3>
                  {motif.is_weak_class && (
                    <span className="shrink-0 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-950/80 text-amber-300 border border-amber-700/50">
                      {t(language, "weak_class_badge")}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 text-xs text-stone-400">
                  <span className="inline-flex items-center gap-1">
                    <MapPin className="w-3 h-3 text-amber-400" />
                    {motif.region}
                  </span>
                  <span className="text-stone-600">•</span>
                  <span className="text-stone-400">{motif.island}</span>
                </div>

                <p className="text-xs text-stone-400 line-clamp-3 leading-relaxed">
                  {desc}
                </p>
              </div>

              <div className="pt-2 border-t border-stone-800/80 flex items-center justify-between text-xs text-amber-400/80 group-hover:text-amber-400 font-medium">
                <span className="flex items-center gap-1">
                  <BookOpen className="w-3.5 h-3.5" />
                  <span>{t(language, "view_details")}</span>
                </span>
                <span className="text-stone-600 font-mono text-[11px]">#{motif.id}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Motif Detail Modal */}
      {activeModalMotif && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fade-in">
          <div className="bg-stone-900 border border-stone-800 rounded-2xl max-w-2xl w-full p-6 md:p-8 space-y-6 shadow-2xl relative max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setActiveModalMotif(null)}
              className="absolute top-4 right-4 p-2 text-stone-400 hover:text-stone-200 rounded-lg hover:bg-stone-800 transition"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-amber-950/80 text-amber-300 border border-amber-800/60">
                  <MapPin className="w-3 h-3 text-amber-400" />
                  {activeModalMotif.region}
                </span>
                <span className="px-2.5 py-1 rounded-lg text-xs font-medium bg-stone-800 text-stone-300 border border-stone-700">
                  {activeModalMotif.island}
                </span>
                {activeModalMotif.is_weak_class && (
                  <span className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-amber-950/80 text-amber-300 border border-amber-700/60 flex items-center gap-1">
                    <Sparkles className="w-3 h-3" />
                    {t(language, "weak_class_badge")}
                  </span>
                )}
              </div>

              <h2 className="text-2xl md:text-3xl font-extrabold text-amber-200">
                {language === "id" ? activeModalMotif.name_id : activeModalMotif.name_en}
              </h2>
              <p className="text-xs font-mono text-stone-400">
                slug: {activeModalMotif.slug} | id: {activeModalMotif.id}
              </p>
            </div>

            <div className="space-y-3 bg-stone-950/60 p-5 rounded-xl border border-stone-800/80">
              <h4 className="text-xs font-bold uppercase tracking-wider text-amber-400 flex items-center gap-1.5">
                <BookOpen className="w-4 h-4 text-amber-400" />
                {t(language, "philosophy")}
              </h4>
              <p className="text-sm md:text-base text-stone-300 leading-relaxed whitespace-pre-line">
                {language === "id" ? activeModalMotif.description_id : activeModalMotif.description_en}
              </p>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setActiveModalMotif(null)}
                className="px-5 py-2 rounded-xl bg-stone-800 hover:bg-stone-700 text-stone-200 text-sm font-semibold transition"
              >
                {t(language, "close_modal")}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
