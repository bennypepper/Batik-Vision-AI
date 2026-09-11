"use client";

import React, { useState, useMemo, useEffect, useRef } from "react";
import { BatikClass, AppLanguage } from "@/types/batik";
import { getAllMotifs, getAllIslands } from "@/lib/labels-data";
import { t } from "@/lib/i18n";
import { Search, MapPin, ChevronDown, ChevronUp, X, Sparkles, AlertTriangle } from "lucide-react";

interface CulturalAtlasProps {
  language: AppLanguage;
  initialQuery?: string;
}

export const CulturalAtlas: React.FC<CulturalAtlasProps> = ({
  language,
  initialQuery,
}) => {
  const [searchQuery, setSearchQuery] = useState(initialQuery || "");
  const [selectedIsland, setSelectedIsland] = useState<string>("ALL");
  const [expandedIds, setExpandedIds] = useState<Record<number, boolean>>({});
  const [flashedId, setFlashedId] = useState<number | null>(null);
  const [selectedModalMotif, setSelectedModalMotif] = useState<BatikClass | null>(null);

  const allMotifs = useMemo(() => getAllMotifs(), []);
  const islands = useMemo(() => getAllIslands(), []);
  const cardRefs = useRef<Record<number, HTMLDivElement | null>>({});

  // Sync initial query when navigated from Scanner or History
  useEffect(() => {
    if (initialQuery) {
      setSearchQuery(initialQuery);
      setSelectedIsland("ALL");

      const match = allMotifs.find(
        (m) =>
          m.name_id.toLowerCase().includes(initialQuery.toLowerCase()) ||
          m.name_en.toLowerCase().includes(initialQuery.toLowerCase()) ||
          m.slug.toLowerCase().includes(initialQuery.toLowerCase())
      );

      if (match) {
        setFlashedId(match.id);
        const timer = setTimeout(() => {
          const el = cardRefs.current[match.id];
          if (el) {
            el.scrollIntoView({ behavior: "smooth", block: "center" });
          }
        }, 100);

        const clearFlashTimer = setTimeout(() => {
          setFlashedId(null);
        }, 2200);

        return () => {
          clearTimeout(timer);
          clearTimeout(clearFlashTimer);
        };
      }
    }
  }, [initialQuery, allMotifs]);

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

  const toggleExpand = (id: number) => {
    setExpandedIds((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  return (
    <div className="w-full space-y-8 pb-16">
      {/* Hero Header */}
      <div className="text-center pt-4 pb-2 px-4">
        <h1 className="display text-2xl sm:text-4xl font-semibold text-[var(--on-canvas)] mb-3">
          {t(language, "atlas_title")}
        </h1>
        <p className="text-[14px] text-[var(--on-canvas-soft)] max-w-[720px] mx-auto leading-relaxed">
          {t(language, "atlas_desc")}
        </p>
      </div>

      {/* Filter and Search Bar Container */}
      <div className="max-w-[1020px] mx-auto px-4 space-y-4">
        {/* Search Input */}
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--on-canvas-faint)]" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t(language, "search_placeholder")}
            className="w-full bg-[var(--canvas-deep)] border border-[var(--canvas-line-strong)] rounded-lg py-3.5 pl-11 pr-16 text-[14px] text-[var(--on-canvas)] placeholder:text-[var(--on-canvas-faint)] focus:outline-none focus:border-[var(--soga)] transition shadow-sm"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-3.5 top-1/2 -translate-y-1/2 text-xs text-[var(--on-canvas-faint)] hover:text-[var(--on-canvas)] transition px-2 py-1 rounded bg-[rgba(237,230,216,0.06)]"
            >
              Clear
            </button>
          )}
        </div>

        {/* Island Filter Pills */}
        <div className="flex flex-wrap gap-2 justify-center pt-1">
          <button
            onClick={() => setSelectedIsland("ALL")}
            className={`text-[12.5px] px-4 py-1.5 rounded-full border transition cursor-pointer ${
              selectedIsland === "ALL"
                ? "bg-[var(--soga)] border-[var(--soga)] text-white font-semibold shadow-sm"
                : "border-[var(--canvas-line-strong)] text-[var(--on-canvas-soft)] hover:bg-[rgba(237,230,216,0.05)]"
            }`}
          >
            {t(language, "all_islands")}
          </button>
          {islands.map((isl) => (
            <button
              key={isl}
              onClick={() => setSelectedIsland(isl)}
              className={`text-[12.5px] px-4 py-1.5 rounded-full border transition cursor-pointer ${
                selectedIsland === isl
                  ? "bg-[var(--soga)] border-[var(--soga)] text-white font-semibold shadow-sm"
                  : "border-[var(--canvas-line-strong)] text-[var(--on-canvas-soft)] hover:bg-[rgba(237,230,216,0.05)]"
              }`}
            >
              {isl}
            </button>
          ))}
        </div>

        {/* Result Count */}
        <div className="text-center text-[12px] text-[var(--on-canvas-faint)] pt-1">
          {filteredMotifs.length} {t(language, "motifs_found")}
        </div>
      </div>

      {/* Expansive Full-Screen Responsive Motif Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-5 sm:gap-6">
        {filteredMotifs.map((motif) => {
          const name = language === "id" ? motif.name_id : motif.name_en;
          const fullDesc = language === "id" ? motif.description_id : motif.description_en;
          const isExpanded = !!expandedIds[motif.id];
          const isFlashed = flashedId === motif.id;

          // Curatorial rarity: Lasem is genuine cultural rarity
          const isCulturallyRare = motif.slug === "Lasem";
          // Model weakness: Priangan Merak Ngibing and Sogan
          const isModelWeak =
            motif.slug === "Priangan_Merak_Ngibing" || motif.slug === "Sogan";

          return (
            <div
              key={motif.id}
              ref={(el) => {
                cardRefs.current[motif.id] = el;
              }}
              className={`bg-[var(--cloth)] text-[var(--on-cloth)] rounded-xl overflow-hidden border border-[var(--cloth-line)] shadow-sm hover:shadow-md transition-all flex flex-col justify-between ${
                isFlashed ? "flash" : ""
              }`}
            >
              {/* Card Top: Authentic Photographic Swatch */}
              <div
                className="relative w-full h-44 sm:h-48 overflow-hidden bg-[var(--cloth-deep)] cursor-pointer group"
                onClick={() => setSelectedModalMotif(motif)}
              >
                <img
                  src={`/motifs/${motif.slug}.jpg`}
                  alt={name}
                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                  loading="lazy"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-between p-3">
                  <span className="text-white text-[11.5px] font-medium bg-black/50 backdrop-blur-sm px-2 py-0.5 rounded">
                    {t(language, "view_full_modal")}
                  </span>
                  <Sparkles className="w-4 h-4 text-amber-200" />
                </div>

                {isCulturallyRare && (
                  <div className="absolute top-2.5 right-2.5">
                    <span className="text-[10px] font-bold px-2.5 py-1 rounded-full bg-[var(--mahogany)] text-white shadow-md">
                      {t(language, "rare_badge")}
                    </span>
                  </div>
                )}
              </div>

              {/* Card Body */}
              <div className="p-4 sm:p-5 flex-1 flex flex-col justify-between">
                <div>
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <h3 className="text-[15.5px] font-semibold text-[var(--on-cloth)] leading-snug">
                      {name}
                    </h3>
                  </div>

                  <div className="text-[12px] text-[var(--on-cloth-faint)] flex items-center gap-1 mb-3">
                    <MapPin className="w-3.5 h-3.5 text-[var(--soga)] shrink-0" />
                    <span className="truncate">
                      {motif.region} · {motif.island}
                    </span>
                  </div>

                  {/* Summary (Collapsed View) */}
                  {!isExpanded && (
                    <p className="text-[12.5px] text-[var(--on-cloth-soft)] leading-relaxed line-clamp-3 mb-3">
                      {fullDesc}
                    </p>
                  )}

                  {/* Deep Philosophy Panel (Expanded View) */}
                  {isExpanded && (
                    <div className="space-y-3 mb-3 pt-2.5 border-t border-[var(--cloth-line)]">
                      <div>
                        <span className="text-[10.5px] uppercase font-bold tracking-wider text-[var(--soga)] block mb-1">
                          {t(language, "deep_philosophy")}
                        </span>
                        <p className="text-[12.5px] text-[var(--on-cloth-soft)] leading-relaxed italic bg-[var(--cloth-deep)] p-3 rounded-lg border border-[var(--cloth-line)]">
                          &ldquo;{fullDesc}&rdquo;
                        </p>
                      </div>

                      <div>
                        <span className="text-[10.5px] uppercase font-bold tracking-wider text-[var(--on-cloth-faint)] block mb-1">
                          {t(language, "cultural_provenance")}
                        </span>
                        <p className="text-[12px] text-[var(--on-cloth-soft)]">
                          {motif.region} ({motif.island})
                        </p>
                      </div>

                      {isModelWeak && (
                        <div className="flex items-center gap-1.5 p-2 rounded bg-amber-500/10 border border-amber-500/20 text-amber-700 dark:text-amber-300 text-[11px]">
                          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                          <span>{t(language, "model_weak_note")}</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Card Footer Action Controls */}
                <div className="flex items-center justify-between pt-3 border-t border-[var(--cloth-line)] mt-auto">
                  <button
                    onClick={() => toggleExpand(motif.id)}
                    className="text-[12px] font-semibold text-[var(--soga)] hover:text-[var(--soga-deep)] flex items-center gap-1 cursor-pointer bg-transparent border-none p-0 transition"
                  >
                    <span>{isExpanded ? t(language, "hide_details") : t(language, "view_details")}</span>
                    {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                  </button>

                  <button
                    onClick={() => setSelectedModalMotif(motif)}
                    className="text-[11.5px] text-[var(--on-cloth-faint)] hover:text-[var(--on-cloth)] underline cursor-pointer bg-transparent border-none p-0 transition"
                  >
                    {t(language, "view_full_modal")}
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Modal Dialog: Full Cultural Philosophy & High-Resolution Swatch */}
      {selectedModalMotif && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4 sm:p-6 animate-fadeIn">
          <div className="bg-[var(--cloth)] text-[var(--on-cloth)] rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl border border-[var(--cloth-line)] flex flex-col">
            {/* Modal Image Header */}
            <div className="relative w-full h-64 sm:h-72 bg-black shrink-0">
              <img
                src={`/motifs/${selectedModalMotif.slug}.jpg`}
                alt={language === "id" ? selectedModalMotif.name_id : selectedModalMotif.name_en}
                className="w-full h-full object-cover"
              />
              <button
                onClick={() => setSelectedModalMotif(null)}
                className="absolute top-3.5 right-3.5 w-8 h-8 rounded-full bg-black/60 hover:bg-black/80 text-white flex items-center justify-center transition cursor-pointer border-none"
                aria-label={t(language, "close_modal")}
              >
                <X className="w-4 h-4" />
              </button>
              <div className="absolute bottom-3.5 left-4">
                <span className="text-[11px] font-bold tracking-wider uppercase px-2.5 py-1 rounded bg-black/60 backdrop-blur-md text-white">
                  {selectedModalMotif.region} · {selectedModalMotif.island}
                </span>
              </div>
            </div>

            {/* Modal Content */}
            <div className="p-6 space-y-5">
              <div className="flex items-center justify-between gap-3">
                <h2 className="display text-2xl font-bold text-[var(--on-cloth)]">
                  {language === "id" ? selectedModalMotif.name_id : selectedModalMotif.name_en}
                </h2>
                {selectedModalMotif.slug === "Lasem" && (
                  <span className="text-[11px] font-bold px-3 py-1 rounded-full bg-[var(--mahogany)] text-white">
                    {t(language, "rare_badge")}
                  </span>
                )}
              </div>

              <div>
                <h4 className="text-[12px] uppercase font-bold tracking-wider text-[var(--soga)] mb-2">
                  {t(language, "deep_philosophy")}
                </h4>
                <p className="text-[14px] text-[var(--on-cloth-soft)] leading-relaxed bg-[var(--cloth-deep)] p-4 rounded-xl border border-[var(--cloth-line)]">
                  {language === "id"
                    ? selectedModalMotif.description_id
                    : selectedModalMotif.description_en}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="p-3.5 rounded-lg bg-[var(--cloth-deep)] border border-[var(--cloth-line)]">
                  <span className="text-[11px] text-[var(--on-cloth-faint)] block mb-1">
                    {t(language, "region")}
                  </span>
                  <span className="text-[13px] font-semibold text-[var(--on-cloth)]">
                    {selectedModalMotif.region}
                  </span>
                </div>
                <div className="p-3.5 rounded-lg bg-[var(--cloth-deep)] border border-[var(--cloth-line)]">
                  <span className="text-[11px] text-[var(--on-cloth-faint)] block mb-1">
                    {t(language, "island")}
                  </span>
                  <span className="text-[13px] font-semibold text-[var(--on-cloth)]">
                    {selectedModalMotif.island}
                  </span>
                </div>
              </div>

              {(selectedModalMotif.slug === "Priangan_Merak_Ngibing" ||
                selectedModalMotif.slug === "Sogan") && (
                <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-700 dark:text-amber-300 text-xs flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{t(language, "model_weak_note")}</span>
                </div>
              )}

              <div className="pt-3 border-t border-[var(--cloth-line)] flex justify-end">
                <button
                  onClick={() => setSelectedModalMotif(null)}
                  className="px-5 py-2.5 rounded-md bg-[var(--soga)] hover:bg-[var(--soga-deep)] text-white text-[13px] font-semibold transition cursor-pointer border-none"
                >
                  {t(language, "close_modal")}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
