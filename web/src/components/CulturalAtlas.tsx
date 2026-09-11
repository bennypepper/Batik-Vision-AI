"use client";

import React, { useState, useMemo, useEffect, useRef } from "react";
import { BatikClass, AppLanguage } from "@/types/batik";
import { getAllMotifs, getAllIslands } from "@/lib/labels-data";
import { t } from "@/lib/i18n";
import { Search, MapPin } from "lucide-react";

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

  const allMotifs = useMemo(() => getAllMotifs(), []);
  const islands = useMemo(() => getAllIslands(), []);
  const cardRefs = useRef<Record<number, HTMLDivElement | null>>({});

  // Sync initial query when navigated from Scanner or History
  useEffect(() => {
    if (initialQuery) {
      setSearchQuery(initialQuery);
      setSelectedIsland("ALL");

      // Locate first matching motif and scroll/flash
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
        }, 1800);

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
    <div className="w-full max-w-[1180px] mx-auto space-y-6 pb-12">
      {/* Hero Header */}
      <div className="text-center pt-8 pb-2 px-4">
        <h1 className="display text-2xl sm:text-3xl font-semibold text-[var(--on-canvas)] mb-2.5">
          {t(language, "atlas_title")}
        </h1>
        <p className="text-[13.5px] text-[var(--on-canvas-soft)] max-w-[620px] mx-auto leading-relaxed">
          {t(language, "atlas_desc")}
        </p>
      </div>

      {/* Search Input */}
      <div className="max-w-[760px] mx-auto px-4">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--on-canvas-faint)]" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t(language, "search_placeholder")}
            className="w-full bg-[var(--canvas-deep)] border border-[var(--canvas-line-strong)] rounded-lg py-3.5 pl-11 pr-4 text-[13.5px] text-[var(--on-canvas)] placeholder:text-[var(--on-canvas-faint)] focus:outline-none focus:border-[var(--soga)] transition"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-3.5 top-1/2 -translate-y-1/2 text-xs text-[var(--on-canvas-faint)] hover:text-[var(--on-canvas)] transition px-1.5 py-0.5"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Island Filter Pills */}
      <div className="flex flex-wrap gap-2 justify-center max-w-[900px] mx-auto px-4">
        <button
          onClick={() => setSelectedIsland("ALL")}
          className={`text-[12px] px-3.5 py-1.5 rounded-full border transition cursor-pointer ${
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
            className={`text-[12px] px-3.5 py-1.5 rounded-full border transition cursor-pointer ${
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

      {/* Motif Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 px-4 sm:px-7">
        {filteredMotifs.map((motif, index) => {
          const name = language === "id" ? motif.name_id : motif.name_en;
          const fullDesc = language === "id" ? motif.description_id : motif.description_en;
          const swatchClass = `swatch-${(index % 5) + 1}`;
          const isExpanded = !!expandedIds[motif.id];
          const isFlashed = flashedId === motif.id;

          // Curatorial rarity: Lasem is the genuine cultural rarity
          const isCulturallyRare = motif.slug === "Lasem";
          // Model weakness: Priangan Merak Ngibing and Sogan have lower classification accuracy
          const isModelWeak =
            motif.slug === "Priangan_Merak_Ngibing" || motif.slug === "Sogan";

          return (
            <div
              key={motif.id}
              ref={(el) => {
                cardRefs.current[motif.id] = el;
              }}
              className={`bg-[var(--cloth)] text-[var(--on-cloth)] rounded-lg p-4 flex flex-col justify-between transition-all ${
                isFlashed ? "flash" : ""
              }`}
            >
              <div>
                {/* Top: Swatch and Title */}
                <div className="flex gap-3 mb-2.5">
                  <div className={`w-10 h-10 rounded-md shrink-0 ${swatchClass}`} />
                  <div className="min-w-0">
                    <h3 className="text-[14.5px] font-semibold text-[var(--on-cloth)] leading-snug truncate">
                      {name}
                    </h3>
                    <div className="text-[11.5px] text-[var(--on-cloth-faint)] flex items-center gap-1 mt-0.5">
                      <MapPin className="w-3 h-3 text-[var(--on-cloth-faint)] shrink-0" />
                      <span className="truncate">
                        {motif.region} · {motif.island}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Description */}
                <p className="text-[12.5px] text-[var(--on-cloth-soft)] leading-relaxed mb-3">
                  {isExpanded ? fullDesc : `${fullDesc.slice(0, 140)}${fullDesc.length > 140 ? "..." : ""}`}
                </p>
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between pt-2 border-t border-[var(--cloth-line)]">
                <button
                  onClick={() => toggleExpand(motif.id)}
                  className="text-[12px] font-semibold text-[var(--soga)] hover:underline cursor-pointer bg-transparent border-none p-0"
                >
                  {isExpanded ? t(language, "hide_details") : t(language, "view_details")}
                </button>

                {isCulturallyRare && (
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[var(--mahogany-tint)] text-[var(--mahogany)]">
                    {t(language, "rare_badge")}
                  </span>
                )}

                {isModelWeak && (
                  <span className="text-[10.5px] text-[var(--on-cloth-faint)] italic text-right max-w-[170px] truncate">
                    {t(language, "model_weak_note")}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
