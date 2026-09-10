"use client";

import React from "react";
import { AppLanguage } from "@/types/batik";
import { t } from "@/lib/i18n";
import { ShieldCheck, Cpu, Layers, RefreshCw, Eye } from "lucide-react";

interface AboutSectionProps {
  language: AppLanguage;
}

export const AboutSection: React.FC<AboutSectionProps> = ({ language }) => {
  return (
    <div className="w-full max-w-4xl mx-auto space-y-8 py-4">
      {/* Header */}
      <div className="text-center space-y-2">
        <h2 className="text-2xl sm:text-3xl font-extrabold text-stone-100 tracking-tight">
          {t(language, "about_title")}
        </h2>
        <p className="text-sm text-stone-400 max-w-2xl mx-auto">
          {t(language, "about_desc")}
        </p>
      </div>

      {/* Grid of Key Architectural Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Pillar 1: Double-Preprocessing Fix */}
        <div className="bg-stone-900/80 rounded-2xl border border-stone-800 p-6 space-y-3 shadow-lg">
          <div className="w-10 h-10 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center border border-amber-500/20">
            <RefreshCw className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-stone-200">
            {language === "id" ? "Penyelesaian Double-Preprocessing" : "Double-Preprocessing Fix"}
          </h3>
          <p className="text-xs text-stone-400 leading-relaxed">
            {t(language, "v3_p1")}
          </p>
        </div>

        {/* Pillar 2: Perceptual Hash Deduplication */}
        <div className="bg-stone-900/80 rounded-2xl border border-stone-800 p-6 space-y-3 shadow-lg">
          <div className="w-10 h-10 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center border border-amber-500/20">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-stone-200">
            {language === "id" ? "Deduplikasi Perceptual Hash" : "Perceptual Hash Deduplication"}
          </h3>
          <p className="text-xs text-stone-400 leading-relaxed">
            {t(language, "v3_p2")}
          </p>
        </div>

        {/* Pillar 3: Focal Loss & Imbalance */}
        <div className="bg-stone-900/80 rounded-2xl border border-stone-800 p-6 space-y-3 shadow-lg">
          <div className="w-10 h-10 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center border border-amber-500/20">
            <Layers className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-stone-200">
            {language === "id" ? "Focal Loss & Sample Weighting" : "Focal Loss & Effective Sample Weighting"}
          </h3>
          <p className="text-xs text-stone-400 leading-relaxed">
            {t(language, "v3_p3")}
          </p>
        </div>

        {/* Pillar 4: Two-Stage YOLO Architecture */}
        <div className="bg-stone-900/80 rounded-2xl border border-stone-800 p-6 space-y-3 shadow-lg">
          <div className="w-10 h-10 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center border border-amber-500/20">
            <Eye className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-stone-200">
            {language === "id" ? "Arsitektur Deteksi Dua Tahap" : "Two-Stage Detection Architecture"}
          </h3>
          <p className="text-xs text-stone-400 leading-relaxed">
            {t(language, "v3_p4")}
          </p>
        </div>
      </div>

      {/* Zero-Backend Section */}
      <div className="bg-gradient-to-br from-stone-900 to-stone-950 rounded-2xl border border-stone-800 p-6 md:p-8 space-y-3 shadow-xl">
        <div className="flex items-center gap-2.5 text-amber-400">
          <Cpu className="w-5 h-5" />
          <h3 className="text-base font-bold text-stone-100">
            {t(language, "zero_backend_title")}
          </h3>
        </div>
        <p className="text-sm text-stone-300 leading-relaxed">
          {t(language, "zero_backend_desc")}
        </p>
      </div>
    </div>
  );
};
