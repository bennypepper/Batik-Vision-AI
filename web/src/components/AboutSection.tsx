"use client";

import React from "react";
import { AppLanguage } from "@/types/batik";
import { t } from "@/lib/i18n";
import { Lock } from "lucide-react";

interface AboutSectionProps {
  language: AppLanguage;
}

export const AboutSection: React.FC<AboutSectionProps> = ({ language }) => {
  return (
    <div className="w-full max-w-[760px] mx-auto pb-16 px-4">
      {/* Hero Header */}
      <div className="text-center pt-8 pb-4">
        <h1 className="display text-2xl sm:text-3xl font-semibold text-[var(--on-canvas)] mb-2.5">
          {t(language, "about_title")}
        </h1>
        <p className="text-[13.5px] text-[var(--on-canvas-soft)] max-w-[620px] mx-auto leading-relaxed">
          {t(language, "about_desc")}
        </p>
      </div>

      {/* Engineering Timeline */}
      <div className="timeline my-8">
        {/* Item 1 */}
        <div className="t-item">
          <h4 className="text-[14.5px] font-semibold text-[var(--on-canvas)] mb-1.5 flex items-center gap-2.5">
            <span>{t(language, "v3_p1_title")}</span>
            <span className="text-[10px] font-semibold text-[var(--on-canvas-faint)] border border-[var(--canvas-line-strong)] px-2 py-0.5 rounded-full">
              {t(language, "v3_p1_tag")}
            </span>
          </h4>
          <p className="text-[13px] text-[var(--on-canvas-soft)] leading-relaxed">
            {t(language, "v3_p1_desc")}
          </p>
        </div>

        {/* Item 2 */}
        <div className="t-item">
          <h4 className="text-[14.5px] font-semibold text-[var(--on-canvas)] mb-1.5">
            {t(language, "v3_p2_title")}
          </h4>
          <p className="text-[13px] text-[var(--on-canvas-soft)] leading-relaxed">
            {t(language, "v3_p2_desc")}
          </p>
        </div>

        {/* Item 3 */}
        <div className="t-item">
          <h4 className="text-[14.5px] font-semibold text-[var(--on-canvas)] mb-1.5">
            {t(language, "v3_p3_title")}
          </h4>
          <p className="text-[13px] text-[var(--on-canvas-soft)] leading-relaxed">
            {t(language, "v3_p3_desc")}
          </p>
        </div>

        {/* Item 4 */}
        <div className="t-item">
          <h4 className="text-[14.5px] font-semibold text-[var(--on-canvas)] mb-1.5">
            {t(language, "v3_p4_title")}
          </h4>
          <p className="text-[13px] text-[var(--on-canvas-soft)] leading-relaxed">
            {t(language, "v3_p4_desc")}
          </p>
        </div>
      </div>

      {/* Highlight Block: Zero-Backend */}
      <div className="bg-[var(--cloth)] text-[var(--on-cloth)] rounded-lg p-5 sm:p-6 shadow-sm border border-[var(--cloth-line)]">
        <h4 className="text-[15px] font-semibold text-[var(--on-cloth)] mb-2 flex items-center gap-2.5">
          <Lock className="w-4 h-4 text-[var(--soga)] shrink-0" />
          <span>{t(language, "zero_backend_title")}</span>
        </h4>
        <p className="text-[13px] text-[var(--on-cloth-soft)] leading-relaxed">
          {t(language, "zero_backend_desc")}
        </p>
      </div>
    </div>
  );
};
