"use client";

import React from "react";
import { AppLanguage } from "@/types/batik";
import { t } from "@/lib/i18n";
import { Lock, ExternalLink, Code, Cpu, BookOpen } from "lucide-react";

const GithubIcon: React.FC<{ className?: string }> = ({ className = "w-4 h-4" }) => (
  <svg className={className} fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
    <path
      fillRule="evenodd"
      d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
      clipRule="evenodd"
    />
  </svg>
);

interface AboutSectionProps {
  language: AppLanguage;
}

export const AboutSection: React.FC<AboutSectionProps> = ({ language }) => {
  return (
    <div className="w-full max-w-5xl mx-auto pb-20 px-4 space-y-12">
      {/* Hero Header */}
      <div className="text-center pt-4 pb-2">
        <h1 className="display text-2xl sm:text-4xl font-semibold text-[var(--on-canvas)] mb-3">
          {t(language, "about_title")}
        </h1>
        <p className="text-[14px] text-[var(--on-canvas-soft)] max-w-[700px] mx-auto leading-relaxed">
          {t(language, "about_desc")}
        </p>
      </div>

      {/* Main Two-Column Layout on Desktop */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Platform Mission & Developer Profile (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* About Platform Card */}
          <div className="bg-[var(--cloth)] text-[var(--on-cloth)] rounded-xl p-6 border border-[var(--cloth-line)] shadow-sm space-y-3.5">
            <h3 className="text-[16px] font-semibold text-[var(--on-cloth)] flex items-center gap-2.5">
              <BookOpen className="w-4 h-4 text-[var(--soga)] shrink-0" />
              <span>{t(language, "web_overview_title")}</span>
            </h3>
            <p className="text-[13px] text-[var(--on-cloth-soft)] leading-relaxed">
              {t(language, "web_overview_desc")}
            </p>
          </div>

          {/* Zero-Backend Privacy Card */}
          <div className="bg-[var(--cloth)] text-[var(--on-cloth)] rounded-xl p-6 border border-[var(--cloth-line)] shadow-sm space-y-3">
            <h3 className="text-[15px] font-semibold text-[var(--on-cloth)] flex items-center gap-2.5">
              <Lock className="w-4 h-4 text-[var(--soga)] shrink-0" />
              <span>{t(language, "zero_backend_title")}</span>
            </h3>
            <p className="text-[12.5px] text-[var(--on-cloth-soft)] leading-relaxed">
              {t(language, "zero_backend_desc")}
            </p>
          </div>

          {/* Developer Profile Card */}
          <div className="bg-[var(--cloth)] text-[var(--on-cloth)] rounded-xl p-6 border border-[var(--cloth-line)] shadow-sm space-y-4">
            <div className="flex items-center gap-3.5 pb-3 border-b border-[var(--cloth-line)]">
              <div className="w-12 h-12 rounded-full bg-[var(--soga-tint)] text-[var(--soga)] flex items-center justify-center font-bold text-lg shrink-0 border border-[rgba(161,90,46,0.3)]">
                BP
              </div>
              <div className="min-w-0">
                <h3 className="text-[16px] font-semibold text-[var(--on-cloth)] leading-tight">
                  {t(language, "dev_name")}
                </h3>
                <p className="text-[12px] text-[var(--soga)] font-medium mt-0.5">
                  {t(language, "dev_role")}
                </p>
              </div>
            </div>

            <p className="text-[12.5px] text-[var(--on-cloth-soft)] leading-relaxed">
              {t(language, "dev_bio")}
            </p>

            {/* Social & Repository Links */}
            <div className="flex flex-wrap gap-2.5 pt-1">
              <a
                href="https://github.com/bennypepper"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-[var(--soga)] hover:bg-[var(--soga-deep)] text-white text-[12px] font-semibold transition shadow-sm no-underline"
              >
                <GithubIcon className="w-3.5 h-3.5" />
                <span>{t(language, "github_profile")}</span>
                <ExternalLink className="w-3 h-3 opacity-75" />
              </a>

              <a
                href="https://github.com/bennypepper/CNN-Batik-Motifs-Detector"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-[var(--cloth-deep)] hover:bg-[var(--canvas-deep)] text-[var(--on-cloth)] text-[12px] font-medium border border-[var(--cloth-line-strong)] transition no-underline"
              >
                <Code className="w-3.5 h-3.5 text-[var(--soga)]" />
                <span>{t(language, "github_repo")}</span>
                <ExternalLink className="w-3 h-3 opacity-60" />
              </a>
            </div>
          </div>
        </div>

        {/* Right Column: Engineering Timeline (7 cols) */}
        <div className="lg:col-span-7 bg-[var(--canvas-deep)] p-6 sm:p-8 rounded-2xl border border-[var(--canvas-line-strong)]">
          <div className="flex items-center gap-2 mb-6 pb-3 border-b border-[var(--canvas-line-strong)]">
            <Cpu className="w-4 h-4 text-[var(--soga)]" />
            <h3 className="text-[16px] font-semibold text-[var(--on-canvas)]">
              {language === "id" ? "Linimasa Rekayasa Sistem" : "System Engineering Milestones"}
            </h3>
          </div>

          <div className="timeline">
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
        </div>
      </div>
    </div>
  );
};
