"use client";

import React from "react";
import { AppLanguage } from "@/types/batik";
import { t } from "@/lib/i18n";
import { Camera, Compass, Clock, Info, Sun, Moon } from "lucide-react";

interface NavbarProps {
  currentTab: "scanner" | "atlas" | "history" | "about";
  onTabChange: (tab: "scanner" | "atlas" | "history" | "about") => void;
  language: AppLanguage;
  onLanguageChange: (lang: AppLanguage) => void;
  theme: "dark" | "light";
  onThemeChange: (theme: "dark" | "light") => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  onTabChange,
  language,
  onLanguageChange,
  theme,
  onThemeChange,
}) => {
  return (
    <header className="w-full bg-[var(--canvas-deep)] border-b border-[var(--canvas-line)] transition-colors">
      <div className="w-full px-4 sm:px-8 lg:px-12 py-3.5 flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Brand */}
        <button
          type="button"
          className="flex items-baseline gap-2.5 cursor-pointer select-none bg-transparent border-none p-0 text-left"
          onClick={() => onTabChange("scanner")}
        >
          <span className="display text-lg font-semibold italic text-[var(--on-canvas)]">
            {t(language, "app_title")}
          </span>
          <span className="text-[11px] text-[var(--on-canvas-faint)] font-medium">
            {t(language, "app_version")}
          </span>
          <span className="hidden lg:inline text-[11.5px] text-[var(--on-canvas-faint)] pl-3.5 border-l border-[var(--canvas-line-strong)]">
            {t(language, "app_subtitle")}
          </span>
        </button>

        {/* Navigation Tabs and Language Switch */}
        <div className="flex items-center gap-1.5 sm:gap-2 flex-wrap justify-center">
          <button
            onClick={() => onTabChange("scanner")}
            className={`flex items-center gap-1.5 text-[13px] font-medium px-3.5 py-2 rounded-md transition-colors border ${
              currentTab === "scanner"
                ? "text-[var(--on-canvas)] bg-[var(--soga-tint)] border-[rgba(161,90,46,0.35)]"
                : "text-[var(--on-canvas-soft)] border-transparent hover:bg-[rgba(237,230,216,0.05)]"
            }`}
          >
            <Camera className="w-3.5 h-3.5 shrink-0" />
            <span>{t(language, "nav_scanner")}</span>
          </button>

          <button
            onClick={() => onTabChange("atlas")}
            className={`flex items-center gap-1.5 text-[13px] font-medium px-3.5 py-2 rounded-md transition-colors border ${
              currentTab === "atlas"
                ? "text-[var(--on-canvas)] bg-[var(--soga-tint)] border-[rgba(161,90,46,0.35)]"
                : "text-[var(--on-canvas-soft)] border-transparent hover:bg-[rgba(237,230,216,0.05)]"
            }`}
          >
            <Compass className="w-3.5 h-3.5 shrink-0" />
            <span>{t(language, "nav_atlas")}</span>
          </button>

          <button
            onClick={() => onTabChange("history")}
            className={`flex items-center gap-1.5 text-[13px] font-medium px-3.5 py-2 rounded-md transition-colors border ${
              currentTab === "history"
                ? "text-[var(--on-canvas)] bg-[var(--soga-tint)] border-[rgba(161,90,46,0.35)]"
                : "text-[var(--on-canvas-soft)] border-transparent hover:bg-[rgba(237,230,216,0.05)]"
            }`}
          >
            <Clock className="w-3.5 h-3.5 shrink-0" />
            <span>{t(language, "nav_history")}</span>
          </button>

          <button
            onClick={() => onTabChange("about")}
            className={`flex items-center gap-1.5 text-[13px] font-medium px-3.5 py-2 rounded-md transition-colors border ${
              currentTab === "about"
                ? "text-[var(--on-canvas)] bg-[var(--soga-tint)] border-[rgba(161,90,46,0.35)]"
                : "text-[var(--on-canvas-soft)] border-transparent hover:bg-[rgba(237,230,216,0.05)]"
            }`}
          >
            <Info className="w-3.5 h-3.5 shrink-0" />
            <span>{t(language, "nav_about")}</span>
          </button>

          {/* Language Switch */}
          <div className="flex ml-1.5 border border-[var(--canvas-line-strong)] rounded-md overflow-hidden">
            <button
              onClick={() => onLanguageChange("id")}
              className={`text-[11.5px] font-semibold px-2.5 py-1.5 transition-colors cursor-pointer ${
                language === "id"
                  ? "bg-[rgba(237,230,216,0.08)] text-[var(--on-canvas)] font-bold"
                  : "text-[var(--on-canvas-faint)] hover:text-[var(--on-canvas-soft)]"
              }`}
            >
              ID
            </button>
            <button
              onClick={() => onLanguageChange("en")}
              className={`text-[11.5px] font-semibold px-2.5 py-1.5 transition-colors cursor-pointer ${
                language === "en"
                  ? "bg-[rgba(237,230,216,0.08)] text-[var(--on-canvas)] font-bold"
                  : "text-[var(--on-canvas-faint)] hover:text-[var(--on-canvas-soft)]"
              }`}
            >
              EN
            </button>
          </div>

          {/* Theme Switch */}
          <button
            onClick={() => onThemeChange(theme === "dark" ? "light" : "dark")}
            className="ml-1 p-2 rounded-md border border-[var(--canvas-line-strong)] text-[var(--on-canvas-soft)] hover:text-[var(--on-canvas)] hover:bg-[rgba(237,230,216,0.06)] transition-colors cursor-pointer"
            title={theme === "dark" ? t(language, "theme_light") : t(language, "theme_dark")}
            aria-label={t(language, "toggle_theme")}
          >
            {theme === "dark" ? (
              <Sun className="w-3.5 h-3.5 text-[var(--soga)]" />
            ) : (
              <Moon className="w-3.5 h-3.5 text-[var(--indigo)]" />
            )}
          </button>
        </div>
      </div>
    </header>
  );
};
