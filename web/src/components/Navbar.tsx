"use client";

import React from "react";
import { AppLanguage } from "@/types/batik";
import { t } from "@/lib/i18n";
import { Compass, Camera, History, Info, Globe } from "lucide-react";

interface NavbarProps {
  currentTab: "scanner" | "atlas" | "history" | "about";
  onTabChange: (tab: "scanner" | "atlas" | "history" | "about") => void;
  language: AppLanguage;
  onLanguageChange: (lang: AppLanguage) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  onTabChange,
  language,
  onLanguageChange,
}) => {
  return (
    <header className="sticky top-0 z-50 bg-stone-900/90 backdrop-blur-md border-b border-stone-800 text-stone-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div
          className="flex items-center space-x-3 cursor-pointer select-none"
          onClick={() => onTabChange("scanner")}
        >
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-600 to-amber-800 flex items-center justify-center shadow-lg shadow-amber-950/40 border border-amber-500/30">
            <svg
              className="w-6 h-6 text-amber-100"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2" />
              <line x1="12" y1="22" x2="12" y2="15.5" />
              <polyline points="22 8.5 12 15.5 2 8.5" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-amber-100 leading-tight">
              {t(language, "app_title")}
            </h1>
            <p className="text-xs text-stone-400 hidden sm:block">
              {t(language, "app_tagline")}
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1 sm:space-x-2">
          <button
            onClick={() => onTabChange("scanner")}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1.5 ${
              currentTab === "scanner"
                ? "bg-amber-500/10 text-amber-400 border border-amber-500/30"
                : "text-stone-400 hover:text-stone-200 hover:bg-stone-800/60"
            }`}
          >
            <Camera className="w-4 h-4" />
            <span>{t(language, "nav_scanner")}</span>
          </button>

          <button
            onClick={() => onTabChange("atlas")}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1.5 ${
              currentTab === "atlas"
                ? "bg-amber-500/10 text-amber-400 border border-amber-500/30"
                : "text-stone-400 hover:text-stone-200 hover:bg-stone-800/60"
            }`}
          >
            <Compass className="w-4 h-4" />
            <span>{t(language, "nav_atlas")}</span>
          </button>

          <button
            onClick={() => onTabChange("history")}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1.5 ${
              currentTab === "history"
                ? "bg-amber-500/10 text-amber-400 border border-amber-500/30"
                : "text-stone-400 hover:text-stone-200 hover:bg-stone-800/60"
            }`}
          >
            <History className="w-4 h-4" />
            <span>{t(language, "nav_history")}</span>
          </button>

          <button
            onClick={() => onTabChange("about")}
            className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1.5 ${
              currentTab === "about"
                ? "bg-amber-500/10 text-amber-400 border border-amber-500/30"
                : "text-stone-400 hover:text-stone-200 hover:bg-stone-800/60"
            }`}
          >
            <Info className="w-4 h-4" />
            <span>{t(language, "nav_about")}</span>
          </button>
        </nav>

        {/* Language Switcher */}
        <div className="flex items-center space-x-1 bg-stone-800/80 p-1 rounded-lg border border-stone-700/60">
          <Globe className="w-3.5 h-3.5 text-stone-400 ml-1 mr-1" />
          <button
            onClick={() => onLanguageChange("id")}
            className={`px-2 py-1 text-xs font-semibold rounded ${
              language === "id"
                ? "bg-amber-600 text-white shadow-sm"
                : "text-stone-400 hover:text-stone-200"
            }`}
          >
            ID
          </button>
          <button
            onClick={() => onLanguageChange("en")}
            className={`px-2 py-1 text-xs font-semibold rounded ${
              language === "en"
                ? "bg-amber-600 text-white shadow-sm"
                : "text-stone-400 hover:text-stone-200"
            }`}
          >
            EN
          </button>
        </div>
      </div>
    </header>
  );
};
