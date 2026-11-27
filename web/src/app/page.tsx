"use client";

import React, { useState, useEffect } from "react";
import { AppLanguage, ScanRecord, PredictionResult } from "@/types/batik";
import { Navbar } from "@/components/Navbar";
import { FabricScanner } from "@/components/FabricScanner";
import { CulturalAtlas } from "@/components/CulturalAtlas";
import { ScanHistory } from "@/components/ScanHistory";
import { AboutSection } from "@/components/AboutSection";
import { validateHistoryRecords } from "@/lib/classifier";
import { t } from "@/lib/i18n";

const LOCAL_STORAGE_KEY_HISTORY = "batik_v3_scan_history";
const LOCAL_STORAGE_KEY_LANG = "batik_v3_language";
const LOCAL_STORAGE_KEY_THEME = "batik_v3_theme";

export default function HomePage() {
  const [currentTab, setCurrentTab] = useState<
    "scanner" | "atlas" | "history" | "about"
  >("scanner");
  const [language, setLanguage] = useState<AppLanguage>("en");
  const [theme, setTheme] = useState<"dark" | "light">("light");
  const [historyRecords, setHistoryRecords] = useState<ScanRecord[]>([]);
  const [targetAtlasQuery, setTargetAtlasQuery] = useState<string | undefined>(
    undefined
  );

  // Hydrate persisted language, theme, and validated history after mount
  useEffect(() => {
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      try {
        const savedLang = localStorage.getItem(LOCAL_STORAGE_KEY_LANG);
        if (savedLang === "id" || savedLang === "en") {
          setLanguage(savedLang);
        }

        const savedTheme = localStorage.getItem(LOCAL_STORAGE_KEY_THEME);
        if (savedTheme === "light" || savedTheme === "dark") {
          setTheme(savedTheme);
        }

        const savedHistory = localStorage.getItem(LOCAL_STORAGE_KEY_HISTORY);
        if (savedHistory) {
          const parsed: unknown = JSON.parse(savedHistory);
          setHistoryRecords(validateHistoryRecords(parsed));
        }
      } catch {
        // LocalStorage access or JSON parse fallback
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Sync document root attributes when language or theme changes
  useEffect(() => {
    document.documentElement.lang = language;
    document.documentElement.setAttribute("data-theme", theme);
  }, [language, theme]);

  const handleLanguageChange = (lang: AppLanguage) => {
    setLanguage(lang);
    document.documentElement.lang = lang;
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY_LANG, lang);
    } catch {
      // LocalStorage fallback
    }
  };

  const handleThemeChange = (newTheme: "dark" | "light") => {
    setTheme(newTheme);
    document.documentElement.setAttribute("data-theme", newTheme);
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY_THEME, newTheme);
    } catch {
      // LocalStorage fallback
    }
  };

  const handleSaveScanRecord = (recordData: {
    predictions: PredictionResult[];
    inferenceTimeMs: number;
  }) => {
    if (!recordData.predictions.length) return;
    const newRecord: ScanRecord = {
      id:
        "scan_" +
        Date.now() +
        "_" +
        Math.random().toString(36).substring(2, 7),
      timestamp: Date.now(),
      topPrediction: recordData.predictions[0],
      topCandidates: recordData.predictions,
      inferenceTimeMs: recordData.inferenceTimeMs,
    };

    setHistoryRecords((prev) => {
      const updated = [newRecord, ...prev].slice(0, 30);
      try {
        localStorage.setItem(
          LOCAL_STORAGE_KEY_HISTORY,
          JSON.stringify(updated)
        );
      } catch {
        // Gracefully handle QuotaExceededError by trimming older entries
        try {
          localStorage.setItem(
            LOCAL_STORAGE_KEY_HISTORY,
            JSON.stringify(updated.slice(0, 10))
          );
        } catch {
          // Storage disabled or completely full
        }
      }
      return updated;
    });
  };

  const handleClearHistory = () => {
    setHistoryRecords([]);
    try {
      localStorage.removeItem(LOCAL_STORAGE_KEY_HISTORY);
    } catch {
      // Fallback
    }
  };

  const handleNavigateToAtlas = (motifName: string) => {
    setTargetAtlasQuery(motifName);
    setCurrentTab("atlas");
  };

  return (
    <div className="min-h-screen bg-[var(--canvas)] text-[var(--on-canvas)] flex flex-col justify-between transition-colors">
      {/* Top Batik Border Strip */}
      <div className="motif-strip" />

      {/* Full-width Navigation Header */}
      <Navbar
        currentTab={currentTab}
        onTabChange={(tab) => {
          setCurrentTab(tab);
          if (tab !== "atlas") {
            setTargetAtlasQuery(undefined);
          }
        }}
        language={language}
        onLanguageChange={handleLanguageChange}
        theme={theme}
        onThemeChange={handleThemeChange}
      />

      {/* Main Expansive Content Viewport */}
      <main className="flex-1 w-full max-w-[1680px] mx-auto px-4 sm:px-8 lg:px-12 py-6 sm:py-8">
        {currentTab === "scanner" && (
          <FabricScanner
            language={language}
            onSaveScanRecord={handleSaveScanRecord}
            onNavigateToAtlas={handleNavigateToAtlas}
          />
        )}

        {currentTab === "atlas" && (
          <CulturalAtlas
            key={targetAtlasQuery ?? "atlas-default"}
            language={language}
            initialQuery={targetAtlasQuery}
          />
        )}

        {currentTab === "history" && (
          <ScanHistory
            records={historyRecords}
            language={language}
            onClearHistory={handleClearHistory}
            onNavigateToAtlas={handleNavigateToAtlas}
            onStartScan={() => setCurrentTab("scanner")}
          />
        )}

        {currentTab === "about" && <AboutSection language={language} />}
      </main>

      {/* Clean Full-Width Footer */}
      <footer className="w-full flex flex-col sm:flex-row items-center justify-between px-6 sm:px-12 py-4 border-t border-[var(--canvas-line)] text-[12px] text-[var(--on-canvas-faint)] gap-2 bg-[var(--canvas-deep)] transition-colors">
        <span>{t(language, "footer_text")}</span>
        <span className="tnum">{t(language, "footer_meta")}</span>
      </footer>
    </div>
  );
}
