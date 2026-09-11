"use client";

import React, { useState, useEffect } from "react";
import { AppLanguage, ScanRecord, PredictionResult } from "@/types/batik";
import { Navbar } from "@/components/Navbar";
import { FabricScanner } from "@/components/FabricScanner";
import { CulturalAtlas } from "@/components/CulturalAtlas";
import { ScanHistory } from "@/components/ScanHistory";
import { AboutSection } from "@/components/AboutSection";
import { t } from "@/lib/i18n";

const LOCAL_STORAGE_KEY_HISTORY = "batik_v3_scan_history";
const LOCAL_STORAGE_KEY_LANG = "batik_v3_language";

export default function HomePage() {
  const [currentTab, setCurrentTab] = useState<"scanner" | "atlas" | "history" | "about">("scanner");
  const [language, setLanguage] = useState<AppLanguage>("id");
  const [historyRecords, setHistoryRecords] = useState<ScanRecord[]>([]);
  const [targetAtlasQuery, setTargetAtlasQuery] = useState<string | undefined>(undefined);

  // Load persisted language and history on mount
  useEffect(() => {
    try {
      const savedLang = localStorage.getItem(LOCAL_STORAGE_KEY_LANG) as AppLanguage | null;
      if (savedLang === "id" || savedLang === "en") {
        setLanguage(savedLang);
      }
      const savedHistory = localStorage.getItem(LOCAL_STORAGE_KEY_HISTORY);
      if (savedHistory) {
        setHistoryRecords(JSON.parse(savedHistory));
      }
    } catch {
      // LocalStorage access fallback
    }
  }, []);

  const handleLanguageChange = (lang: AppLanguage) => {
    setLanguage(lang);
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY_LANG, lang);
    } catch {
      // LocalStorage fallback
    }
  };

  const handleSaveScanRecord = (recordData: {
    thumbnail: string;
    predictions: PredictionResult[];
    inferenceTimeMs: number;
  }) => {
    const newRecord: ScanRecord = {
      id: "scan_" + Date.now() + "_" + Math.random().toString(36).substring(2, 7),
      timestamp: Date.now(),
      imageThumbnail: recordData.thumbnail,
      topPrediction: recordData.predictions[0],
      topCandidates: recordData.predictions,
      inferenceTimeMs: recordData.inferenceTimeMs,
    };

    setHistoryRecords((prev) => {
      const updated = [newRecord, ...prev].slice(0, 30);
      try {
        localStorage.setItem(LOCAL_STORAGE_KEY_HISTORY, JSON.stringify(updated));
      } catch {
        // Fallback
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
    <div className="min-h-screen bg-[var(--canvas)] text-[var(--on-canvas)] flex flex-col justify-between py-0 sm:py-6 px-0 sm:px-4">
      {/* App Container */}
      <div className="w-full max-w-[1180px] mx-auto border border-[var(--canvas-line-strong)] rounded-md overflow-hidden bg-[var(--canvas)] shadow-2xl flex flex-col min-h-[90vh]">
        {/* Batik Border Strip */}
        <div className="motif-strip" />

        {/* Navigation Header */}
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
        />

        {/* Main Content Viewport */}
        <main className="flex-1 w-full bg-[var(--canvas)]">
          {currentTab === "scanner" && (
            <FabricScanner
              language={language}
              onSaveScanRecord={handleSaveScanRecord}
              onNavigateToAtlas={handleNavigateToAtlas}
            />
          )}

          {currentTab === "atlas" && (
            <CulturalAtlas
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
            />
          )}

          {currentTab === "about" && (
            <AboutSection language={language} />
          )}
        </main>

        {/* Clean Footer */}
        <footer className="flex flex-col sm:flex-row items-center justify-between px-6 py-4 border-t border-[var(--canvas-line)] text-[11.5px] text-[var(--on-canvas-faint)] gap-2 bg-[var(--canvas-deep)]">
          <span>{t(language, "footer_text")}</span>
          <span className="tnum">{t(language, "footer_meta")}</span>
        </footer>
      </div>
    </div>
  );
}
