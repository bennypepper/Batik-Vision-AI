"use client";

import React, { useState, useEffect } from "react";
import { AppLanguage, ScanRecord, PredictionResult } from "@/types/batik";
import { Navbar } from "@/components/Navbar";
import { FabricScanner } from "@/components/FabricScanner";
import { CulturalAtlas } from "@/components/CulturalAtlas";
import { ScanHistory } from "@/components/ScanHistory";
import { AboutSection } from "@/components/AboutSection";

const LOCAL_STORAGE_KEY_HISTORY = "batik_v3_scan_history";
const LOCAL_STORAGE_KEY_LANG = "batik_v3_language";

export default function HomePage() {
  const [currentTab, setCurrentTab] = useState<"scanner" | "atlas" | "history" | "about">("scanner");
  const [language, setLanguage] = useState<AppLanguage>("id");
  const [historyRecords, setHistoryRecords] = useState<ScanRecord[]>([]);

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

  return (
    <div className="min-h-screen flex flex-col bg-stone-950 text-stone-100 font-sans">
      {/* Navigation Header */}
      <Navbar
        currentTab={currentTab}
        onTabChange={setCurrentTab}
        language={language}
        onLanguageChange={handleLanguageChange}
      />

      {/* Main Content Viewport */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12">
        {currentTab === "scanner" && (
          <FabricScanner
            language={language}
            onSaveScanRecord={handleSaveScanRecord}
          />
        )}

        {currentTab === "atlas" && (
          <CulturalAtlas language={language} />
        )}

        {currentTab === "history" && (
          <ScanHistory
            records={historyRecords}
            language={language}
            onClearHistory={handleClearHistory}
          />
        )}

        {currentTab === "about" && (
          <AboutSection language={language} />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-stone-900/60 border-t border-stone-800/80 py-8 text-xs text-stone-500 text-center space-y-2">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-stone-400">
            Batik Vision AI v3 | Indonesian Batik Motifs Recognition Engine
          </p>
          <div className="flex items-center space-x-4 text-stone-500 font-mono text-[11px]">
            <span>Model: EfficientNetV2S</span>
            <span>Classes: 28</span>
            <span>Runtime: WebAssembly/WebGL</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
