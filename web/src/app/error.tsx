"use client";

import React, { useEffect } from "react";
import Link from "next/link";
import { AlertTriangle, RotateCcw, Home } from "lucide-react";

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Application error boundary caught:", error);
  }, [error]);

  return (
    <div className="min-h-screen bg-[var(--canvas)] text-[var(--on-canvas)] flex flex-col">
      <div className="motif-strip" />
      <main className="flex-1 flex items-center justify-center p-6">
        <div className="max-w-md w-full bg-[var(--cloth)] text-[var(--on-cloth)] rounded-2xl p-8 border border-[var(--cloth-line)] shadow-sm text-center space-y-5">
          <div className="w-12 h-12 rounded-full bg-[var(--mahogany-tint)] text-[var(--mahogany)] flex items-center justify-center mx-auto">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <div className="space-y-2">
            <h1 className="display text-2xl font-semibold text-[var(--on-cloth)]">
              Something went wrong
            </h1>
            <p className="text-sm text-[var(--on-cloth-soft)] leading-relaxed">
              An unexpected error occurred while rendering Batik Vision AI. Your
              local scan history remains safe in your browser.
            </p>
          </div>
          <div className="flex items-center justify-center gap-3 pt-2">
            <button
              type="button"
              onClick={reset}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-[var(--soga)] hover:bg-[var(--soga-deep)] text-white text-xs font-semibold transition cursor-pointer border-none"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Try Again</span>
            </button>
            <Link
              href="/"
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-lg bg-[var(--cloth-deep)] hover:bg-[var(--canvas-deep)] text-[var(--on-cloth)] text-xs font-medium border border-[var(--cloth-line-strong)] transition no-underline"
            >
              <Home className="w-3.5 h-3.5" />
              <span>Back to Scanner</span>
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
