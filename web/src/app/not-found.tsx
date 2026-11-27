import React from "react";
import Link from "next/link";
import { Compass, ArrowLeft } from "lucide-react";

export default function NotFoundPage() {
  return (
    <div className="min-h-screen bg-[var(--canvas)] text-[var(--on-canvas)] flex flex-col">
      <div className="motif-strip" />
      <main className="flex-1 flex items-center justify-center p-6">
        <div className="max-w-md w-full bg-[var(--cloth)] text-[var(--on-cloth)] rounded-2xl p-8 border border-[var(--cloth-line)] shadow-sm text-center space-y-5">
          <div className="w-12 h-12 rounded-full bg-[var(--soga-tint)] text-[var(--soga)] flex items-center justify-center mx-auto">
            <Compass className="w-6 h-6" />
          </div>
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-widest text-[var(--soga)] tnum">
              404
            </p>
            <h1 className="display text-2xl font-semibold text-[var(--on-cloth)]">
              Page Not Found
            </h1>
            <p className="text-sm text-[var(--on-cloth-soft)] leading-relaxed">
              The motif or page you are looking for could not be found in the
              Batik Vision AI archive.
            </p>
          </div>
          <div className="pt-2">
            <Link
              href="/"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-[var(--soga)] hover:bg-[var(--soga-deep)] text-white text-xs font-semibold transition no-underline"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Return to Scanner</span>
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
