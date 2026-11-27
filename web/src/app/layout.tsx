import type { Metadata } from "next";
import { Fraunces, Work_Sans } from "next/font/google";
import "./globals.css";

const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
  style: ["normal", "italic"],
  display: "swap",
});

const workSans = Work_Sans({
  variable: "--font-work-sans",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Batik Vision AI v3 | Indonesian Batik Motifs Recognition Engine",
  description:
    "Two-stage Indonesian batik motif classification powered by adaptive fabric localization and EfficientNetV2S running client-side via WebGPU, WebGL, and WebAssembly.",
};

const themeInitScript = `(function(){try{var t=localStorage.getItem("batik_v3_theme");if(t==="dark"||t==="light"){document.documentElement.setAttribute("data-theme",t);}var l=localStorage.getItem("batik_v3_language");if(l==="id"||l==="en"){document.documentElement.lang=l;}}catch(e){}})();`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      data-theme="light"
      suppressHydrationWarning
      className={`${fraunces.variable} ${workSans.variable} h-full antialiased`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body className="min-h-full flex flex-col font-sans">{children}</body>
    </html>
  );
}
