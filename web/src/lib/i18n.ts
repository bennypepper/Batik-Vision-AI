import { AppLanguage } from "@/types/batik";

export const DICTIONARY = {
  id: {
    app_title: "Batik Vision AI v3",
    app_tagline: "Klasifikasi Motif Batik Nusantara Dua Tahap Berbasis WebAssembly",
    nav_scanner: "Pindai Motif",
    nav_atlas: "Atlas Budaya",
    nav_history: "Riwayat",
    nav_about: "Tentang Model",

    // Scanner
    scanner_title: "Deteksi dan Klasifikasi Motif Batik",
    scanner_desc: "Unggah foto pakaian atau kain batik. Sistem akan mendeteksi area kain melalui YOLO dan mengklasifikasikan motifnya melalui EfficientNetV2S.",
    upload_prompt: "Tarik dan lepas gambar di sini, atau klik untuk memilih",
    upload_sub: "Mendukung JPG, PNG, WEBP (Maksimal 10MB)",
    btn_camera: "Gunakan Kamera",
    btn_stop_camera: "Tutup Kamera",
    btn_capture: "Ambil Foto",
    btn_browse: "Pilih File",
    stage1_label: "Tahap 1: Deteksi Area Kain (YOLO ROI)",
    stage1_sub: "Sesuaikan kotak pembatas untuk memfokuskan pola motif batik sebelum klasifikasi.",
    btn_auto_crop: "Crop Otomatis",
    btn_reset_crop: "Reset Kotak",
    btn_classify: "Klasifikasikan Motif",
    btn_reset: "Pindai Ulang",
    status_loading_model: "Memuat model AI ke memori browser (WebAssembly)...",
    status_detecting_fabric: "Menjalankan deteksi kain batik tahap 1...",
    status_classifying: "Menganalisis tekstur motif dengan EfficientNetV2S...",
    status_ready: "Model siap. Silakan unggah atau potret kain batik.",

    // Results
    top_prediction: "Hasil Prediksi Utama",
    confidence: "Tingkat Keyakinan",
    heritage_origin: "Asal Budaya",
    region: "Daerah",
    island: "Pulau",
    philosophy: "Makna dan Filosofi Budaya",
    top_candidates: "5 Kemungkinan Teratas",
    rank: "Peringkat",
    probability: "Probabilitas",
    scan_again: "Pindai Kain Lain",
    save_history: "Simpan ke Riwayat",
    saved_to_history: "Tersimpan",

    // Cultural Atlas
    atlas_title: "Atlas Budaya Batik Nusantara",
    atlas_desc: "Eksplorasi 28 motif batik tradisional dari seluruh kepulauan Indonesia lengkap dengan filosofi dan makna luhurnya.",
    search_placeholder: "Cari nama motif, daerah, pulau, atau filosofi...",
    filter_island: "Filter Berdasarkan Kepulauan:",
    all_islands: "Semua Pulau",
    motifs_found: "motif ditemukan",
    weak_class_badge: "Motif Langka",
    view_details: "Lihat Filosofi Lengkap",
    close_modal: "Tutup",

    // History
    history_title: "Riwayat Pemindaian",
    history_desc: "Daftar motif batik yang telah Anda pindai dalam sesi ini.",
    no_scans: "Belum ada riwayat pemindaian.",
    clear_history: "Hapus Semua Riwayat",

    // About
    about_title: "Arsitektur dan Metodologi Batik Vision v3",
    about_desc: "Solusi perbaikan mendasar model visi komputer klasifikasi batik nusantara.",
    v3_pipeline_title: "Perbaikan Utama Pipeline v3",
    v3_p1: "Penyelesaian Masalah Double-Preprocessing: Pada v2, gambar dinormalisasi dua kali sehingga rentang dinamis runtuh menjadi [-1.008, -0.992]. Pada v3, model EfficientNetV2S secara native menerima piksel float32 [0, 255] dengan lapisan Rescaling internal.",
    v3_p2: "Deduplikasi Perceptual Hash: Menghilangkan 799 pasangan duplikat data antar fold dan split pengujian menggunakan 64-bit dHash.",
    v3_p3: "Focal Categorical Crossentropy: Memprioritaskan kelas minoritas langka (seperti Priangan Merak Ngibing) dengan parameter gamma=2.0 dan bobot sampel efektif.",
    v3_p4: "Deteksi Dua Tahap: YOLO mendeteksi kain pakaian dalam foto bebas sebelum klasifikasi tekstur motif dilakukan.",
    zero_backend_title: "Inferensi Klien Tanpa Server (Zero-Backend)",
    zero_backend_desc: "Model berjalan langsung di dalam peramban web pengguna menggunakan ONNX Runtime WebAssembly dan WebGL. Data foto tidak dikirim ke server eksternal, menjamin privasi penuh dan latensi instan.",
  },
  en: {
    app_title: "Batik Vision AI v3",
    app_tagline: "Client-Side Two-Stage Indonesian Batik Classification with WebAssembly",
    nav_scanner: "Scan Motif",
    nav_atlas: "Cultural Atlas",
    nav_history: "History",
    nav_about: "About Model",

    // Scanner
    scanner_title: "Batik Motif Detection and Classification",
    scanner_desc: "Upload a photo of batik clothing or fabric. The system detects fabric regions via YOLO and classifies patterns using EfficientNetV2S.",
    upload_prompt: "Drag and drop your image here, or click to browse",
    upload_sub: "Supports JPG, PNG, WEBP (Up to 10MB)",
    btn_camera: "Use Camera",
    btn_stop_camera: "Close Camera",
    btn_capture: "Capture Photo",
    btn_browse: "Browse Files",
    stage1_label: "Stage 1: Fabric Region of Interest (YOLO ROI)",
    stage1_sub: "Adjust the bounding box to isolate the motif pattern before classification.",
    btn_auto_crop: "Auto Crop",
    btn_reset_crop: "Reset Box",
    btn_classify: "Classify Motif",
    btn_reset: "Scan Another",
    status_loading_model: "Loading AI model into browser memory (WebAssembly)...",
    status_detecting_fabric: "Running stage 1 fabric detection...",
    status_classifying: "Analyzing motif texture with EfficientNetV2S...",
    status_ready: "Model ready. Upload or capture a batik fabric photo.",

    // Results
    top_prediction: "Primary Prediction Result",
    confidence: "Confidence Score",
    heritage_origin: "Cultural Heritage Origin",
    region: "Region / City",
    island: "Island",
    philosophy: "Symbolism and Cultural Philosophy",
    top_candidates: "Top 5 Candidates",
    rank: "Rank",
    probability: "Probability",
    scan_again: "Scan Another Fabric",
    save_history: "Save to History",
    saved_to_history: "Saved",

    // Cultural Atlas
    atlas_title: "Indonesian Batik Cultural Atlas",
    atlas_desc: "Explore 28 traditional batik motifs across the Indonesian archipelago, complete with their philosophical heritage and history.",
    search_placeholder: "Search motif name, region, island, or symbolism...",
    filter_island: "Filter by Archipelago Region:",
    all_islands: "All Islands",
    motifs_found: "motifs found",
    weak_class_badge: "Rare Motif",
    view_details: "View Full Heritage",
    close_modal: "Close",

    // History
    history_title: "Scan History",
    history_desc: "Batik motifs you have classified during this session.",
    no_scans: "No scan history recorded yet.",
    clear_history: "Clear Scan History",

    // About
    about_title: "Batik Vision v3 Architecture and Methodology",
    about_desc: "Foundational computer vision overhaul for Indonesian batik classification.",
    v3_pipeline_title: "Key V3 Pipeline Fixes",
    v3_p1: "Double-Preprocessing Resolution: Fixed v2 bug where images were normalized twice, collapsing the dynamic range to [-1.008, -0.992]. V3 natively accepts raw [0, 255] float32 pixels with an explicit top-level Rescaling layer.",
    v3_p2: "Perceptual Hash Deduplication: Eliminated 799 duplicate image pairs across train/test splits using 64-bit dHash.",
    v3_p3: "Focal Categorical Crossentropy: Downweights easy dominant classes to prioritize rare minority classes (gamma=2.0, label smoothing=0.1, effective sample weighting).",
    v3_p4: "Two-Stage Detection: YOLO isolates the fabric region from unconstrained garment photos before texture classification.",
    zero_backend_title: "Zero-Backend Client-Side Runtime",
    zero_backend_desc: "The model runs directly inside the browser using ONNX Runtime WebAssembly and WebGL. No images are transmitted to external servers, ensuring privacy and instant latency.",
  },
};

export function t(lang: AppLanguage, key: keyof typeof DICTIONARY["en"]): string {
  return DICTIONARY[lang][key] || DICTIONARY["en"][key] || key;
}
