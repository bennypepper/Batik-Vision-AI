import { AppLanguage } from "@/types/batik";

export const DICTIONARY = {
  id: {
    app_title: "Batik Vision AI",
    app_version: "v4",
    app_subtitle:
      "Klasifikasi Motif Batik Nusantara - Dua Tahap dengan DINOv2 & WebGPU/WASM",
    nav_scanner: "Pindai Motif",
    nav_atlas: "Atlas Budaya",
    nav_history: "Riwayat",
    nav_about: "Tentang Model",

    // Scanner
    scanner_title: "Deteksi dan Klasifikasi Motif Batik",
    scanner_desc:
      "Unggah foto pakaian atau kain batik. Sistem melokalisasi area kain melalui Lokalisasi Tekstur Adaptif, lalu mengklasifikasikan motifnya menggunakan DINOv2-ViT-S/14 (INT8) — seluruhnya berjalan di peramban Anda.",
    step_yolo: "Lokalisasi kain (Tekstur Adaptif)",
    step_classifier: "Klasifikasi motif (DINOv2-ViT-S/14 INT8)",
    upload_prompt: "Tarik dan lepas gambar di sini, atau klik untuk memilih",
    upload_sub: "Mendukung JPG, PNG, WEBP (maksimal 10MB)",
    btn_browse: "Pilih File",
    btn_camera: "Gunakan Kamera",
    btn_stop_camera: "Tutup Kamera",
    btn_capture: "Ambil Foto",
    fabric_detected: "Area kain terlokalisasi",
    conf_confident: "yakin",
    see_in_atlas: "Lihat filosofi lengkap di Atlas Budaya ->",
    scan_again: "Pindai Gambar Lain",
    save_history: "Simpan ke Riwayat",
    saved_to_history: "Tersimpan",
    stage1_label: "Tahap 1: Lokalisasi Area Kain (ROI Adaptif)",
    stage1_sub:
      "Sesuaikan kotak pembatas untuk memfokuskan pola motif batik sebelum klasifikasi.",
    btn_auto_crop: "Crop Otomatis",
    btn_reset_crop: "Reset Kotak",
    btn_classify: "Klasifikasikan Motif",
    framing_hint:
      "Motif batik terdeteksi otomatis. Untuk foto pakaian, sesuaikan kotak agar membingkai motif secara terfokus.",
    preset_full_swatch: "Pola Penuh (100%)",
    preset_focus_center: "Fokus Tengah (50%)",
    hint_swatch_detected:
      "Terdeteksi kain merata (close-up). Pola penuh (100%) aktif otomatis untuk akurasi terbaik.",
    hint_garment_detected:
      "Terdeteksi foto pakaian atau manekin. Lokalisasi adaptif memfokuskan area motif utama kain.",
    btn_adjust_roi: "Sesuaikan Bingkai Motif (ROI)",
    status_loading_model:
      "Memuat model AI ke memori peramban (WebGPU/WebGL/WASM)...",
    status_downloading_model: "Mengunduh bobot model ONNX...",
    status_detecting_fabric: "Melokalisasi area kain...",
    status_classifying: "Menganalisis tekstur motif di Web Worker...",
    model_error_title: "Model AI tidak dapat dimuat",
    model_error_retry: "Coba Muat Ulang",
    file_too_large: "Ukuran file melebihi batas maksimal 10MB.",
    invalid_file_type: "Format file tidak didukung. Harap unggah gambar (JPG, PNG, WEBP).",
    image_load_failed: "Gambar gagal dibaca. Harap coba file lain.",

    // Results
    top_prediction: "Hasil Klasifikasi Utama",
    confidence: "Keyakinan",
    heritage_origin: "Asal Budaya",
    region: "Daerah",
    island: "Pulau",
    philosophy: "Makna Filosofis",
    top_candidates: "5 Kemungkinan Teratas",
    rank: "Peringkat",
    probability: "Probabilitas",
    ood_warning_title: "Keyakinan Rendah — Kemungkinan Bukan Motif Batik",
    ood_warning_desc:
      "Distribusi probabilitas sangat menyebar atau keyakinan model di bawah ambang batas. Pastikan area kain batik terbingkai jelas dengan pencahayaan cukup, atau gambar ini bukan salah satu dari 28 motif batik yang dikenal.",
    provider_webgpu: "WebGPU",
    provider_webgl: "WebGL",
    provider_wasm: "WebAssembly",

    // Cultural Atlas
    atlas_title: "Atlas Budaya Batik Nusantara",
    atlas_desc:
      "Eksplorasi 28 motif batik tradisional dari seluruh kepulauan Indonesia, lengkap dengan filosofi dan makna luhurnya.",
    search_placeholder: "Cari nama motif, daerah, pulau, atau filosofi...",
    filter_island: "Pulau:",
    all_islands: "Semua Pulau",
    motifs_found: "motif ditemukan",
    rare_badge: "Motif Langka",
    model_weak_note: "akurasi klasifikasi model masih rendah pada motif ini",
    read_full_philosophy: "Baca filosofi lengkap",
    close_modal: "Tutup",

    // History
    history_title: "Riwayat Pemindaian",
    history_desc:
      "Semua pemindaian tersimpan secara lokal di perangkat Anda - tidak pernah dikirim ke server manapun.",
    saved_scans: "pemindaian tersimpan",
    no_scans: "Belum ada riwayat pemindaian.",
    no_scans_prompt:
      "Hasil klasifikasi kain batik Anda akan otomatis tersimpan di sini.",
    btn_start_scan: "Mulai Pindai Kain",
    clear_history: "Kosongkan riwayat",
    view_detail_link: "Lihat detail",

    // About
    about_title: "Arsitektur dan Metodologi Batik Vision v4",
    about_desc:
      "Catatan rekayasa nyata dari v2 ke v4 — meliputi kurasi dataset bebas kebocoran, benchmark 3 arsitektur, kalibrasi OOD, dan kuantisasi INT8.",
    v3_p1_title: "Backbone DINOv2-ViT-S/14 INT8 (22.5 MB)",
    v3_p1_tag: "v3 -> v4",
    v3_p1_desc:
      "Menggantikan EfficientNetV2S FP32 (82.1 MB) dengan DINOv2-ViT-S/14 terkuantisasi INT8 (22.5 MB, 3.65x lebih kecil) dengan normalisasi [0, 255] NHWC tertanam langsung di grafik ONNX — mencapai 89.82% Top-1 / 96.01% Top-3 / 87.47% Macro-F1 pada uji holdout bebas kebocoran.",
    v3_p2_title: "Kurasi dataset bebas kebocoran (3.470 gambar)",
    v3_p2_desc:
      "Mengaudit 22 sumber publik (22.077 gambar mentah) dengan deduplikasi MD5 + D4-dHash + DINOv2 (cos >= 0.94) dan filter kualitas SigLIP, serta pemisahan StratifiedGroupKFold berdasarkan 3.140 klaster semantik.",
    v3_p3_title: "Kalibrasi Temperature & deteksi OOD (AUROC 0.959)",
    v3_p3_desc:
      "Temperature Scaling (T=0.824) menurunkan ECE menjadi 0.039 dan dikalibrasi terhadap 177 gambar negatif non-batik (termasuk Songket, Tenun, dan Ulos) untuk menolak foto bukan batik secara jujur.",
    v3_p4_title: "Arsitektur dua tahap & multi-patch voting",
    v3_p4_desc:
      "Tahap 1 menggunakan Lokalisasi Tekstur Adaptif berbasis kisi Laplacian 6x6 untuk melokalisasi area kain, diikuti ekstraksi 3 tile persegi 1:1 dengan voting bobot entropi terbalik di DINOv2-ViT-S/14.",
    zero_backend_title: "Inferensi klien tanpa server (zero-backend)",
    zero_backend_desc:
      "Model berjalan langsung di dalam Web Worker peramban Anda menggunakan ONNX Runtime (WebGPU, WebGL, dan WebAssembly self-hosted). Foto yang Anda unggah tidak pernah dikirim ke server eksternal manapun - privasi penuh, latensi instan.",

    // Developer & Platform
    web_overview_title: "Tentang Platform",
    web_overview_desc:
      "Batik Vision AI adalah inisiatif pelestarian warisan budaya tekstil Nusantara melalui teknologi kecerdasan buatan. Menggabungkan Lokalisasi Tekstur Adaptif dan klasifikasi DINOv2-ViT-S/14 INT8 terkalibrasi OOD, platform ini bekerja 100% di sisi klien menggunakan WebGPU/WebAssembly untuk menjaga privasi mutlak pengguna.",
    dev_title: "Tentang Pengembang",
    dev_name: "Benny Pepper",
    dev_role: "AI / Machine Learning Engineer & Full-Stack Developer",
    dev_bio:
      "Fokus pada arsitektur computer vision, deep learning edge, dan sistem web modern. Berdedikasi melestarikan dan mendokumentasikan kekayaan motif batik Nusantara melalui rekayasa data yang teliti dan sistem terbuka.",
    github_profile: "Profil GitHub",
    github_repo: "Repositori Proyek",
    theme_light: "Mode Terang (Mori)",
    theme_dark: "Mode Gelap (Wedelan)",
    toggle_theme: "Ganti Tema",
    cultural_provenance: "Asal daerah",
    visual_identity: "Karakteristik visual",
    deep_philosophy: "Makna filosofis",

    // Footer
    footer_text: "Batik Vision AI · Indonesian Batik Motifs Recognition Engine",
    footer_meta: "DINOv2-ViT-S/14 INT8 (22.5 MB) · 28 kelas · WebGPU/WebGL/WASM",
  },
  en: {
    app_title: "Batik Vision AI",
    app_version: "v4",
    app_subtitle:
      "Client-Side Two-Stage Indonesian Batik Classification via DINOv2 & WebGPU/WASM",
    nav_scanner: "Scan Motif",
    nav_atlas: "Cultural Atlas",
    nav_history: "History",
    nav_about: "About Model",

    // Scanner
    scanner_title: "Batik Motif Detection and Classification",
    scanner_desc:
      "Upload a photo of batik clothing or fabric. The system isolates the fabric area via Adaptive Texture Localization, then classifies the pattern using DINOv2-ViT-S/14 (INT8) — running entirely in your browser.",
    step_yolo: "Fabric localization (Adaptive Texture)",
    step_classifier: "Motif classification (DINOv2-ViT-S/14 INT8)",
    upload_prompt: "Drag and drop your image here, or click to browse",
    upload_sub: "Supports JPG, PNG, WEBP (maximum 10MB)",
    btn_browse: "Browse Files",
    btn_camera: "Use Camera",
    btn_stop_camera: "Close Camera",
    btn_capture: "Capture Photo",
    fabric_detected: "Fabric region localized",
    conf_confident: "confident",
    see_in_atlas: "View full philosophy in Cultural Atlas ->",
    scan_again: "Scan Another Image",
    save_history: "Save to History",
    saved_to_history: "Saved",
    stage1_label: "Stage 1: Adaptive Fabric Region of Interest (ROI)",
    stage1_sub:
      "Adjust the bounding box to isolate the motif pattern before classification.",
    btn_auto_crop: "Auto Crop",
    btn_reset_crop: "Reset Box",
    btn_classify: "Classify Motif",
    framing_hint:
      "Fabric pattern automatically framed. For clothing or full-body photos, fine-tune the box to focus closely on the motif.",
    preset_full_swatch: "Full Swatch (100%)",
    preset_focus_center: "Focus Center (50%)",
    hint_swatch_detected:
      "Detected close-up fabric swatch. Full pattern (100%) automatically selected for optimal accuracy.",
    hint_garment_detected:
      "Detected clothing or mannequin photo. Adaptive localization focused on the primary garment fabric region.",
    btn_adjust_roi: "Fine-tune Pattern Box (ROI)",
    status_loading_model:
      "Loading AI model into browser memory (WebGPU/WebGL/WASM)...",
    status_downloading_model: "Downloading ONNX model weights...",
    status_detecting_fabric: "Localizing fabric region...",
    status_classifying: "Analyzing motif texture in Web Worker...",
    model_error_title: "AI Model Unavailable",
    model_error_retry: "Retry Loading Model",
    file_too_large: "File size exceeds the 10MB limit.",
    invalid_file_type:
      "Unsupported file format. Please upload an image (JPG, PNG, WEBP).",
    image_load_failed: "Failed to decode image. Please try another file.",

    // Results
    top_prediction: "Primary Prediction Result",
    confidence: "Confidence",
    heritage_origin: "Cultural Origin",
    region: "Region",
    island: "Island",
    philosophy: "Philosophical Meaning",
    top_candidates: "Top 5 Candidates",
    rank: "Rank",
    probability: "Probability",
    ood_warning_title: "Low Confidence — Possibly Not Batik",
    ood_warning_desc:
      "Prediction entropy is high or calibrated confidence is below threshold. Ensure the batik fabric is clearly framed with good lighting, or this image may not belong to one of the 28 recognized motifs.",
    provider_webgpu: "WebGPU",
    provider_webgl: "WebGL",
    provider_wasm: "WebAssembly",

    // Cultural Atlas
    atlas_title: "Indonesian Batik Cultural Atlas",
    atlas_desc:
      "Explore 28 traditional batik motifs from across the Indonesian archipelago, complete with their philosophical heritage and history.",
    search_placeholder: "Search motif name, region, island, or philosophy...",
    filter_island: "Island:",
    all_islands: "All Islands",
    motifs_found: "motifs found",
    rare_badge: "Rare Motif",
    model_weak_note:
      "model classification accuracy is currently low on this motif",
    read_full_philosophy: "Read full philosophy",
    close_modal: "Close",

    // History
    history_title: "Scan History",
    history_desc:
      "All scans are stored locally on your device - never transmitted to any external server.",
    saved_scans: "saved scans",
    no_scans: "No scans recorded yet.",
    no_scans_prompt:
      "Your classified batik textile scans will appear here automatically.",
    btn_start_scan: "Start Fabric Scan",
    clear_history: "Clear history",
    view_detail_link: "View details",

    // About
    about_title: "Batik Vision v4 Architecture and Methodology",
    about_desc:
      "Concrete engineering decisions from v2 to v4 — leak-free dataset curation, 3-backbone benchmarking, OOD calibration, and INT8 quantization.",
    v3_p1_title: "DINOv2-ViT-S/14 INT8 backbone (22.5 MB)",
    v3_p1_tag: "v3 -> v4",
    v3_p1_desc:
      "Replaced EfficientNetV2S FP32 (82.1 MB) with INT8-quantized DINOv2-ViT-S/14 (22.5 MB, 3.65x smaller) embedding [0, 255] NHWC preprocessing inside ONNX — achieving 89.82% Top-1 / 96.01% Top-3 / 87.47% Macro-F1 on a strictly leak-free holdout split.",
    v3_p2_title: "Leak-free dataset curation (3,470 images)",
    v3_p2_desc:
      "Audited 22 public sources (22,077 raw images) using MD5 + 8-way D4-dHash + DINOv2 semantic deduplication (cos >= 0.94) and SigLIP noise filtering, with StratifiedGroupKFold splits across 3,140 semantic clusters.",
    v3_p3_title: "Temperature Scaling & OOD rejection (AUROC 0.959)",
    v3_p3_desc:
      "Post-hoc Temperature Scaling (T=0.824) reduced Test ECE to 0.039 and calibrated entropy/confidence rejection against 177 open-licensed non-batik negatives (including Songket, Tenun, and Ulos).",
    v3_p4_title: "Two-stage localization & multi-patch voting",
    v3_p4_desc:
      "Stage 1 uses 256px resolution-normalized 6x6 grid-Laplacian Adaptive Texture Localization, followed by 3 aspect-preserving 1:1 square tiles combined via inverse-entropy weighted voting.",
    zero_backend_title: "Client-side serverless inference (zero-backend)",
    zero_backend_desc:
      "The model runs directly inside a browser Web Worker using self-hosted ONNX Runtime (WebGPU, WebGL, and WebAssembly). Uploaded photos are never sent to any external server - complete privacy and zero server latency.",

    // Developer & Platform
    web_overview_title: "About the Platform",
    web_overview_desc:
      "Batik Vision AI is an open cultural preservation platform bridging traditional Indonesian textile heritage with edge AI. Powered by Adaptive Texture Localization and an OOD-calibrated INT8 DINOv2-ViT-S/14 classifier, it recognizes 28 batik motifs directly in your browser with zero server latency and absolute privacy.",
    dev_title: "About the Developer",
    dev_name: "Benny Pepper",
    dev_role: "AI / Machine Learning Engineer & Full-Stack Developer",
    dev_bio:
      "Specializing in computer vision architectures, edge deep learning, and modern web systems. Dedicated to preserving and documenting Indonesian textile heritage through rigorous data engineering and open platforms.",
    github_profile: "GitHub Profile",
    github_repo: "Project Repository",
    theme_light: "Light Mode (Mori)",
    theme_dark: "Dark Mode (Wedelan)",
    toggle_theme: "Toggle Theme",
    cultural_provenance: "Regional provenance",
    visual_identity: "Visual characteristics",
    deep_philosophy: "Cultural philosophy",

    // Footer
    footer_text: "Batik Vision AI · Indonesian Batik Motifs Recognition Engine",
    footer_meta: "DINOv2-ViT-S/14 INT8 (22.5 MB) · 28 classes · WebGPU/WebGL/WASM",
  },
};

export function t(
  lang: AppLanguage,
  key: keyof (typeof DICTIONARY)["en"]
): string {
  return DICTIONARY[lang][key] || DICTIONARY["en"][key] || key;
}
