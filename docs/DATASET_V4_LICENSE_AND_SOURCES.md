# Licensing, Provenance & Source Attribution (v2.0)

## Combined Dataset License
This curated v2.0 benchmark is released under **CC BY-SA 4.0** (`CC-BY-SA-4.0`) to honor the attribution and share-alike terms of its contributing public archives (including Wikimedia Commons `CC BY-SA 4.0` museum textiles and Roboflow / Mendeley `CC BY 4.0` datasets).

> **Important Notice for Downstream Users:** Every image in `metadata.csv` carries its exact `source` identifier so commercial or strict-compliance users can filter by upstream license if desired (for example, keeping only `CC0-1.0` + `Apache-2.0` + `CC BY 4.0` + `Public Domain` rows).

---

## Contributing Batik Sources (`3,470` Final Curated Images)

All **21** contributing sources that survived multi-stage deduplication (`MD5`, 8-way D4-dihedral `dHash <= 3`, Roboflow `.rf.` stem collapse, and `DINOv2 cosine < 0.94`), resolution filtering, and `SigLIP-SO400M` + OOF noise filtering:

| Source ID | Dataset / Archive | Platform | License | Images Retained | Notes |
|---|---|---|---|---:|---|
| `existing_local` | [v1 Base Release (fisheightcharacter / Batik-Indonesia by M. Salman Al Faridzi)](https://www.kaggle.com/datasets/fisheightcharacter/indonesian-batik-dataset-enhanced-and-cleaned) | Kaggle / Hugging Face | Apache-2.0 | **1136** | 28-class v1 base dataset derived from Hugging Face Batik-Indonesia + manually verified v1 expansions (deduplicated & noise-filtered). |
| `kaggle_buyungsaloka_motif_batik` | [motif-batik-dataset (buyungsaloka)](https://www.kaggle.com/datasets/buyungsaloka/motif-batik-dataset) | Kaggle | Unlisted on Kaggle (attributed; verify upstream terms for commercial use) | **608** | Major contributor for regional motifs: Bali_Barong, Sulawesi_Selatan_Lontara, Madura_Mataketeran, Jakarta_Ondel_Ondel, Sumatera_Barat_Rumah_Minang, Jawa_Timur_Pring. |
| `kaggle_alfanme_corak_app` | [Indonesian Batik Motifs (Corak App) (alfanme)](https://www.kaggle.com/datasets/alfanme/indonesian-batik-motifs-corak-app) | Kaggle | CC0-1.0 | **420** | Unambiguous motifs mapped (Cendrawasih, Dayak, Ikat Celup, Insang, Kawung, Lasem, Megamendung, Pala, Sekar Jagad, Tambal); generic Parang/Bali/Betawi excluded. |
| `Batik_v6i_multiclass` | [Batik Multiclass v6 (Roboflow Universe — a-co6il/batik-ifddz)](https://universe.roboflow.com/a-co6il/batik-ifddz) | Roboflow Universe | CC BY 4.0 | **296** | 1,491 synthetic .rf. augmentation duplicates stripped prior to inclusion; mapped Ikat+Celup, Corak+Insang, Pring+Sedapur, Dayak, Kawung, Megamendung. |
| `Batik_Indonesia_Skripsi_v52i_yolov8` | [Batik Indonesia Skripsi v52 (Roboflow Universe — aku-cinta-indonesia/batik-indonesia-skripsi)](https://universe.roboflow.com/aku-cinta-indonesia/batik-indonesia-skripsi) | Roboflow Universe | CC BY 4.0 | **241** | 1,270 synthetic .rf. augmentation duplicates stripped; single-class tight bounding boxes cropped when bbox < 70% frame area. |
| `kaggle_sendytegar_batik_keraton` | [Dataset Batik Keraton (sendytegar)](https://www.kaggle.com/datasets/sendytegar/dataset-batik-keraton) | Kaggle | CC0-1.0 | **159** | Contributed Kawung, Megamendung, and high-precision DINOv2-mined Solo_Parang images. |
| `Indonesia_Batik_v7_folder` | [Indonesia Batik Classification v7 (Roboflow Universe — melanie-gabriela-tjandrasubrata)](https://universe.roboflow.com/melanie-gabriela-tjandrasubrata/indonesia-batik-classification-mktx6) | Roboflow Universe | CC BY 4.0 | **118** | 112 synthetic .rf. duplicates stripped; mapped Kawung, Mega-Mendung, Sekar-Jagad, Sidoluhur. |
| `gh_dionisiusfb_batik` | [batik-motif-classification (DionisiusFB)](https://github.com/DionisiusFB/batik-motif-classification) | GitHub | Public GitHub Repository (academic) | **110** | Contributed Ceplok and Kawung motifs. |
| `kaggle_dionisiusdh_batik_motifs` | [Indonesian Batik Motifs (dionisiusdh)](https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs) | Kaggle | CC0-1.0 | **97** | Contributed Ceplok, Sogan, Sidoluhur, Lasem, Tambal, Sekar, Ikat_Celup, Cendrawasih; generic Bali/Betawi/Parang/Priangan excluded. |
| `kaggle_brendhaadiyanvianca_batik_indonesian` | [Batik Indonesian (brendhaadiyanvianca)](https://www.kaggle.com/datasets/brendhaadiyanvianca/batik-indonesian) | Kaggle | Apache-2.0 | **78** | Explicit Yogyakarta_Parang, JawaTengah_Lasem, Kalimantan_Insang, Yogyakarta_SekarJagad, Yogyakarta_Tambal. |
| `kaggle_syahdanputra_batiksnap` | [BatikSnap Dataset (syahdanputra)](https://www.kaggle.com/datasets/syahdanputra/batiksnap-dataset) | Kaggle | DbCL-1.0 | **62** | Contributed Insang, Kawung, Megamendung, Sidoluhur. |
| `Batik_Jawa_Barat_v6-5classes_yolov8` | [Batik Jawa Barat v6 (Roboflow Universe — batikjabar/batik-jawa-barat)](https://universe.roboflow.com/batikjabar/batik-jawa-barat) | Roboflow Universe | CC BY 4.0 | **50** | 126 synthetic .rf. duplicates stripped; primary source for Priangan_Merak_Ngibing expansion (+39 net-new) + Megamendung. |
| `Mendeley_Lasem_GunungRinggit_Originals` | [Batik Lasem Motifs Collections (Gunung Ringgit, Kricak, Latohan, Nyuk Pitu, Seritan) — Full-Res Originals](https://data.mendeley.com/datasets/kw5b7d9zgn) | Mendeley Data | CC BY 4.0 | **31** | ONLY full-resolution original workshop textile photographs retained (tiny 28x28 and 142x142 isen-isen patches excluded). |
| `wikimedia_batik` | [Wikimedia Commons Museum Textile Archives (Collectie NMvWereldculturen & open-licensed batik)](https://commons.wikimedia.org/) | Wikimedia Commons | Public Domain / CC BY-SA 4.0 | **25** | High-resolution museum & archival batik photos (Sekar Jagad, Yogyakarta Parang, Megamendung, Papua Cendrawasih). |
| `kaggle_fisheightcharacter_batik_enhanced` | [Indonesian Batik Dataset (Enhanced & Cleaned) v1 Delta](https://www.kaggle.com/datasets/fisheightcharacter/indonesian-batik-dataset-enhanced-and-cleaned) | Kaggle | Apache-2.0 | **20** | Net-new unique Priangan_Merak_Ngibing images present in the v1 Kaggle release. |
| `hf_salman_batik_indonesia` | [Batik-Indonesia (Muhammad Salman Al Faridzi)](https://huggingface.co/datasets/muhammadsalmanalfaridzi/Batik-Indonesia) | Hugging Face | Apache-2.0 | **7** | Original upstream backbone for v1; 7 additional non-overlapping images retained after deduplication. |
| `kaggle_hendryhb_batik_nusantara` | [Batik Nusantara (Batik Indonesia) Dataset (hendryhb)](https://www.kaggle.com/datasets/hendryhb/batik-nusantara-batik-indonesia-dataset) | Kaggle | CC0-1.0 | **6** | 90.6% overlap with v1 base removed; 6 net-new clean images retained. |
| `gh_reyyichiro_batik` | [batik-motif-classification (ReyyIchiro)](https://github.com/ReyyIchiro/batik-motif-classification) | GitHub | Public GitHub Repository (academic) | **3** | 3 high-precision DINOv2-mined Solo_Parang images retained after strict dedup. |
| `mendeley_lasem_kw5b7d9zgn` | [Batik Lasem Motifs Collections (Mendeley API mirror kw5b7d9zgn)](https://data.mendeley.com/datasets/kw5b7d9zgn) | Mendeley Data | CC BY 4.0 | **1** | 1 full-res Lasem textile photograph passing >=200px & dedup gates. |
| `kaggle_yohanpermanautm_batik_madura` | [Citra Batik Madura & Flora (yohanpermanautm)](https://www.kaggle.com/datasets/yohanpermanautm/citra-batik-madura-and-flora) | Kaggle | Apache-2.0 | **1** | 1 high-precision DINOv2-mined Madura_Mataketeran image passing >=0.82 prob & >=0.35 margin gate. |
| `kaggle_fiin28_batik_papua` | [Batik Papua (fiin28)](https://www.kaggle.com/datasets/fiin28/batik-papua) | Kaggle | MIT | **1** | 1 high-precision DINOv2-mined Papua_Tifa image passing >=0.82 prob & >=0.35 margin gate. |

### Audited Sources With 0 Retained Images (100% Overlap or Quarantined Ambiguous)
The following public datasets were also audited against the master pool and contributed **0** retained images because `100%` of their mapped images were already exact/D4 duplicates of higher-resolution copies in earlier sources:
- `dwibudisantoso/batik-wastra-nusantara` (Kaggle, CC0-1.0) — 100% overlap with `dionisiusdh` & v1 base.
- `panduagas/batik-jawa-barat` (Kaggle) — 100% overlap with `alfanme/indonesian-batik-motifs-corak-app`.

---

## Wikimedia Commons Batik Museum Credits (`25` Retained Images)
Full title, license, and Wikimedia Commons URL provenance for the archival batik photos (primarily *Collectie Nationaal Museum van Wereldculturen*, Public Domain pre-1891 batik drawings & `CC BY-SA 4.0` modern textiles):

| Class | Wikimedia File Title | License |
|---|---|---|
| `Yogyakarta_Parang` | [File:Collectie NMvWereldculturen, RV-847-117, Batikpatroon, 'Parang rusak putri', voor 1891.jpg](https://upload.wikimedia.org/wikipedia/commons/c/c4/Collectie_NMvWereldculturen%2C_RV-847-117%2C_Batikpatroon%2C_%27Parang_rusak_putri%27%2C_voor_1891.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | Public domain |
| `Yogyakarta_Parang` | [File:Collectie NMvWereldculturen, RV-847-77, Batikpatroon, 'Parang rusak barong', voor 1891.jpg](https://upload.wikimedia.org/wikipedia/commons/0/04/Collectie_NMvWereldculturen%2C_RV-847-77%2C_Batikpatroon%2C_%27Parang_rusak_barong%27%2C_voor_1891.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | Public domain |
| `Yogyakarta_Parang` | [File:Collectie NMvWereldculturen, RV-847-78, Batikpatroon, 'Parang rusak putra', voor 1891.jpg](https://upload.wikimedia.org/wikipedia/commons/8/82/Collectie_NMvWereldculturen%2C_RV-847-78%2C_Batikpatroon%2C_%27Parang_rusak_putra%27%2C_voor_1891.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | Public domain |
| `Yogyakarta_Parang` | [File:Collectie NMvWereldculturen, RV-847-79, Batikpatroon, 'Parang rusak alit', voor 1891.jpg](https://upload.wikimedia.org/wikipedia/commons/7/77/Collectie_NMvWereldculturen%2C_RV-847-79%2C_Batikpatroon%2C_%27Parang_rusak_alit%27%2C_voor_1891.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | Public domain |
| `Yogyakarta_Parang` | [File:Collectie NMvWereldculturen, RV-847-80, Batikpatroon, 'Parang rusak gendrèh', voor 1891.jpg](https://upload.wikimedia.org/wikipedia/commons/0/05/Collectie_NMvWereldculturen%2C_RV-847-80%2C_Batikpatroon%2C_%27Parang_rusak_gendr%C3%A8h%27%2C_voor_1891.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | Public domain |
| `Yogyakarta_Parang` | [File:Collectie NMvWereldculturen, RV-847-82, Batikpatroon, 'Parang rusak klitik', voor 1891.jpg](https://upload.wikimedia.org/wikipedia/commons/3/3f/Collectie_NMvWereldculturen%2C_RV-847-82%2C_Batikpatroon%2C_%27Parang_rusak_klitik%27%2C_voor_1891.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | Public domain |
| `Sekar` | [File:Batik Madura18.jpg](https://upload.wikimedia.org/wikipedia/commons/e/ee/Batik_Madura18.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Batik Madura 11.jpg](https://upload.wikimedia.org/wikipedia/commons/3/3a/Batik_Madura_11.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Batik Madura 15.jpg](https://upload.wikimedia.org/wikipedia/commons/2/20/Batik_Madura_15.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Batik Madura 16.jpg](https://upload.wikimedia.org/wikipedia/commons/e/e1/Batik_Madura_16.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Batik Madura 26.jpg](https://upload.wikimedia.org/wikipedia/commons/a/a7/Batik_Madura_26.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Batik Madura 48.jpg](https://upload.wikimedia.org/wikipedia/commons/9/97/Batik_Madura_48.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Batik Madura 50.jpg](https://upload.wikimedia.org/wikipedia/commons/8/87/Batik_Madura_50.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Batik Madura 52.jpg](https://upload.wikimedia.org/wikipedia/commons/1/16/Batik_Madura_52.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Batik Madura 53.jpg](https://upload.wikimedia.org/wikipedia/commons/8/8c/Batik_Madura_53.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Batik Madura 54.jpg](https://upload.wikimedia.org/wikipedia/commons/2/2a/Batik_Madura_54.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Batik Madura 56.jpg](https://upload.wikimedia.org/wikipedia/commons/1/1b/Batik_Madura_56.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Batik Madura 7.jpg](https://upload.wikimedia.org/wikipedia/commons/5/5e/Batik_Madura_7.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Batik Madura 8.jpg](https://upload.wikimedia.org/wikipedia/commons/d/d6/Batik_Madura_8.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Detail batik motif sekar jagad warna merah khas pekalongan.jpg](https://upload.wikimedia.org/wikipedia/commons/6/65/Detail_batik_motif_sekar_jagad_warna_merah_khas_pekalongan.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Kain batik motif sekar jagad.jpg](https://upload.wikimedia.org/wikipedia/commons/6/67/Kain_batik_motif_sekar_jagad.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Kain batik tulis Lasem di seri sarung.jpg](https://upload.wikimedia.org/wikipedia/commons/f/fc/Kain_batik_tulis_Lasem_di_seri_sarung.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Motif Batik Madura 14.jpg](https://upload.wikimedia.org/wikipedia/commons/d/de/Motif_Batik_Madura_14.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Motif Batik Madura 55.jpg](https://upload.wikimedia.org/wikipedia/commons/4/49/Motif_Batik_Madura_55.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Sekar` | [File:Motif Batik Madura 9.jpg](https://upload.wikimedia.org/wikipedia/commons/4/4b/Motif_Batik_Madura_9.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Jawa_Barat_Megamendung` | [File:Batik Mega Mendung.jpg](https://upload.wikimedia.org/wikipedia/commons/8/83/Batik_Mega_Mendung.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |
| `Jawa_Barat_Megamendung` | [File:Stamp of Indonesia - 2011 - Colnect 691620 - Batik Mega Mendung Jawa Barat.jpeg](https://upload.wikimedia.org/wikipedia/commons/3/33/Stamp_of_Indonesia_-_2011_-_Colnect_691620_-_Batik_Mega_Mendung_Jawa_Barat.jpeg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | Public domain |
| `Papua_Cendrawasih` | [File:Batik Papua Cendrawasih.jpg](https://upload.wikimedia.org/wikipedia/commons/a/ab/Batik_Papua_Cendrawasih.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original) | CC BY-SA 4.0 |

---

## Out-of-Distribution (OOD) Negative Calibration Set (`177` Images)
All `177` OOD negative images in `ood_negatives/` were sourced from Wikimedia Commons under open licenses (`Public Domain`, `CC0`, `No restrictions`, `CC BY 2.0/3.0/4.0`, `CC BY-SA 2.0/3.0/4.0`). Full image-level title, license, and source URL attribution is provided in `ood_negatives/ood_manifest.csv`.

| OOD Bucket | Count | Purpose |
|---|---:|---|
| `indoor_textures` | 50 | Wood, brick, marble, carpets, tiles — prevents texture false positives |
| `indo_non_batik_textiles` | 48 | **Hard OOD negatives**: Indonesian *Songket*, *Tenun Ikat*, *Ulos* (woven heritage textiles that are NOT wax-resist batik!) |
| `non_batik_patterns` | 40 | Plaid, tartan, paisley, stripes, floral wallpaper |
| `everyday_objects` | 27 | Common household / street scenes |
| `plain_garments` | 7 | Solid-color shirts, denim, plain fabrics |
| `portraits_people` | 5 | Human portraits / selfies without batik motifs |
| **Total OOD Negatives** | **177** | For Temperature Scaling & MSP / Energy OOD threshold calibration |
