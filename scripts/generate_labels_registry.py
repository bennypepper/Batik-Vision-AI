import json
import os
import sys

# Island mapping based on region or class
ISLAND_MAP = {
    'Bali': 'Bali',
    'Jawa Tengah': 'Jawa',
    'Kalimantan Barat': 'Kalimantan',
    'Nusantara': 'Nusantara',
    'DKI Jakarta': 'Jawa',
    'Jawa Barat': 'Jawa',
    'Jawa Timur': 'Jawa',
    'Kalimantan': 'Kalimantan',
    'Lampung': 'Sumatera',
    'Madura': 'Madura',
    'Maluku': 'Maluku',
    'Nusa Tenggara Barat': 'Nusa Tenggara',
    'Papua': 'Papua',
    'Jawa': 'Jawa',
    'Jawa Tengah / Yogyakarta': 'Jawa',
    'Solo / Surakarta': 'Jawa',
    'Sulawesi Selatan': 'Sulawesi',
    'Sumatera Barat': 'Sumatera',
    'Sumatera Utara': 'Sumatera',
    'Yogyakarta': 'Jawa',
}

ENGLISH_NAMES = {
    'Bali_Barong': 'Bali Barong',
    'Bali_Merak': 'Bali Peacock',
    'Ceplok': 'Ceplok Geometric',
    'Corak_Insang': 'Pontianak Gill Pattern',
    'Ikat_Celup': 'Tie-Dye Batik',
    'Jakarta_Ondel_Ondel': 'Jakarta Ondel-Ondel',
    'Jawa_Barat_Megamendung': 'Cirebon Cloud (Megamendung)',
    'Jawa_Timur_Pring': 'East Java Bamboo (Pring Sedapur)',
    'Kalimantan_Dayak': 'Dayak Tree of Life',
    'Lampung_Gajah': 'Lampung Elephant',
    'Lasem': 'Lasem Sino-Javanese',
    'Madura_Mataketeran': 'Madura Dove Eye',
    'Maluku_Pala': 'Maluku Nutmeg Spice',
    'NTB_Lumbung': 'Lombok Rice Barn (Lumbung)',
    'Papua_Asmat': 'Papua Asmat Woodcarving',
    'Papua_Cendrawasih': 'Papua Bird of Paradise',
    'Papua_Tifa': 'Papua Tifa Drum',
    'Priangan_Merak_Ngibing': 'Priangan Dancing Peacock',
    'Sekar': 'Sekar Jagad Floral Universe',
    'Sidoluhur': 'Sidoluhur Noble Attainment',
    'Sogan': 'Solo Classic Golden-Brown Sogan',
    'Solo_Parang': 'Solo Parang Wave',
    'Sulawesi_Selatan_Lontara': 'South Sulawesi Lontara Script',
    'Sumatera_Barat_Rumah_Minang': 'West Sumatra Minang House',
    'Sumatera_Utara_Boraspati': 'North Sumatra Boraspati Lizard',
    'Tambal': 'Tambal Patchwork Healing',
    'Yogyakarta_Kawung': 'Yogyakarta Aren Palm (Kawung)',
    'Yogyakarta_Parang': 'Yogyakarta Parang Rusak',
}

ENGLISH_DESCRIPTIONS = {
    'Bali_Barong': "Inspired by Barong, the mythological lion of Bali symbolizing virtue and spiritual protection in Hindu-Balinese traditions.",
    'Bali_Merak': "Depicts the grace and beauty of the peacock, blending Balinese floral ornaments with vibrant ceremonial aesthetics.",
    'Ceplok': "One of the oldest symmetrical geometric patterns in Java, representing universal balance and cosmic order.",
    'Corak_Insang': "Traditional Malay Pontianak motif inspired by fish gills and scales, honoring coastal Kapuas River ecology.",
    'Ikat_Celup': "Organic tie-dye fabric technique producing free-flowing color gradients without the use of wax stencils.",
    'Jakarta_Ondel_Ondel': "Features the iconic Betawi giant protective figures surrounded by festive coconut flower motifs.",
    'Jawa_Barat_Megamendung': "Cirebon cloud formation with seven color gradations representing seven cosmological sky tiers and patience.",
    'Jawa_Timur_Pring': "Depicts bamboo groves symbolizing communal solidarity, resilience, and harmony under life's challenges.",
    'Kalimantan_Dayak': "Portrays the sacred Dayak Batang Garing (Tree of Life) connecting cosmological realms in indigenous Kaharingan lore.",
    'Lampung_Gajah': "Showcases the majestic Sumatran elephant paired with golden Siger crowns denoting wisdom and regional pride.",
    'Lasem': "Celebrated coastal red batik reflecting rich Sino-Javanese acculturation featuring dragons, phoenixes, and sea flora.",
    'Madura_Mataketeran': "Coastal Madurese pattern inspired by dove eyes, expressing sharpness, expressiveness, and festive pride.",
    'Maluku_Pala': "Honors the legendary nutmeg spice that positioned Maluku as the historic center of world maritime commerce.",
    'NTB_Lumbung': "Represents the conical traditional rice barns (Uma Lengge) symbolizing agricultural abundance and thrift.",
    'Papua_Asmat': "Sacred geometric tribal patterns derived from world-renowned Asmat woodcarvings and ancestor totems.",
    'Papua_Cendrawasih': "Celebrates the Bird of Paradise, a sacred fauna embodying eternal beauty and Papuan cultural identity.",
    'Papua_Tifa': "Features the ceremonial wooden hourglass drum integral to Papuan dances, rituals, and communal unity.",
    'Priangan_Merak_Ngibing': "Depicts a dancing peacock pair from West Java symbolizing romance, marital grace, and fine hand-drawn craft.",
    'Sekar': "Ornate floral patchwork representing the manifold beauty of the universe and world diversity united.",
    'Sidoluhur': "Prestigious royal Javanese pattern conveying prayers for nobility, dignity, and upright moral character.",
    'Sogan': "Refined court batik of Surakarta characterized by warm golden-brown dyes extracted from the soga tree bark.",
    'Solo_Parang': "Historic continuous diagonal blade motif symbolizing unyielding perseverance like ocean waves beating the shore.",
    'Sulawesi_Selatan_Lontara': "Artistic integration of ancient Bugis-Makassar script celebrating literary scholarship and maritime heritage.",
    'Sumatera_Barat_Rumah_Minang': "Features the horned Gadang house roof reflecting matrilineal heritage and communal protection.",
    'Sumatera_Utara_Boraspati': "Portrays the mythical Boraspati earth lizard representing fertility, soil bounty, and Batak guardianship.",
    'Tambal': "Traditional patchwork motif used historically as a healing cloth uniting diverse prayers and pattern blessings.",
    'Yogyakarta_Kawung': "Ancient four-lobed aren fruit motif denoting the four cardinal directions, heart purity, and social utility.",
    'Yogyakarta_Parang': "Grand royal Parang Rusak motif worn by Sultans, embodying spiritual battle against inner temptation.",
}

def build_labels_registry(labels_txt_path: str = None, output_path: str = None) -> dict:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if labels_txt_path is None:
        labels_txt_path = os.path.join(repo_root, "models", "labels.txt")
    if output_path is None:
        output_path = os.path.join(repo_root, "models", "labels.json")

    sys.path.insert(0, repo_root)
    from src.utils.data import REGION_MAP, WEAK_CLASSES, BATIK_INFO

    with open(labels_txt_path, "r", encoding="utf-8") as f:
        slugs = [line.strip() for line in f if line.strip()]

    classes = []
    for idx, slug in enumerate(slugs):
        name_id = slug.replace("_", " ")
        name_en = ENGLISH_NAMES.get(slug, name_id)
        region = REGION_MAP.get(slug, "Indonesia")
        island = ISLAND_MAP.get(region, "Nusantara")
        is_weak = slug in WEAK_CLASSES
        desc_id = BATIK_INFO.get(slug, "")
        desc_en = ENGLISH_DESCRIPTIONS.get(slug, "")

        classes.append({
            "id": idx,
            "slug": slug,
            "name_id": name_id,
            "name_en": name_en,
            "region": region,
            "island": island,
            "is_weak_class": is_weak,
            "description_id": desc_id,
            "description_en": desc_en
        })

    registry = {
        "version": "3.0",
        "total_classes": len(classes),
        "classes": classes
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    return registry

if __name__ == "__main__":
    reg = build_labels_registry()
    print(f"Successfully generated labels.json with {reg['total_classes']} classes.")
