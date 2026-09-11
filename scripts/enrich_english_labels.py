"""Enriches labels.json with comprehensive English descriptions for all 28 motifs."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_MODELS = REPO_ROOT / "models" / "labels.json"
LABELS_WEB = REPO_ROOT / "web" / "src" / "data" / "labels.json"

EN_DESCRIPTIONS = {
    "Bali_Barong": "Inspired by Barong, the mythological king of spirits in Balinese Hinduism representing the eternal cosmic force of virtue (Dharma) against Rangda (Adharma). The motif showcases the crowned lion-like spirit with intricate flame ornaments and sacred Balinese floral flourishes. Revered as a spiritual shield against malice, it is traditionally worn during sacred temple ceremonies and festive rites.",
    "Bali_Merak": "Depicts the regal peacock unfurling its magnificent plumage, an enduring Balinese symbol of elegance, aesthetic beauty, and aristocratic poise. Often combined with frangipani blossoms and lush tropical greenery, this motif radiates vitality and vibrant chromatic harmony. It is widely worn during ceremonial celebrations, formal weddings, and performing arts.",
    "Ceplok": "One of the oldest symmetrical geometric patterns in Javanese textile heritage, composed of repeated rosettes, circles, squares, or star polygons. Derived from the Javanese concept of cut fruit showing radial symmetry, Ceplok signifies universal order, equilibrium, and cosmic balance. Historical variants like Ceplok Grompol and Sriwedari convey distinct blessings for communal harmony and marriage.",
    "Corak_Insang": "Traditional coastal motif from the Malay community of Pontianak along the majestic Kapuas River in West Kalimantan. Inspired by the undulating gills and scales of freshwater fish, the parallel Chevron bands celebrate the riverine ecosystem as the wellspring of sustenance. It symbolizes flowing prosperity, resilience against tidal currents, and deep respect for aquatic heritage.",
    "Ikat_Celup": "Organic resist-dye textile artistry (popularly known as tie-dye or jumputan) crafted by binding selected sections of fabric with cords prior to dipping into natural dye baths. Lacking rigid wax stencils, Ikat Celup produces spontaneous chromatic ripples and ethereal tonal gradients. Each hand-tied piece is singular and irreproducible, celebrated across the archipelago as an expressive textile tradition.",
    "Jakarta_Ondel_Ondel": "Celebrates the quintessential cultural icon of Betawi folk heritage from Jakarta: the towering Ondel-Ondel effigies crafted to ward off misfortune and safeguard communities. Displaying vibrant depictions of male and female pairs framed by glittering coconut flower ornaments (kembang kelapa), the pattern features vibrant crimson and emerald hues embodying the spirited, expressive hospitality of Betawi culture.",
    "Jawa_Barat_Megamendung": "The iconic cloud formation of Cirebon, West Java, displaying rolling, tiered cloud billows with seven color gradations from deep indigo to ethereal white. These seven tiers symbolize the celestial heavens and stages of spiritual enlightenment. Philosophically, Megamendung counsels leaders to remain cool-headed, patient, and comforting, sheltering their people like clouds shielding the earth from scorching heat.",
    "Jawa_Timur_Pring": "Hailing from Magetan at the foot of Mount Lawu in East Java, Pring Sedapur depicts swaying bamboo groves. In Javanese philosophy, bamboo never grows in isolation but thrives in dense colonies, embodying communal solidarity (gotong royong), resilience, and humble strength that bends gracefully under tempestuous winds without breaking.",
    "Kalimantan_Dayak": "Depicts the sacred Batang Garing (Tree of Life) central to Dayak cosmology and Kaharingan philosophy in the heart of Borneo. With deep roots sinking into the underworld and branches ascending toward the celestial firmament, it links humanity, nature, and the supreme creator (Ranying Hatalla). Rendered in earthy forest pigments, it signifies sacred equilibrium and cosmic harmony.",
    "Lampung_Gajah": "Honors the majestic Sumatran elephant native to the Way Kambas conservation rainforests of Lampung. Portrayed with ceremonial regalia and paired with the golden Siger horn headdress, this pattern symbolizes sovereign dignity, wisdom, noble endurance, and the urgent ecological imperative to preserve Sumatra's endangered wildlife.",
    "Lasem": "Renowned as the Sino-Javanese coastal masterpiece from Central Java, Lasem batik intertwines Chinese mythological icons such as the phoenix (Lok Can) and dragon with classic Javanese floral and geometric foliage. Dyed using natural mengkudu roots (producing the legendary 'getih pitik' chicken-blood red), it stands as a celebrated historical testament to centuries of harmonious cultural acculturation.",
    "Madura_Mataketeran": "Distinctive Madurese coastal batik motif depicting bird-eye geometrics and vibrant coastal foliage. Reflecting the passionate, assertive temperament of the Madurese people, Mataketeran features dramatic contrasts of fiery reds, deep blues, and mustard yellows, celebrated for its bold freehand waxing technique and uninhibited creative vitality.",
    "Maluku_Pala": "Commemorates the legendary spice heritage of the Maluku Banda Islands, the historic spice islands that reshaped global maritime trade. The pattern depicts split nutmeg fruits (pala) with crimson mace (fuli) nestled within glossy leaves, symbolizing sovereign abundance, fragrant nobility, and enduring pride in eastern Indonesian agricultural heritage.",
    "NTB_Lumbung": "From West Nusa Tenggara (Lombok and Sumbawa), this motif honors the traditional Lumbung (Uma Lengge) thatch-roofed rice granaries. Rising on sturdy stilts to protect the harvest, the granary embodies food security, collective foresight, and familial gratitude to the earth for agricultural prosperity.",
    "Papua_Asmat": "Draws from the profound woodcarving traditions of the Asmat people along the southwest coast of Papua. Featuring stylized spiraling ancestors, paddle shields, and hornbill talons in dramatic iron-red, ochre, and charcoal, the motif serves as a sacred bridge between the living community and the ancestral spirit realm.",
    "Papua_Cendrawasih": "Celebrates the Bird of Paradise (Paradisaea apoda), the sacred emblem of Papua renowned for its shimmering plumage and majestic courtship display. Revered as the 'Bird of the Gods' fallen from paradise, the motif symbolizes celestial elegance, natural sovereignty, and the enchanting biological richness of Papuan primeval forests.",
    "Papua_Tifa": "Features the sacred hourglass-shaped Tifa drum, the traditional percussion instrument crafted from hollowed hardwood and monitor lizard skin. Embellished with regional Papuan tribal patterns, it symbolizes rhythmic communal unity, festive joy, and the heartbeat that accompanies tribal ceremonies and welcoming dances.",
    "Priangan_Merak_Ngibing": "Originating from the Priangan highlands of West Java (Garut and Tasikmalaya), this motif captures a dancing peacock ('ngibing') amidst delicate Sundanese flora. It symbolizes graceful celebration, refined joy, and youthful beauty. (Note: Machine learning classification accuracy is currently limited on this class due to high intra-class stylistic diversity).",
    "Sekar": "A celebrated coastal motif from the Sekar Jagad lineage, meaning 'the floral map of the universe'. Composed of patchy floral clusters separated by organic borders resembling islands, it celebrates the kaleidoscopic diversity of flora across the archipelago, symbolizing unity in diversity and universal beauty.",
    "Sidoluhur": "A revered royal (Keraton) motif from Central Java, derived from 'Sido' (to become/continuous) and 'Luhur' (noble and exalted). Historically worn during matrimonial rites, it bestows prayers that the wearer will achieve noble character, spiritual dignity, and enduring honor in serving society.",
    "Sogan": "The classic aristocratic palette and pattern of the Central Javanese royal courts (Surakarta and Yogyakarta), dyed exclusively with natural bark extract from the Soga tree (Peltophorum pterocarpum) alongside indigo. With deep ochre and chocolate hues, Sogan symbolizes humility, earthliness, and grounded wisdom. (Note: Model classification accuracy is currently developing due to subtle stylistic overlap with regional variants).",
    "Solo_Parang": "The legendary diagonal blade motif of the Surakarta Hadiningrat Court. The rhythmic, slanted S-curves evoke crashing ocean waves pounding against rocky sea cliffs. It symbolizes unyielding determination, relentless perseverance, continuous self-improvement, and royal valor in overcoming the storms of life.",
    "Sulawesi_Selatan_Lontara": "Inspired by the ancestral Lontara syllabic script and geometric motifs of the Bugis and Makassarese seafaring people of South Sulawesi. Reflecting deep philosophical principles of Siri' na Pacce (honor, dignity, and communal empathy), the sharp angular glyphs celebrate literacy, navigational courage, and maritime heritage.",
    "Sumatera_Barat_Rumah_Minang": "Depicts the soaring swooping gables of the Minangkabau Rumah Gadang (buffalo horn roofs) framed by intricate Pucuak Rabuang (bamboo sprout) woodcarvings. Reflecting matrilineal wisdom, democratic consensus (musyawarah), and the principle that nature is the supreme teacher ('Alam Takambang Jadi Guru').",
    "Sumatera_Utara_Boraspati": "Honors the sacred Boraspati ni Tano (earth monitor lizard) of the Batak people around Lake Toba, North Sumatra. Frequently carved upon the wooden facades of Batak Gorga dwellings and woven in Ulos, Boraspati symbolizes agricultural fertility, maternal nurture, and enduring protection over family and granary.",
    "Tambal": "Derived from the Javanese word for 'patchwork', this motif combines diverse geometric remnants into a unified mosaic. Historically wrapped around ailing individuals, Tambal symbolizes restorative healing, repair of spiritual flaws, and the gathering of virtues to mend what has been broken.",
    "Yogyakarta_Kawung": "One of the most ancient royal sacred motifs of the Yogyakarta Sultanate, formed by four-lobed intersecting elliptical spheres with a central seed, representing the fruit of the sugar palm (Aren / Kolang-kaling). Symbolizing the four cardinal directions and cosmic center, Kawung embodies immaculate purity, spiritual self-restraint, and righteous service to the community.",
    "Yogyakarta_Parang": "The iconic sacred Parang Rusak motif of the Yogyakarta Royal Palace (Kraton Ngayogyakarta). Characterized by unbroken diagonal knife-like waves with Mlinjon diamond accents, it was historically reserved strictly for royalty (Awisan Dalem). It signifies the sovereign spirit mastering its primal impulses through continuous moral vigilance."
}


def update_file(path: Path):
    if not path.exists():
        print(f"Skipping non-existent {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data["classes"]:
        slug = item["slug"]
        if slug in EN_DESCRIPTIONS:
            item["description_en"] = EN_DESCRIPTIONS[slug]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {path} successfully.")


def main():
    update_file(LABELS_MODELS)
    update_file(LABELS_WEB)


if __name__ == "__main__":
    main()
