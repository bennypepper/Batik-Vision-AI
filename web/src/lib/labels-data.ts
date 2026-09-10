import rawLabels from "@/data/labels.json";
import { BatikClass, AppLanguage } from "@/types/batik";

export const MOTIFS: BatikClass[] = ((rawLabels as any).classes || rawLabels) as BatikClass[];

export function getAllMotifs(): BatikClass[] {
  return MOTIFS;
}

export function getMotifById(id: number): BatikClass | undefined {
  return MOTIFS.find((m) => m.id === id);
}

export function getMotifBySlug(slug: string): BatikClass | undefined {
  return MOTIFS.find((m) => m.slug.toLowerCase() === slug.toLowerCase());
}

export function getAllIslands(): string[] {
  const islands = new Set<string>();
  MOTIFS.forEach((m) => {
    if (m.island) islands.add(m.island);
  });
  return Array.from(islands).sort();
}

export function getMotifsByIsland(island: string): BatikClass[] {
  if (!island || island === "ALL") return MOTIFS;
  return MOTIFS.filter((m) => m.island.toLowerCase() === island.toLowerCase());
}

export function searchMotifs(query: string, lang: AppLanguage): BatikClass[] {
  const q = query.trim().toLowerCase();
  if (!q) return MOTIFS;

  return MOTIFS.filter((m) => {
    const name = lang === "id" ? m.name_id : m.name_en;
    const desc = lang === "id" ? m.description_id : m.description_en;
    return (
      name.toLowerCase().includes(q) ||
      m.region.toLowerCase().includes(q) ||
      m.island.toLowerCase().includes(q) ||
      desc.toLowerCase().includes(q) ||
      m.slug.toLowerCase().includes(q)
    );
  });
}
