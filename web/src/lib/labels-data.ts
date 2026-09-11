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

export function getMotifRegion(m: BatikClass, lang: AppLanguage): string {
  if (lang === "en") {
    return m.region_en || m.region;
  }
  return m.region_id || m.region;
}

export function getMotifIsland(m: BatikClass, lang: AppLanguage): string {
  if (lang === "en") {
    return m.island_en || m.island;
  }
  return m.island_id || m.island;
}

export interface IslandOption {
  key: string;
  label: string;
}

export function getIslandOptions(lang: AppLanguage): IslandOption[] {
  const map = new Map<string, string>();
  MOTIFS.forEach((m) => {
    const key = m.island_id || m.island;
    const label = lang === "en" ? m.island_en || m.island : m.island_id || m.island;
    if (key && !map.has(key)) {
      map.set(key, label);
    }
  });

  return Array.from(map.entries())
    .sort((a, b) => a[1].localeCompare(b[1]))
    .map(([key, label]) => ({ key, label }));
}

export function getAllIslands(): string[] {
  const islands = new Set<string>();
  MOTIFS.forEach((m) => {
    if (m.island) islands.add(m.island);
  });
  return Array.from(islands).sort();
}

export function getMotifsByIsland(islandKey: string): BatikClass[] {
  if (!islandKey || islandKey === "ALL") return MOTIFS;
  return MOTIFS.filter((m) => {
    const key = m.island_id || m.island;
    return key.toLowerCase() === islandKey.toLowerCase() || m.island.toLowerCase() === islandKey.toLowerCase();
  });
}

export function searchMotifs(query: string, lang: AppLanguage): BatikClass[] {
  const q = query.trim().toLowerCase();
  if (!q) return MOTIFS;

  return MOTIFS.filter((m) => {
    const name = lang === "id" ? m.name_id : m.name_en;
    const desc = lang === "id" ? m.description_id : m.description_en;
    const region = getMotifRegion(m, lang);
    const island = getMotifIsland(m, lang);
    return (
      name.toLowerCase().includes(q) ||
      region.toLowerCase().includes(q) ||
      island.toLowerCase().includes(q) ||
      desc.toLowerCase().includes(q) ||
      m.slug.toLowerCase().includes(q)
    );
  });
}
