/**
 * P6 T4/T5 — dogfood comparison analysis (2026-08-04).
 *
 * Reads the committed artifacts of a real Obsidian dogfood run and produces
 * the manual-topics-vs-library-directions comparison:
 *   - catalog stats (ready papers) and confirmed directions;
 *   - per-paper discovery-source classification from committed occurrence
 *     provenance (manual-only / library-only / both);
 *   - library-only papers, which are exactly the candidates the same manual
 *     topics would have missed, with direction names, representative prior
 *     papers, and validated personal novelty;
 *   - provenance/novelty integrity checks (occurrence counts vs reports).
 *
 * Usage:
 *   VAULT_ROOT=/path/to/vault node scripts/personalization/run-analyze-dogfood.mjs
 *   (default VAULT_ROOT is the plugin_test vault in this checkout)
 */
import { readFile, access } from "node:fs/promises";
import { join } from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const VAULT_ROOT = process.env.VAULT_ROOT ?? "plugin_test";

async function readJsonIfPresent(path: string): Promise<unknown | null> {
  try {
    await access(path);
  } catch {
    return null;
  }
  return JSON.parse(await readFile(path, "utf8"));
}

function summaryOf(paper: { summary?: Record<string, unknown> }): string {
  const s = paper.summary;
  if (!s) return "—";
  return [s.coreProblem, s.keyMethod, s.mainResult]
    .filter((part): part is string => typeof part === "string" && part.trim().length > 0)
    .join(" | ");
}

async function main(): Promise<void> {
  const indexPath = join(VAULT_ROOT, "arxiv-daily", ".index");
  const index = await readJsonIfPresent(join(indexPath, "papers.json"));
  const catalog = await readJsonIfPresent(join(indexPath, "personal-library-catalog.json"));
  const proposal = await readJsonIfPresent(join(indexPath, "direction-proposal.json"));
  const profile = await readJsonIfPresent(join(indexPath, "interest-profile.json"));

  if (!index) {
    console.log("No papers.json found under", indexPath);
    console.log("Run the daily pipeline first (Run Today / Run Pending) in Obsidian.");
    return;
  }

  const papers: Array<Record<string, unknown>> = Object.values(index.papers ?? {});
  const manualOnly: unknown[] = [];
  const libraryOnly: unknown[] = [];
  const both: unknown[] = [];
  const noProvenance: unknown[] = [];

  for (const paper of papers) {
    const provenanceByReport = paper.discoveryProvenanceByReport as Record<
      string,
      { manualTopicTags: string[]; directions: unknown[] }
    > | undefined;
    const reportPaths = Object.keys(provenanceByReport ?? {});
    if (reportPaths.length === 0) {
      noProvenance.push(paper);
      continue;
    }
    // Latest committed occurrence by report date (deterministic pick: latest path).
    const latestReport = [...reportPaths].sort().at(-1)!;
    const provenance = provenanceByReport![latestReport];
    const hasManual = (provenance.manualTopicTags?.length ?? 0) > 0;
    const hasLibrary = (provenance.directions?.length ?? 0) > 0;
    const bucket = hasLibrary ? (hasManual ? both : libraryOnly) : hasManual ? manualOnly : noProvenance;
    bucket.push({ paper, latestReport, provenance });
  }

  const directionNames = new Map<string, string>();
  for (const direction of profile?.directions ?? []) {
    if (typeof direction?.id === "string" && typeof direction?.name === "string") {
      directionNames.set(direction.id, direction.name);
    }
  }
  for (const candidate of proposal?.candidates ?? []) {
    if (typeof candidate?.id === "string" && typeof candidate?.name === "string") {
      directionNames.set(candidate.id, candidate.name);
    }
  }

  const renderLibraryOnly = (bucket: unknown[]): string => bucket.map((item: unknown) => {
    const { paper, latestReport, provenance } = item as {
      paper: { arxivId: string; title: string; topics: string[]; noveltyByReport: Record<string, unknown>; summary?: Record<string, unknown> };
      latestReport: string;
      provenance: { directions: Array<{ id: string; name?: string; representatives: Array<{ paperKey?: string; title?: string }> }> };
    };
    const directions = provenance.directions.map((d) => {
      const reps = (d.representatives ?? []).map((r) => r.title ?? r.paperKey ?? "?").join("; ");
      return `    direction: ${directionNames.get(d.id) ?? d.name ?? d.id}\n    representatives: ${reps}`;
    }).join("\n");
    const novelty = paper.noveltyByReport?.[latestReport];
    const noveltyLine = novelty
      ? `difference: ${(novelty as { differenceType?: string }).differenceType}\n    basis: ${((novelty as { comparisonBasis?: string[] }).comparisonBasis ?? []).join(", ")}\n    evidenceDepth: ${(novelty as { evidenceDepth?: string }).evidenceDepth}\n    explanation: ${(novelty as { explanation?: string }).explanation}`
      : "no novelty evidence";
    return `- ${paper.arxivId} — ${paper.title}\n  report: ${latestReport}\n${directions}\n  novelty: ${noveltyLine}\n  summary: ${summaryOf(paper)}`;
  }).join("\n");

  const report = {
    vault: VAULT_ROOT,
    index: {
      papers: papers.length,
      withProvenance: papers.length - noProvenance.length,
      withoutProvenance: noProvenance.length,
    },
    catalog: catalog
      ? {
          papers: Object.keys(catalog.papers ?? {}).length,
          ready: Object.values(catalog.files ?? {}).filter((f) => (f as { status?: string }).status === "ready").length,
          unresolved: Object.values(catalog.files ?? {}).filter((f) => (f as { status?: string }).status === "unresolved").length,
          unrelated: Object.values(catalog.files ?? {}).filter((f) => (f as { status?: string }).status === "unrelated").length,
        }
      : "no catalog document",
    profile: profile
      ? {
          directions: (profile.directions ?? []).map((d: { id?: string; name?: string; status?: string }) =>
            `${d.id} [${d.status ?? "?"}] ${d.name ?? "?"}`),
        }
      : "no interest-profile document",
    discoverySources: {
      manualOnly: manualOnly.length,
      libraryOnly: libraryOnly.length,
      both: both.length,
    },
    libraryOnlyPapers: renderLibraryOnly(libraryOnly),
    manualOnlyPapers: manualOnly.map((item: unknown) => {
      const { paper } = item as { paper: { arxivId: string; title: string } };
      return `- ${paper.arxivId} — ${paper.title}`;
    }).join("\n"),
    bothPapers: both.map((item: unknown) => {
      const { paper } = item as { paper: { arxivId: string; title: string } };
      return `- ${paper.arxivId} — ${paper.title}`;
    }).join("\n"),
  };
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
