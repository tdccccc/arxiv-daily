/**
 * Opt-in check against a real knowledge base and a real embedding endpoint.
 *
 * The rest of the suite proves retrieval behaviour on fixtures; this proves the
 * same code answers sensibly over a researcher's actual corpus, with a negative
 * control so "topically coherent" cannot be satisfied by always returning the
 * corpus's head papers.
 *
 *   REAL_KB_DIR=<.../personal-library-knowledge-base/<scope>/<ident>> \
 *     npm test -- tests/real-corpus-retrieval.test.ts
 *
 * REAL_EMBED_URL and REAL_EMBED_MODEL must name the same model the knowledge
 * base was built with, or the dense channel compares vectors from two different
 * spaces. Skipped entirely when REAL_KB_DIR is unset.
 */
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  decodeFullTextPaperDocument,
  fusePaperRankingsRrf,
  searchKnowledgeBase,
  searchKnowledgeBaseBm25,
  type FullTextPaperDocument,
} from "../src/index";

const KB = process.env.REAL_KB_DIR;
const EMBED_URL = process.env.REAL_EMBED_URL ?? "http://127.0.0.1:11434/v1/embeddings";
const EMBED_MODEL = process.env.REAL_EMBED_MODEL ?? "nomic-embed-text";

async function embed(text: string): Promise<Float32Array> {
  const res = await fetch(EMBED_URL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: EMBED_MODEL, input: [text] }),
  });
  const json = (await res.json()) as { data: { embedding: number[] }[] };
  return Float32Array.from(json.data[0].embedding);
}

function loadCorpus(dir: string): FullTextPaperDocument[] {
  const papers: FullTextPaperDocument[] = [];
  let skipped = 0;
  for (const name of readdirSync(join(dir, "papers"))) {
    if (!name.endsWith(".json") || name === "manifest.json") continue;
    const parsed: unknown = JSON.parse(readFileSync(join(dir, "papers", name), "utf8"));
    const doc = decodeFullTextPaperDocument(parsed);
    if (doc) papers.push(doc);
    else skipped += 1;
  }
  // eslint-disable-next-line no-console
  console.log(`corpus: ${papers.length} papers decoded, ${skipped} rejected`);
  return papers;
}

function report(
  label: string,
  matches: readonly { paperKey: string; score: number; title?: string; hits?: readonly { page?: number; text?: string }[] }[],
  titles?: ReadonlyMap<string, string>,
) {
  // eslint-disable-next-line no-console
  console.log(`\n--- ${label} ---`);
  for (const [i, m] of matches.slice(0, 5).entries()) {
    const hit = m.hits?.[0];
    // eslint-disable-next-line no-console
    console.log(
      `${i + 1}. ${(m.title ?? titles?.get(m.paperKey) ?? m.paperKey).replace(/\s+/g, " ").slice(0, 72)}  score=${m.score.toFixed(4)}` +
        (hit ? `\n   p.${hit.page ?? "?"}: ${String(hit.text ?? "").replace(/\s+/g, " ").slice(0, 110)}` : ""),
    );
  }
}

describe.skipIf(!KB)("real knowledge base retrieval", () => {
  it("returns topically coherent results through dense, lexical and hybrid channels", async () => {
    const papers = loadCorpus(KB as string);
    expect(papers.length).toBeGreaterThan(0);

    const titles = new Map<string, string>();
    for (const paper of papers) if (paper.title) titles.set(paper.paperKey, paper.title);

    const run = async (query: string) => {
      const queryVector = await embed(query);
      expect(queryVector.length).toBe(papers[0].dimension);
      const dense = searchKnowledgeBase({ papers, queryVector, limit: 10 });
      const lexical = searchKnowledgeBaseBm25({ papers, queryText: query, titles, limit: 10 });
      const hybrid = fusePaperRankingsRrf({ rankings: [dense, lexical], limit: 10 });
      // eslint-disable-next-line no-console
      console.log(`\n================ query: ${JSON.stringify(query)}`);
      report("dense", dense, titles);
      report("lexical (BM25)", lexical, titles);
      report("hybrid (RRF)", hybrid, titles);
      expect(dense.length).toBeGreaterThan(0);
      expect(lexical.length).toBeGreaterThan(0);
      expect(hybrid.length).toBeGreaterThan(0);
      for (const m of hybrid) expect(m.hits?.length ?? 0).toBeGreaterThan(0);
      return hybrid;
    };

    const a = await run("galaxy morphology classification with deep learning");
    const b = await run("stellar population synthesis and star formation history");

    // Negative control: an unrelated query must not return the same ranking,
    // otherwise "topically coherent" would just be the corpus's head papers.
    const overlap = a.slice(0, 5).filter((m) => b.slice(0, 5).some((n) => n.paperKey === m.paperKey));
    // eslint-disable-next-line no-console
    console.log(`\ntop-5 overlap between the two queries: ${overlap.length}/5`);
    expect(overlap.length).toBeLessThan(5);
  }, 120_000);
});
