/**
 * Deterministic lexical similarity between a query and a paper title.
 *
 * Why lexical, not embedding-based: remote embedding models observed in the
 * wild (Ollama's `nomic-embed-text` without instruction prefixes) collapse
 * short texts — two unrelated titles scored 0.93 while a title differing
 * from the query only in letter case scored 0.66. Embedding similarity is
 * therefore unusable as the title signal. Lexical matching is exact for the
 * dominant use case (typing a paper's title) and degrades gracefully for
 * partial titles, while staying at 0 for free-text queries so it never
 * disturbs passage retrieval.
 *
 * Scoring (all on normalized lowercase alphanumeric tokens):
 * - identical normalized text → 1
 * - the query is a token-prefix of the title → 0.95 (e.g. "BERT" →
 *   "BERT: Pre-training of …", where the collapse makes "learning" → the
 *   ResNet title fail because it is not a prefix)
 * - token-set Jaccard ≥ 0.5 → the Jaccard value (word-order/word-count
 *   variants of a title stay high; unrelated titles stay below the cutoff)
 * - otherwise → 0 (no evidence; passage retrieval decides)
 *
 * Pure and side-effect-free: the same inputs always produce the same score.
 */

/** Lowercase, strip to alphanumeric runs, collapse whitespace. */
export function normalizeTitleText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

/**
 * Similarity in [0, 1] between a query and a paper title; see the module
 * comment for the rules. Either side empty yields 0.
 */
export function lexicalTitleSimilarity(query: string, title: string): number {
  const normalizedQuery = normalizeTitleText(query);
  const normalizedTitle = normalizeTitleText(title);
  if (!normalizedQuery || !normalizedTitle) return 0;
  if (normalizedQuery === normalizedTitle) return 1;
  const queryTokens = normalizedQuery.split(" ");
  const titleTokens = normalizedTitle.split(" ");
  // Token-prefix: the query matches the start of the title. A prefix is a
  // strong signal (titles are written front-loaded), and the directionality
  // keeps a single common word like "learning" from matching every title.
  if (
    queryTokens.length < titleTokens.length
    && titleTokens.slice(0, queryTokens.length).join(" ") === queryTokens.join(" ")
  ) {
    return 0.95;
  }
  const querySet = new Set(queryTokens);
  const titleSet = new Set(titleTokens);
  let intersection = 0;
  for (const token of querySet) if (titleSet.has(token)) intersection += 1;
  const union = querySet.size + titleSet.size - intersection;
  if (union === 0) return 0;
  const jaccard = intersection / union;
  return jaccard >= 0.5 ? jaccard : 0;
}
