/**
 * Lexical token-hit scores for hybrid retrieval.
 *
 * Short keyword queries ("panstarrs", "dropout") embed into the collapsed
 * region of the embedding space and score ~0.5 against every chunk, so pure
 * vector retrieval cannot rank them. A query token that literally appears in
 * a paper's text is the strongest possible signal, so papers get a lexical
 * score: the fraction of significant query tokens found in their text.
 *
 * Robustness rules:
 * - tokens shorter than 3 characters and stop words are dropped
 * - tokens appearing in more than 40% of papers are dropped (common words
 *   like "learning" would otherwise hit everything)
 * - matching runs on a compact lowercase text (all non-alphanumerics
 *   removed), so "panstarrs" matches "Pan-STARRS survey" and hyphenation
 *   cannot hide a token
 * - the score is the hit ratio over the remaining (significant, rare)
 *   tokens, graded by frequency: `count / (count + 3)` keeps a passing
 *   single mention (0.25) well below a topic paper that returns to the term
 *   again and again (0.84 for 16 mentions), so one-token queries do not
 *   collapse into a meaningless tie
 * - a paper whose title contains every significant token gets a 0.95 score
 *   floor; repeated body use can raise it further, preserving order among
 *   several titles that all contain the same one-token query
 * - papers with zero hits get no entry and are unaffected
 */

import { normalizeTitleText } from "./title-similarity";

const STOP_WORDS = new Set([
  "the", "a", "an", "of", "for", "to", "in", "on", "with", "from", "by", "at",
  "and", "or", "but", "than", "then", "into", "over", "under", "between",
  "about", "not", "are", "was", "were", "has", "have", "had", "this", "that",
  "these", "those", "their", "its", "our", "your", "we", "you", "they", "it",
  "as", "be", "been", "being", "is", "what", "which", "who", "whom", "when",
  "where", "why", "how", "do", "does", "did", "can", "could", "should",
  "would", "will", "may", "might", "such", "more", "most", "very", "much",
  "many", "some", "any", "all", "each", "few", "both", "also", "only", "via",
  "using", "used", "use", "based", "results", "result", "new", "data", "one",
]);

/** Token ratio of a paper collection above which a token is too common to rank. */
const MAX_DOCUMENT_FREQUENCY_RATIO = 0.4;

/** Occurrences per token at which the frequency factor mostly saturates (count/(count+3)). */
const FREQUENCY_SATURATION = 3;

/** Strong title-topic signal that still leaves room for body frequency to rank ties. */
const TITLE_MATCH_SCORE_FLOOR = 0.95;

/**
 * Body-token fusion is only safe for short keyword / title queries. A
 * title+abstract blob (Find similar papers) contains many ordinary academic
 * words that would otherwise lift incidental mentions above thematic matches.
 */
const KEYWORD_QUERY_MAX_SIGNIFICANT_TOKENS = 12;
const KEYWORD_QUERY_MAX_CHARS = 160;

/** Query tokens that can discriminate papers: normalized, stop words and short tokens removed. */
export function significantQueryTokens(query: string): string[] {
  return normalizeTitleText(query)
    .split(" ")
    .filter((token) => token.length >= 3 && !STOP_WORDS.has(token));
}

/**
 * Whether a free-text query is short enough for body-token fusion. Keyword and
 * short title queries return true; title+abstract from-paper queries return false.
 */
export function isKeywordQuery(query: string): boolean {
  const trimmed = query.trim();
  if (!trimmed) return false;
  if (trimmed.length > KEYWORD_QUERY_MAX_CHARS) return false;
  return significantQueryTokens(trimmed).length <= KEYWORD_QUERY_MAX_SIGNIFICANT_TOKENS;
}

/** Lowercase text with every non-alphanumeric character removed. */
export function compactText(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

/**
 * Lexical scores per paper (paperKey → score in [0, 1]) over the given
 * significant tokens, with common tokens dropped by document frequency.
 * Scores: a 0.95 floor when the paper's title contains every token (its topic),
 * combined with the hit ratio graded by frequency (`count / (count + 3)`).
 * Papers with no hits are absent from the result.
 */
export function computeTokenHitScores(
  papers: readonly { paperKey: string; text: string; title?: string }[],
  tokens: readonly string[],
): Map<string, number> {
  if (papers.length === 0 || tokens.length === 0) return new Map();
  const compacts = papers.map((paper) => ({
    paperKey: paper.paperKey,
    compact: compactText(paper.text),
    titleCompact: paper.title ? compactText(paper.title) : "",
  }));
  const documentFrequency = new Map(tokens.map((token) => [token, 0]));
  for (const { compact } of compacts) {
    for (const token of tokens) {
      if (compact.includes(token)) documentFrequency.set(token, documentFrequency.get(token)! + 1);
    }
  }
  const rareTokens = tokens.filter(
    (token) => documentFrequency.get(token)! <= papers.length * MAX_DOCUMENT_FREQUENCY_RATIO,
  );
  if (rareTokens.length === 0) return new Map();
  const hits = new Map<string, number>();
  for (const { paperKey, compact, titleCompact } of compacts) {
    const titleMatches = titleCompact
      && rareTokens.every((token) => titleCompact.includes(token));
    let hitTokens = 0;
    let totalOccurrences = 0;
    for (const token of rareTokens) {
      const count = countOccurrences(compact, token);
      if (count > 0) {
        hitTokens += 1;
        totalOccurrences += count;
      }
    }
    if (hitTokens === 0) {
      if (titleMatches) hits.set(paperKey, TITLE_MATCH_SCORE_FLOOR);
      continue;
    }
    const ratio = hitTokens / rareTokens.length;
    const averageOccurrences = totalOccurrences / hitTokens;
    const frequencyFactor = averageOccurrences / (averageOccurrences + FREQUENCY_SATURATION);
    hits.set(
      paperKey,
      Math.max(ratio * frequencyFactor, titleMatches ? TITLE_MATCH_SCORE_FLOOR : 0),
    );
  }
  return hits;
}

/** Number of non-overlapping occurrences of `needle` in `haystack`. */
function countOccurrences(haystack: string, needle: string): number {
  let count = 0;
  let index = haystack.indexOf(needle);
  while (index !== -1) {
    count += 1;
    index = haystack.indexOf(needle, index + needle.length);
  }
  return count;
}

