import { tokenizeUnicode, tokenizeUnicodeWithHanSingles } from "./bm25-retrieval";
import { compareTermCodePoints, lexicalTermBucket, type LexicalDictionaryEntry, type LexicalNamespace, type LexicalOccurrence, type LexicalPostingsBlock } from "./generation-index-format";

export interface DerivedLexicalChunk {
  readonly baseLength: number;
  readonly expandedLength: number;
  readonly compactText: string;
  readonly occurrences: readonly LexicalOccurrence[];
}

const NAMESPACE_ORDER: Record<LexicalNamespace, number> = { alias: 0, base: 1, expanded: 2 };
const textEncoder = new TextEncoder();

/** Canonical lexical projection shared by generation construction and store closure validation. */
export function deriveLexicalChunk(text: string, chunkOrdinal: number): DerivedLexicalChunk {
  const base = tokenizeUnicode(text);
  const expanded = tokenizeUnicodeWithHanSingles(text);
  const compactText = text.normalize("NFKC").toLocaleLowerCase("und").replace(/[^\p{L}\p{N}]+/gu, "");
  const occurrences: LexicalOccurrence[] = [];
  const appendFrequencies = (namespace: LexicalNamespace, tokens: readonly string[]) => {
    const frequencies = new Map<string, number>();
    for (const token of tokens) {
      const prior = frequencies.get(token);
      if (prior === undefined && textEncoder.encode(token).byteLength > 65_536) {
        throw new Error("evidence-derived lexical term exceeds 65536 bytes");
      }
      frequencies.set(token, (prior ?? 0) + 1);
      if (occurrences.length + frequencies.size > 65_536) {
        throw new Error("evidence-derived lexical occurrences exceed 65536 per chunk");
      }
    }
    for (const [term, tf] of frequencies) occurrences.push({ chunkOrdinal, namespace, term, tf });
  };
  appendFrequencies("base", base);
  appendFrequencies("expanded", expanded);
  const characters = Array.from(compactText);
  const grams = new Set<string>();
  for (const size of [1, 2, 3]) {
    for (let offset = 0; offset + size <= characters.length; offset += 1) {
      grams.add(characters.slice(offset, offset + size).join(""));
      if (occurrences.length + grams.size > 65_536) {
        throw new Error("evidence-derived lexical occurrences exceed 65536 per chunk");
      }
    }
  }
  for (const term of grams) occurrences.push({ chunkOrdinal, namespace: "alias", term, tf: 1 });
  if (occurrences.length > 65_536) throw new Error("evidence-derived lexical occurrences exceed 65536 per chunk");
  occurrences.sort(compareLexicalOccurrences);
  return { baseLength: base.length, expandedLength: expanded.length, compactText, occurrences };
}

/** Derive one posting block's authoritative dictionary run from its term catalog. */
export function deriveLexicalDictionaryEntries(block: LexicalPostingsBlock): LexicalDictionaryEntry[] {
  const entries: LexicalDictionaryEntry[] = [];
  let catalogIndex = 0;
  while (catalogIndex < block.termCatalog.length) {
    const first = block.occurrences[block.termCatalog[catalogIndex]!]!;
    let chunkDf = 0;
    let totalTf = 0;
    do {
      const occurrence = block.occurrences[block.termCatalog[catalogIndex]!]!;
      chunkDf += 1;
      totalTf += occurrence.tf;
      catalogIndex += 1;
    } while (catalogIndex < block.termCatalog.length
      && sameNamespaceTerm(first, block.occurrences[block.termCatalog[catalogIndex]!]!));
    entries.push({ postingOrdinal: block.postingOrdinal, namespace: first.namespace, term: first.term, chunkDf, totalTf });
  }
  return entries;
}

/**
 * Bucket per entry. The bucket is the first byte of a SHA-256 over canonicalized
 * term bytes, so it is far too expensive to recompute: callers that need it more
 * than once derive it here and pass it along.
 */
export function lexicalTermBuckets(entries: readonly LexicalDictionaryEntry[]): number[] {
  return entries.map((entry) => lexicalTermBucket(entry.namespace, entry.term));
}

export function lexicalQueryCatalog(
  entries: readonly LexicalDictionaryEntry[],
  precomputed?: readonly number[],
): number[] {
  // Computing the bucket inside the comparator would cost O(n log n) hashes.
  const buckets = precomputed ?? lexicalTermBuckets(entries);
  return entries.map((_, index) => index).sort((leftIndex, rightIndex) => {
    const left = entries[leftIndex]!;
    const right = entries[rightIndex]!;
    return buckets[leftIndex]! - buckets[rightIndex]!
      || compareNamespaceTerm(left, right)
      || left.postingOrdinal - right.postingOrdinal;
  });
}

export function lexicalBucketMask(
  entries: readonly LexicalDictionaryEntry[],
  precomputed?: readonly number[],
): string {
  const bytes = new Uint8Array(32);
  const buckets = precomputed ?? lexicalTermBuckets(entries);
  for (const bucket of buckets) {
    bytes[bucket >>> 3]! |= 1 << (bucket & 7);
  }
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function compareLexicalOccurrences(left: LexicalOccurrence, right: LexicalOccurrence): number {
  return left.chunkOrdinal - right.chunkOrdinal || compareNamespaceTerm(left, right);
}

export function compareNamespaceTerm(left: Pick<LexicalOccurrence, "namespace" | "term">, right: Pick<LexicalOccurrence, "namespace" | "term">): number {
  return NAMESPACE_ORDER[left.namespace] - NAMESPACE_ORDER[right.namespace] || compareUtf8Strings(left.term, right.term);
}

function compareUtf8Strings(left: string, right: string): number {
  // UTF-8 preserves code point order, so the comparison needs no encoding.
  return compareTermCodePoints(left, right);
}

function sameNamespaceTerm(left: Pick<LexicalOccurrence, "namespace" | "term">, right: Pick<LexicalOccurrence, "namespace" | "term">): boolean {
  return left.namespace === right.namespace && left.term === right.term;
}
