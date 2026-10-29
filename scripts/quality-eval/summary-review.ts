import { existsSync } from "node:fs";
import { mkdir, readFile, readdir, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { seededRandom } from "./export";
import { requireDirectory, resolveVaultLayout } from "./summary-check";
import { JUDGE_DIMENSIONS, type JudgeDimension } from "./summary-judge-contract";
import { loadJudgments, type JudgmentRecord } from "./summary-judge-report";
import { SUMMARY_FIELDS } from "./summary-checks";
import { loadSummaryCorpus, parseBody, type SummaryCorpus } from "./summary-corpus";
import { readSourceRecord } from "./summary-sources";
import { cell } from "./report";
import { FIELD_LABELS } from "./summary-report";

const CONTEXT_CHARS = 300;
const DIMENSION_LABELS: Record<JudgeDimension, string> = { accuracy: "准确", completeness: "完整", clarity: "清楚" };
const SEVERITY_LABELS = { major: "大问题", minor: "小问题" } as const;
const DECISIONS = ["agree", "disagree", "unsure"] as const;
/** On a dispute card the researcher sides with the judge, with the reviewer, or neither. */
const DISPUTE_DECISIONS = ["judge", "reviewer", "unsure"] as const;
type Decision = (typeof DECISIONS)[number] | (typeof DISPUTE_DECISIONS)[number];
const DECISION_LABELS: Record<Decision, string> = { agree: "同意", disagree: "不同意", unsure: "拿不准", judge: "评审对", reviewer: "复核对" };
const decisionsFor = (card: Pick<ReviewCard, "meta">): readonly Decision[] => (card.meta?.group === "dispute" ? DISPUTE_DECISIONS : DECISIONS);
const REVIEWER_VERDICT_LABELS = { agree: "同意", disagree: "不同意", unsure: "拿不准" } as const;
const OPINION_LABELS = { ok: "合适", "too-high": "偏重", "too-low": "偏轻" } as const;

/** What a reader sees for a location: a daily field by its Chinese label, a note section by its own heading. */
export function locationLabel(kind: "daily" | "note", key: string): string {
  return kind === "daily" ? (FIELD_LABELS as Record<string, string>)[key] ?? key : key;
}

export type ReviewOrigin =
  | { kind: "historical"; judgeDir: string }
  | { kind: "regenerated"; judgeDir: string; generator: string; provider: string; generationPromptHash: string; summariesDir: string };

export interface ReviewCard {
  cardId: string;
  pool: "judgments" | "extra" | "meta";
  kind: "daily" | "note";
  id: string;
  title: string;
  origin: ReviewOrigin;
  location: string;
  dimension: JudgeDimension;
  severity: "major" | "minor";
  summaryQuote: string;
  /** The whole field or section the quote sits in; null when it cannot be recovered. */
  summaryText: string | null;
  sourceQuote: string | null;
  /** About 300 characters of the cached source before and after the source quote. */
  sourceContext: string | null;
  /** Why there is no source context although there is a source quote. */
  sourceContextNote: "no-sources" | "cached-source-lacks-quote" | null;
  explanation: string;
  /** Present on cards of a dispute page: what the second model said about this judge issue. */
  meta?: CardMeta;
}

export interface CardMeta {
  /** `dispute`: the reviewer disagreed or was unsure; `agreed-sample`: a spot-check of an issue it agreed with. */
  group: "dispute" | "agreed-sample";
  verdict: "agree" | "disagree" | "unsure";
  severityOpinion: "ok" | "too-high" | "too-low";
  reason: string;
  evidenceQuote: string | null;
  /** About 300 characters of the cached source around the evidence quote. */
  evidenceContext: string | null;
}

export interface BuildCardsOptions {
  /** Output directory of summary-judge.mjs; not needed with `metaDir`, which names its own. */
  judgmentsDir?: string;
  /** Output directory of summary-meta-review.mjs: the page holds the issues its reviewer disputed. */
  metaDir?: string;
  /** With `metaDir`: how many issues the reviewer agreed with to add as a spot-check (default 3). */
  agreedSample?: number;
  extraDir?: string;
  /** Cached sources, for the source context around each quote. */
  sourcesDir?: string;
  /** Only to recover the text of historical summaries the judgment did not keep. */
  vaultRoot?: string;
  configPath?: string;
  dailyDir?: string;
  papersDir?: string;
  /** Without `metaDir` required; with it, an optional cap on the whole page (disputes first). */
  count?: number;
  seed: string;
}

export interface BuiltCards { cards: ReviewCard[]; pageId: string; shortfall: number; available: { judgments: number; extra: number } }

const sha = async (text: string): Promise<string> => (await import("node:crypto")).createHash("sha256").update(text).digest("hex");

interface Candidate { record: JudgmentRecord; index: number; card: Omit<ReviewCard, "summaryText" | "sourceContext" | "sourceContextNote"> }

function candidatesOf(records: readonly JudgmentRecord[], dir: string, pool: ReviewCard["pool"]): Candidate[] {
  const tag = pool === "judgments" ? "j" : pool === "extra" ? "x" : "m";
  return records.flatMap((record) => (record.status === "judged" && record.judgment ? record.judgment.issues : []).map((issue, index) => ({
    record, index,
    card: {
      cardId: `${record.item.kind}-${record.item.id}#${index}@${tag}`, pool, kind: record.item.kind, id: record.item.id, title: record.item.title,
      origin: record.origin?.kind === "regenerated"
        ? { kind: "regenerated" as const, judgeDir: dir, generator: record.origin.generator.model, provider: record.origin.generator.provider, generationPromptHash: record.origin.generationPromptHash, summariesDir: record.origin.summariesDir }
        : { kind: "historical" as const, judgeDir: dir },
      location: issue.location, dimension: issue.dimension, severity: issue.severity, summaryQuote: issue.summaryQuote, sourceQuote: issue.sourceQuote, explanation: issue.explanation,
    },
  })));
}

/**
 * Pick `count` candidates: all majors before any minor, and within a severity the ones that add the least to what is
 * already picked (a paper not yet used, then a dimension not yet used), ties broken by a seeded key.
 */
export function pickSpread<T extends { card: { cardId: string; id: string; dimension: string; severity: string } }>(candidates: readonly T[], count: number, seed: string): T[] {
  const remaining = candidates.map((candidate) => ({ candidate, key: seededRandom(`${seed}|${candidate.card.cardId}`)() }));
  const papers = new Map<string, number>();
  const dimensions = new Map<string, number>();
  const picked: T[] = [];
  while (picked.length < count && remaining.length > 0) {
    let best = 0;
    const score = (entry: (typeof remaining)[number]): [number, number, number, number] => [
      entry.candidate.card.severity === "major" ? 0 : 1, papers.get(entry.candidate.card.id) ?? 0, dimensions.get(entry.candidate.card.dimension) ?? 0, entry.key,
    ];
    for (let at = 1; at < remaining.length; at += 1) {
      const a = score(remaining[at]!);
      const b = score(remaining[best]!);
      for (let part = 0; part < 4; part += 1) {
        if (a[part]! !== b[part]!) { if (a[part]! < b[part]!) best = at; break; }
      }
    }
    const [entry] = remaining.splice(best, 1);
    const card = entry!.candidate.card;
    picked.push(entry!.candidate);
    papers.set(card.id, (papers.get(card.id) ?? 0) + 1);
    dimensions.set(card.dimension, (dimensions.get(card.dimension) ?? 0) + 1);
  }
  return picked;
}

/** A whitespace-tolerant pattern for a quote the judge was only required to reproduce up to whitespace. */
function quotePattern(quote: string): RegExp | null {
  const words = quote.trim().split(/\s+/u).filter(Boolean);
  if (words.length === 0) return null;
  return new RegExp(words.map((word) => word.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&")).join("\\s+"), "u");
}

/** The cached source text the generator saw (abstract and conclusion, then the full-text excerpts). */
function sourceTextOf(record: { content: { abstractConclusion: string; fullSections: string | null } }): string {
  return [record.content.abstractConclusion, record.content.fullSections].filter(Boolean).join("\n\n");
}

export function contextAround(text: string, quote: string, chars = CONTEXT_CHARS): string | null {
  const match = quotePattern(quote)?.exec(text);
  if (!match) return null;
  return text.slice(Math.max(0, match.index - chars), match.index + match[0].length + chars);
}

async function regeneratedText(summariesDir: string, kind: "daily" | "note", id: string, location: string): Promise<string | null> {
  try {
    const record = JSON.parse(await readFile(path.join(summariesDir, "summaries", `${kind}-${id}.json`), "utf8")) as { output?: { kind: string; summary?: Record<string, string>; markdown?: string } };
    if (record.output?.kind === "structured") return record.output.summary?.[location] ?? null;
    if (record.output?.kind === "markdown") return parseBody(record.output.markdown ?? "").sections.find((section) => section.heading === location)?.body ?? null;
  } catch { /* the regenerated record is gone: no context */ }
  return null;
}

function vaultText(corpus: SummaryCorpus, kind: "daily" | "note", id: string, location: string, date: string | null): string | null {
  if (kind === "daily") {
    const entry = corpus.entries.find((candidate) => candidate.id === id && (!date || candidate.date === date)) ?? corpus.entries.find((candidate) => candidate.id === id);
    return entry ? entry.fields[location as (typeof SUMMARY_FIELDS)[number]] ?? null : null;
  }
  return corpus.notes.find((note) => note.id === id)?.sections.find((section) => section.heading === location)?.body ?? null;
}

interface Chosen { record: JudgmentRecord; card: Omit<ReviewCard, "summaryText" | "sourceContext" | "sourceContextNote">; meta?: Omit<CardMeta, "evidenceContext"> }

/** Fill in what a page needs around each chosen issue: the full summary text and the source context. */
async function completeCards(chosen: readonly Chosen[], options: BuildCardsOptions): Promise<ReviewCard[]> {
  let corpus: SummaryCorpus | null = null;
  const needsVault = options.vaultRoot !== undefined || options.configPath !== undefined;
  if (needsVault && chosen.some((candidate) => !candidate.record.summary && candidate.card.origin.kind === "historical")) {
    const layout = await resolveVaultLayout(options);
    corpus = await loadSummaryCorpus(layout);
  }
  const sourceCache = new Map<string, string | null>();
  const sourceOf = async (id: string): Promise<string | null> => {
    if (!options.sourcesDir) return null;
    if (!sourceCache.has(id)) {
      const cached = await readSourceRecord(options.sourcesDir, id);
      sourceCache.set(id, cached ? sourceTextOf(cached) : null);
    }
    return sourceCache.get(id) ?? null;
  };
  const cards: ReviewCard[] = [];
  for (const { record, card, meta } of chosen) {
    let summaryText: string | null = record.summary?.locations.find((location) => location.key === card.location)?.text ?? null;
    if (summaryText === null && card.origin.kind === "regenerated") summaryText = await regeneratedText(card.origin.summariesDir, card.kind, card.id, card.location);
    if (summaryText === null && corpus) summaryText = vaultText(corpus, card.kind, card.id, card.location, record.item.date);

    let sourceContext: string | null = null;
    let sourceContextNote: ReviewCard["sourceContextNote"] = null;
    if (card.sourceQuote !== null) {
      if (!options.sourcesDir) sourceContextNote = "no-sources";
      else {
        const text = await sourceOf(card.id);
        sourceContext = text ? contextAround(text, card.sourceQuote) : null;
        if (sourceContext === null) sourceContextNote = text ? "cached-source-lacks-quote" : "no-sources";
      }
    }
    const completed: ReviewCard = { ...card, summaryText, sourceContext, sourceContextNote };
    if (meta) {
      const text = meta.evidenceQuote ? await sourceOf(card.id) : null;
      completed.meta = { ...meta, evidenceContext: text && meta.evidenceQuote ? contextAround(text, meta.evidenceQuote) : null };
    }
    cards.push(completed);
  }
  return cards;
}

async function pageIdOf(cards: readonly ReviewCard[]): Promise<string> {
  return (await sha(JSON.stringify(cards.map((card) => [card.cardId, card.origin.judgeDir, card.summaryQuote, card.meta?.verdict ?? null])))).slice(0, 16);
}

interface MetaFileRecord {
  status: string;
  item: JudgmentRecord["item"] & { origin?: JudgmentRecord["origin"] };
  judge: JudgmentRecord["judge"];
  issues: JudgeIssueLike[];
  reviews: Array<{ index: number; verdict: CardMeta["verdict"]; severity: CardMeta["severityOpinion"]; reason: string; evidenceQuote: string | null }> | null;
}
type JudgeIssueLike = NonNullable<JudgmentRecord["judgment"]>["issues"][number];

async function readJsonDir<T>(dir: string): Promise<T[]> {
  if (!existsSync(dir)) return [];
  const records: T[] = [];
  for (const name of (await readdir(dir)).filter((entry) => entry.endsWith(".json")).sort()) {
    try { records.push(JSON.parse(await readFile(path.join(dir, name), "utf8")) as T); } catch { /* not a record */ }
  }
  return records;
}

/**
 * The issues a second model disputed (disagree or unsure) from a meta-review directory, plus a seeded spot-check of
 * the ones it agreed with. The judge's own records, found through the meta run's `judged.dir`, supply the summary text.
 */
async function buildMetaCards(options: BuildCardsOptions): Promise<BuiltCards> {
  const metaDir = path.resolve(options.metaDir!);
  const records = (await readJsonDir<MetaFileRecord>(path.join(metaDir, "meta"))).filter((record) => record.status === "reviewed" && record.reviews);
  if (records.length === 0) throw new Error(`no reviews found in ${path.join(metaDir, "meta")}; run summary-meta-review.mjs with --yes first`);
  let judgedDir = options.judgmentsDir ? path.resolve(options.judgmentsDir) : metaDir;
  try {
    const run = JSON.parse(await readFile(path.join(metaDir, "run.json"), "utf8")) as { judged?: { dir?: string } };
    if (!options.judgmentsDir && run.judged?.dir) judgedDir = run.judged.dir;
  } catch { /* the summary text then comes from the other sources */ }
  const judgeRecords = new Map<string, JudgmentRecord>();
  try { for (const record of (await loadJudgments(judgedDir)).records) judgeRecords.set(`${record.item.kind}-${record.item.id}`, record); } catch { /* no judge records */ }

  const disputes: Array<Candidate & { meta: Chosen["meta"] }> = [];
  const agreed: Array<Candidate & { meta: Chosen["meta"] }> = [];
  for (const record of records) {
    const original = judgeRecords.get(`${record.item.kind}-${record.item.id}`);
    const pseudo = {
      schemaVersion: 1, status: "judged", invalid: false, item: record.item, judge: record.judge, origin: record.item.origin, attempts: [], judgePromptHash: "",
      judgment: { verdicts: { accuracy: "ok", completeness: "ok", clarity: "ok" }, issues: record.issues }, ...(original?.summary ? { summary: original.summary } : {}),
    } as unknown as JudgmentRecord;
    for (const candidate of candidatesOf([pseudo], judgedDir, "meta")) {
      const review = record.reviews!.find((entry) => entry.index === candidate.index);
      if (!review) continue;
      const entry = { ...candidate, meta: { group: review.verdict === "agree" ? "agreed-sample" as const : "dispute" as const, verdict: review.verdict, severityOpinion: review.severity, reason: review.reason, evidenceQuote: review.evidenceQuote } };
      (review.verdict === "agree" ? agreed : disputes).push(entry);
    }
  }
  if (disputes.length + agreed.length === 0) throw new Error("no issues to review: the reviews contain no issues");
  const chosenDisputes = pickSpread(disputes, options.count ?? disputes.length, options.seed);
  // --count caps the whole page: the disputes come first and the spot-check only takes the room left.
  const room = options.count === undefined ? Infinity : Math.max(0, options.count - chosenDisputes.length);
  const chosenAgreed = pickSpread(agreed, Math.min(options.agreedSample ?? 3, room), options.seed);
  const cards = await completeCards([...chosenDisputes, ...chosenAgreed], options);
  return { cards, pageId: await pageIdOf(cards), shortfall: 0, available: { judgments: disputes.length, extra: agreed.length } };
}

/** Choose the issues to show a reviewer and gather what the page needs around each one. */
export async function buildReviewCards(options: BuildCardsOptions): Promise<BuiltCards> {
  if (options.metaDir) return buildMetaCards(options);
  if (!options.judgmentsDir || options.count === undefined) throw new Error("buildReviewCards needs judgmentsDir and count (or metaDir)");
  const count = options.count;
  const load = async (dir: string, pool: ReviewCard["pool"]) => {
    const resolved = path.resolve(dir);
    return candidatesOf((await loadJudgments(resolved)).records, resolved, pool);
  };
  const main = await load(options.judgmentsDir, "judgments");
  const extra = options.extraDir ? await load(options.extraDir, "extra") : [];
  if (main.length + extra.length === 0) throw new Error("no issues to review: the judgments contain no issues");
  const fromMain = pickSpread(main, count, options.seed);
  const fromExtra = pickSpread(extra, count - fromMain.length, options.seed);
  const cards = await completeCards([...fromMain, ...fromExtra], options);
  return { cards, pageId: await pageIdOf(cards), shortfall: Math.max(0, count - cards.length), available: { judgments: main.length, extra: extra.length } };
}

// ---- the page --------------------------------------------------------------------------------------

export function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/** Escaped text with the first occurrence of `quote` wrapped in <mark>. */
function highlighted(text: string, quote: string | null): string {
  const match = quote ? quotePattern(quote)?.exec(text) : null;
  if (!match) return escapeHtml(text);
  return `${escapeHtml(text.slice(0, match.index))}<mark>${escapeHtml(match[0])}</mark>${escapeHtml(text.slice(match.index + match[0].length))}`;
}

const safeJson = (value: unknown): string => JSON.stringify(value).replace(/</g, "\\u003c").replace(/>/g, "\\u003e").replace(/&/g, "\\u0026").replace(/\u2028/g, "\\u2028").replace(/\u2029/g, "\\u2029");

function originLabel(origin: ReviewOrigin): string {
  return origin.kind === "regenerated"
    ? `重新生成（生成模型 ${origin.generator}，提示词哈希 ${origin.generationPromptHash.slice(0, 12)}）· 评审目录 ${origin.judgeDir}`
    : `历史摘要（Vault 里当时生成的内容）· 评审目录 ${origin.judgeDir}`;
}

function renderMeta(meta: CardMeta): string {
  const evidence = meta.evidenceQuote === null ? "" : `<p class="label">复核引用的原文</p><blockquote class="quote">${escapeHtml(meta.evidenceQuote)}</blockquote>${meta.evidenceContext
    ? `<p class="label">原文上下文</p><div class="text">…${highlighted(meta.evidenceContext, meta.evidenceQuote)}…</div>` : ""}`;
  return `<section class="meta"><h3>复核理由</h3>
<p class="chips"><span class="chip">复核的裁定：${REVIEWER_VERDICT_LABELS[meta.verdict]}</span><span class="chip">严重程度意见：${OPINION_LABELS[meta.severityOpinion]}</span></p>
<p class="text">${escapeHtml(meta.reason)}</p>${evidence}</section>`;
}

function groupBanner(card: ReviewCard): string {
  if (!card.meta) return "";
  return card.meta.group === "dispute"
    ? `<p class="banner banner-dispute">复核分歧：第二个模型对这一条的裁定是“${REVIEWER_VERDICT_LABELS[card.meta.verdict]}”。请你裁定谁对。</p>`
    : `<p class="banner banner-agreed">复核同意的抽查：评审和复核都认为这个问题成立。请看看是否真的成立（同意 = 两个模型都对）。</p>`;
}

function renderCard(card: ReviewCard, position: number, total: number): string {
  const sourceBlock = card.sourceQuote === null
    ? `<p class="absent">原文未提及（评审认为摘要的这个说法在原文里没有依据）</p>`
    : `<blockquote class="quote">${escapeHtml(card.sourceQuote)}</blockquote>${card.sourceContext
      ? `<p class="label">原文上下文</p><div class="text">…${highlighted(card.sourceContext, card.sourceQuote)}…</div>`
      : `<p class="note">${card.sourceContextNote === "no-sources" ? "没有提供原文缓存，无法显示上下文。" : "当前缓存的原文里找不到这句话，无法显示上下文（原文可能已更新）。"}</p>`}`;
  return `<article class="card" id="card-${position}" data-card-id="${escapeHtml(card.cardId)}">
<header><span class="count">${position} / ${total}</span><h2>${escapeHtml(card.title)}</h2><p class="id">${escapeHtml(card.id)} · ${card.kind === "daily" ? "日报条目" : "详细笔记"}</p><p class="origin origin-${card.origin.kind}">被评摘要来源：${escapeHtml(originLabel(card.origin))}</p>${groupBanner(card)}</header>
<p class="chips"><span class="chip">位置：${escapeHtml(locationLabel(card.kind, card.location))}</span><span class="chip">维度：${DIMENSION_LABELS[card.dimension]}</span><span class="chip sev-${card.severity}">${SEVERITY_LABELS[card.severity]}</span></p>
<h3>摘要原句</h3>
<blockquote class="quote">${escapeHtml(card.summaryQuote)}</blockquote>
${card.summaryText === null ? `<p class="note">找不到这一段摘要的全文，只能显示上面的原句。</p>` : `<p class="label">所在的完整内容</p><div class="text">${highlighted(card.summaryText, card.summaryQuote)}</div>`}
<h3>原文依据</h3>
${sourceBlock}
<h3>评审说明</h3>
<p class="text">${escapeHtml(card.explanation)}</p>
${card.meta ? renderMeta(card.meta) : ""}
<div class="actions" role="group" aria-label="你的判断">${decisionsFor(card).map((decision) => `<button type="button" class="decide decide-${decision}" data-decision="${decision}" aria-pressed="false">${DECISION_LABELS[decision]}</button>`).join("")}</div>
<label class="notelabel">备注<textarea rows="2" placeholder="可不填" data-note></textarea></label>
</article>`;
}

const STYLE = `:root{color-scheme:light dark;--bg:#fafaf8;--fg:#1d1d1b;--muted:#68685f;--card:#fff;--line:#d9d8d0;--accent:#1f5fbf;--mark:#fff1a8;--major:#b3261e;--minor:#8a5a00;--ok:#1b6b3a}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--muted:#a2a199;--card:#1f1f1c;--line:#38372f;--accent:#7fb0ff;--mark:#5c4c00;--major:#ff8a80;--minor:#e0b04a;--ok:#6fcf97}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}h1{font-size:1.35rem;margin:.4em 0}h2{font-size:1.1rem;margin:.3em 0}h3{font-size:.95rem;margin:1.1em 0 .3em;color:var(--muted)}
.bar{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.bar .progress{flex:1 1 140px;font-weight:600}button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:8px 14px;cursor:pointer;min-height:44px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:16px 0}.count{color:var(--muted);font-size:.85rem}.id,.origin{margin:.2em 0;color:var(--muted);font-size:.88rem;overflow-wrap:anywhere}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:.8em 0 0}.chip{border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-size:.85rem}.sev-major{color:var(--major);border-color:var(--major)}.sev-minor{color:var(--minor);border-color:var(--minor)}
.quote{margin:.3em 0;padding:8px 12px;border-left:4px solid var(--accent);background:rgba(127,127,127,.08);white-space:pre-wrap;overflow-wrap:anywhere}.text{white-space:pre-wrap;overflow-wrap:anywhere}
.label{margin:.6em 0 .2em;color:var(--muted);font-size:.85rem}.note,.absent{color:var(--muted)}.absent{font-weight:600}mark{background:var(--mark);color:inherit;padding:0 2px;border-radius:3px}
.actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:1em}.decide{flex:1 1 90px}.decide[aria-pressed=true]{outline:3px solid var(--accent);font-weight:700}.decide-agree[aria-pressed=true]{background:var(--ok);color:#fff}.decide-disagree[aria-pressed=true]{background:var(--major);color:#fff}
.notelabel{display:block;margin-top:.8em;color:var(--muted);font-size:.9rem}textarea{display:block;width:100%;margin-top:4px;font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
.banner{margin:.6em 0 0;padding:6px 10px;border-radius:8px;font-size:.92rem}.banner-dispute{background:rgba(179,38,30,.12)}.banner-agreed{background:rgba(31,95,191,.12)}.meta{margin-top:1em;padding:8px 12px;border:1px dashed var(--line);border-radius:8px}.decide-judge[aria-pressed=true]{background:var(--ok);color:#fff}.decide-reviewer[aria-pressed=true]{background:var(--accent);color:#fff}
.intro{color:var(--muted)}@media (max-width:520px){main{padding:8px}.card{padding:12px}}`;

const SCRIPT = `(function(){
var PAGE_ID = __PAGE_ID__;
var KEY = "summary-review:" + PAGE_ID;
var state = {};
function load(){ try { var raw = window.localStorage.getItem(KEY); if (raw) state = JSON.parse(raw) || {}; } catch (e) { state = {}; } }
function save(){ try { window.localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* works without storage */ } }
var cards = Array.prototype.slice.call(document.querySelectorAll("[data-card-id]"));
var progress = document.getElementById("progress");
function render(){
  var answered = 0;
  cards.forEach(function(card){
    var id = card.getAttribute("data-card-id");
    var entry = state[id] || {};
    if (entry.decision) answered += 1;
    Array.prototype.forEach.call(card.querySelectorAll("[data-decision]"), function(button){
      button.setAttribute("aria-pressed", button.getAttribute("data-decision") === entry.decision ? "true" : "false");
    });
    var note = card.querySelector("[data-note]");
    if (note && note.value !== (entry.note || "")) note.value = entry.note || "";
  });
  progress.textContent = "已判断 " + answered + " / " + cards.length;
}
function results(){
  return { schemaVersion: 1, pageId: PAGE_ID, exportedAt: new Date().toISOString(), results: cards.map(function(card){
    var id = card.getAttribute("data-card-id"); var entry = state[id] || {};
    return entry.decision ? { cardId: id, decision: entry.decision, note: entry.note || "" } : null;
  }).filter(Boolean) };
}
cards.forEach(function(card){
  var id = card.getAttribute("data-card-id");
  Array.prototype.forEach.call(card.querySelectorAll("[data-decision]"), function(button){
    button.addEventListener("click", function(){ var entry = state[id] || {}; entry.decision = button.getAttribute("data-decision"); state[id] = entry; save(); render(); });
  });
  var note = card.querySelector("[data-note]");
  note.addEventListener("input", function(){ var entry = state[id] || {}; entry.note = note.value; state[id] = entry; save(); });
});
document.getElementById("export").addEventListener("click", function(){
  var blob = new Blob([JSON.stringify(results(), null, 2)], { type: "application/json" });
  var link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "review-results-" + PAGE_ID + ".json";
  document.body.appendChild(link); link.click(); document.body.removeChild(link);
});
document.getElementById("copy").addEventListener("click", function(){
  var text = JSON.stringify(results(), null, 2); var status = document.getElementById("status");
  function fallback(){ var area = document.createElement("textarea"); area.value = text; document.body.appendChild(area); area.select(); try { document.execCommand("copy"); status.textContent = "已复制"; } catch (e) { status.textContent = "复制失败，请用“导出结果”"; } document.body.removeChild(area); }
  try { if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(text).then(function(){ status.textContent = "已复制"; }, fallback); } else { fallback(); } } catch (e) { fallback(); }
});
load(); render();
})();`;

export interface RenderOptions { pageId: string; createdAt: string }

/** One self-contained HTML page (no external resources) with a card per issue; all text is escaped. */
export function renderReviewPage(cards: readonly ReviewCard[], options: RenderOptions): string {
  return `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>评审判定核对</title>
<style>${STYLE}</style>
</head>
<body>
<div class="bar"><span class="progress" id="progress">已判断 0 / ${cards.length}</span><button type="button" id="export">导出结果</button><button type="button" id="copy">复制结果</button><span id="status" role="status"></span></div>
<main>
<h1>评审判定核对</h1>
${cards.some((card) => card.meta) ? `<p class="intro">下面每一条是评审模型提出的一个问题，第二个模型复核后与它有分歧（“复核分歧”），或者同意它（“复核同意的抽查”，只抽少数）。每条都列出了评审说明和复核理由。请斟酌后裁定：分歧的条目点“评审对”“复核对”或“拿不准”；抽查的条目点“同意”（两个模型都对）、“不同意”或“拿不准”，需要时写备注。进度保存在这台设备的浏览器里，全部完成后点“导出结果”（或“复制结果”）把结果发回。页面编号 ${escapeHtml(options.pageId)}，生成于 ${escapeHtml(options.createdAt)}。</p>` : `<p class="intro">下面每一条是评审模型提出的一个问题。请对照“原文依据”和摘要，判断评审说得对不对：对就点“同意”，评审看错了或说错了点“不同意”，一时判断不了点“拿不准”，需要时写备注。进度保存在这台设备的浏览器里，全部完成后点“导出结果”（或“复制结果”）把结果发回。页面编号 ${escapeHtml(options.pageId)}，生成于 ${escapeHtml(options.createdAt)}。</p>`}
${cards.map((card, index) => renderCard(card, index + 1, cards.length)).join("\n")}
</main>
<script>${SCRIPT.replace("__PAGE_ID__", safeJson(options.pageId))}</script>
</body>
</html>
`;
}

// ---- generating and importing ----------------------------------------------------------------------

export interface SummaryReviewOptions extends BuildCardsOptions { outDir: string }

async function writeAtomic(file: string, text: string): Promise<void> {
  const temporary = `${file}.tmp`;
  await writeFile(temporary, text);
  await rename(temporary, file);
}

/** What the meta-review recorded about the reviewer's verdicts over all issues (the page holds only some of them). */
async function readMetaAggregate(metaDir: string): Promise<MetaAggregateSummary> {
  const file = path.join(path.resolve(metaDir), "meta-review.json");
  let data: { counts?: { issues: number }; aggregate?: MetaAggregateSummary["aggregate"] };
  try { data = JSON.parse(await readFile(file, "utf8")); }
  catch { throw new Error(`${file} is missing; run summary-meta-review.mjs (or its --report-only) on ${metaDir} first`); }
  if (!data.aggregate || !data.counts) throw new Error(`${file} is not a meta-review report`);
  return { issues: data.counts.issues, aggregate: data.aggregate };
}

interface MetaAggregateSummary { issues: number; aggregate: { overall: Tally; byDimension: Record<JudgeDimension, Tally>; bySeverity: Record<"major" | "minor", Tally>; byKind: Record<"daily" | "note", Tally> } }

export async function runSummaryReview(options: SummaryReviewOptions, deps: { now?: () => Date } = {}): Promise<{ files: string[]; pageId: string; cards: number; shortfall: number }> {
  if (options.sourcesDir) await requireDirectory(options.sourcesDir, "--sources");
  const built = await buildReviewCards(options);
  const metaAggregate = options.metaDir ? await readMetaAggregate(options.metaDir) : undefined;
  const createdAt = (deps.now ?? (() => new Date()))().toISOString();
  const outDir = path.resolve(options.outDir);
  await mkdir(outDir, { recursive: true });
  const files = [path.join(outDir, "review.html"), path.join(outDir, "review.json")];
  await writeAtomic(files[0]!, renderReviewPage(built.cards, { pageId: built.pageId, createdAt }));
  await writeAtomic(files[1]!, `${JSON.stringify({ schemaVersion: 1, pageId: built.pageId, createdAt, seed: options.seed, requested: options.count ?? null, available: built.available, ...(metaAggregate ? { mode: "meta", metaDir: path.resolve(options.metaDir!), metaAggregate } : {}), cards: built.cards }, null, 2)}\n`);
  return { files, pageId: built.pageId, cards: built.cards.length, shortfall: built.shortfall };
}

interface Tally { agree: number; disagree: number; unsure: number; rate: number | null }

export interface Calibration {
  schemaVersion: 1;
  generatedAt: string;
  pageId: string;
  total: number;
  answered: number;
  unanswered: number;
  overall: Tally;
  byDimension: Record<JudgeDimension, Tally>;
  bySeverity: Record<"major" | "minor", Tally>;
  byOrigin: Record<string, Tally>;
  disagreements: Array<{ cardId: string; id: string; title: string; location: string; locationLabel: string; dimension: JudgeDimension; severity: string; origin: string; summaryQuote: string; explanation: string; note: string }>;
  unsureCards: Array<{ cardId: string; id: string; location: string; locationLabel: string; note: string }>;
  /** Only for a dispute page (`--meta`). */
  meta?: MetaCalibration;
}

interface DisputeTally { judge: number; reviewer: number; unsure: number; unanswered: number; total: number }
interface SampleTally { agree: number; disagree: number; unsure: number; unanswered: number; rate: number | null; total: number }

export interface MetaCalibration {
  reviewer: MetaAggregateSummary["aggregate"];
  /** Judge issues the reviewer looked at. */
  issues: number;
  disputes: { overall: DisputeTally; byDimension: Record<JudgeDimension, DisputeTally>; bySeverity: Record<"major" | "minor", DisputeTally> };
  agreedSample: SampleTally;
  precisionEstimate: {
    /** Unsure and unanswered disputes count as the judge being wrong. */
    point: number | null;
    /** Unsure and unanswered disputes count as the judge being right. */
    upper: number | null;
    components: { reviewerAgreed: number; agreedSampleRate: number | null; judgeSidedDisputes: number; disputesTotal: number; disputesNotJudged: number; issues: number };
  };
  disagreedWithJudge: Array<{ cardId: string; id: string; title: string; group: "dispute" | "agreed-sample"; location: string; locationLabel: string; dimension: JudgeDimension; severity: string; summaryQuote: string; explanation: string; note: string }>;
}

const emptyTally = (): Tally => ({ agree: 0, disagree: 0, unsure: 0, rate: null });
function finish(tally: Tally): Tally {
  return { ...tally, rate: tally.agree + tally.disagree === 0 ? null : tally.agree / (tally.agree + tally.disagree) };
}

/** On a dispute page "judge" and "reviewer" are the researcher's view of whether the judge's issue stands. */
const asJudgeCorrect = (decision: Decision): "agree" | "disagree" | "unsure" => (decision === "judge" ? "agree" : decision === "reviewer" ? "disagree" : decision);

export function computeCalibration(cards: readonly ReviewCard[], pageId: string, answers: ReadonlyMap<string, { decision: Decision; note: string }>, generatedAt: string, metaAggregate?: MetaAggregateSummary): Calibration {
  const overall = emptyTally();
  const byDimension = Object.fromEntries(JUDGE_DIMENSIONS.map((dimension) => [dimension, emptyTally()])) as Record<JudgeDimension, Tally>;
  const bySeverity = { major: emptyTally(), minor: emptyTally() };
  const byOrigin: Record<string, Tally> = {};
  const disagreements: Calibration["disagreements"] = [];
  const unsureCards: Calibration["unsureCards"] = [];
  let answered = 0;
  for (const card of cards) {
    byOrigin[card.origin.kind] ??= emptyTally();
    const raw = answers.get(card.cardId);
    if (!raw) continue;
    const answer = { decision: asJudgeCorrect(raw.decision), note: raw.note };
    answered += 1;
    for (const tally of [overall, byDimension[card.dimension], bySeverity[card.severity], byOrigin[card.origin.kind]!]) tally[answer.decision] += 1;
    if (answer.decision === "disagree") {
      disagreements.push({ cardId: card.cardId, id: card.id, title: card.title, location: card.location, locationLabel: locationLabel(card.kind, card.location), dimension: card.dimension, severity: card.severity, origin: card.origin.kind, summaryQuote: card.summaryQuote, explanation: card.explanation, note: answer.note });
    } else if (answer.decision === "unsure") unsureCards.push({ cardId: card.cardId, id: card.id, location: card.location, locationLabel: locationLabel(card.kind, card.location), note: answer.note });
  }
  const calibration: Calibration = {
    schemaVersion: 1, generatedAt, pageId, total: cards.length, answered, unanswered: cards.length - answered,
    overall: finish(overall),
    byDimension: Object.fromEntries(JUDGE_DIMENSIONS.map((dimension) => [dimension, finish(byDimension[dimension])])) as Record<JudgeDimension, Tally>,
    bySeverity: { major: finish(bySeverity.major), minor: finish(bySeverity.minor) },
    byOrigin: Object.fromEntries(Object.entries(byOrigin).map(([kind, tally]) => [kind, finish(tally)])),
    disagreements, unsureCards,
  };
  if (metaAggregate) calibration.meta = computeMetaCalibration(cards, answers, metaAggregate, calibration.disagreements);
  return calibration;
}

const emptyDispute = (): DisputeTally => ({ judge: 0, reviewer: 0, unsure: 0, unanswered: 0, total: 0 });

function computeMetaCalibration(cards: readonly ReviewCard[], answers: ReadonlyMap<string, { decision: Decision; note: string }>, summary: MetaAggregateSummary, disagreements: Calibration["disagreements"]): MetaCalibration {
  const overall = emptyDispute();
  const byDimension = Object.fromEntries(JUDGE_DIMENSIONS.map((dimension) => [dimension, emptyDispute()])) as Record<JudgeDimension, DisputeTally>;
  const bySeverity = { major: emptyDispute(), minor: emptyDispute() };
  const sample = { agree: 0, disagree: 0, unsure: 0, unanswered: 0, total: 0 };
  for (const card of cards) {
    const answer = answers.get(card.cardId)?.decision;
    if (card.meta?.group === "agreed-sample") {
      sample.total += 1;
      if (answer === "agree" || answer === "disagree" || answer === "unsure") sample[answer] += 1; else sample.unanswered += 1;
      continue;
    }
    for (const tally of [overall, byDimension[card.dimension], bySeverity[card.severity]]) {
      tally.total += 1;
      if (answer === "judge" || answer === "reviewer" || answer === "unsure") tally[answer] += 1; else tally.unanswered += 1;
    }
  }
  const sampleRate = sample.agree + sample.disagree === 0 ? null : sample.agree / (sample.agree + sample.disagree);
  const reviewerAgreed = summary.aggregate.overall.agree;
  const disputesTotal = summary.aggregate.overall.disagree + summary.aggregate.overall.unsure;
  const disputesNotJudged = Math.max(0, disputesTotal - overall.judge - overall.reviewer);
  const issues = summary.issues;
  const estimate = (extra: number): number | null => (sampleRate === null || issues === 0 ? null : (reviewerAgreed * sampleRate + overall.judge + extra) / issues);
  return {
    reviewer: summary.aggregate, issues,
    disputes: { overall, byDimension, bySeverity },
    agreedSample: { ...sample, rate: sampleRate },
    precisionEstimate: {
      point: estimate(0), upper: estimate(disputesNotJudged),
      components: { reviewerAgreed, agreedSampleRate: sampleRate, judgeSidedDisputes: overall.judge, disputesTotal, disputesNotJudged, issues },
    },
    disagreedWithJudge: disagreements.map((entry) => {
      const card = cards.find((candidate) => candidate.cardId === entry.cardId)!;
      return { cardId: entry.cardId, id: entry.id, title: entry.title, group: card.meta!.group, location: entry.location, locationLabel: entry.locationLabel, dimension: entry.dimension, severity: entry.severity, summaryQuote: entry.summaryQuote, explanation: entry.explanation, note: entry.note };
    }),
  };
}

const rateText = (tally: Tally): string => (tally.rate === null ? "n/a" : `${(tally.rate * 100).toFixed(1)}%（${tally.agree}/${tally.agree + tally.disagree}）`);
const row = (label: string, tally: Tally): string => `| ${label} | ${tally.agree} | ${tally.disagree} | ${tally.unsure} | ${rateText(tally)} |`;
const ORIGIN_LABELS: Record<string, string> = { historical: "历史摘要", regenerated: "重新生成" };

const pct = (value: number | null): string => (value === null ? "n/a" : `${(value * 100).toFixed(1)}%`);

function renderMetaCalibration(calibration: Calibration): string {
  const m = calibration.meta!;
  const r = m.reviewer;
  const d = (label: string, t: DisputeTally): string => `| ${label} | ${t.judge} | ${t.reviewer} | ${t.unsure} | ${t.unanswered} | ${t.total} |`;
  const c = m.precisionEstimate.components;
  const lines = [
    "# 评审判定核对报告（含第二个模型的复核）",
    "",
    `核对页面 ${calibration.pageId}；页面共 ${calibration.total} 条，已判断 ${calibration.answered} 条，未判断 ${calibration.unanswered} 条。生成时间 ${calibration.generatedAt}。`,
    "",
    "第二个模型先逐条复核了评审提出的所有问题；研究者只裁定它与评审有分歧的问题（复核不同意或拿不准），并抽查了少数双方都同意的问题。",
    "",
    "## 复核一致率",
    "",
    `复核模型对评审提出的 ${m.issues} 个问题：同意 ${r.overall.agree}，不同意 ${r.overall.disagree}，拿不准 ${r.overall.unsure}；复核一致率（同意 /（同意 + 不同意））${pct(r.overall.rate)}。`,
    "",
    "| 维度 | 同意 | 不同意 | 拿不准 | 复核一致率 |",
    "| --- | --- | --- | --- | --- |",
    ...JUDGE_DIMENSIONS.map((dimension) => `| ${DIMENSION_LABELS[dimension]} | ${r.byDimension[dimension].agree} | ${r.byDimension[dimension].disagree} | ${r.byDimension[dimension].unsure} | ${pct(r.byDimension[dimension].rate)} |`),
    "",
    "## 分歧条目的裁定",
    "",
    "研究者认为谁对：评审对 = 评审提出的问题成立；复核对 = 评审的问题不成立或不该这么重。",
    "",
    "### 总体",
    "",
    "| 分组 | 评审对 | 复核对 | 拿不准 | 未回答 | 合计 |",
    "| --- | --- | --- | --- | --- | --- |",
    d("全部分歧", m.disputes.overall),
    "",
    "### 按维度",
    "",
    "| 维度 | 评审对 | 复核对 | 拿不准 | 未回答 | 合计 |",
    "| --- | --- | --- | --- | --- | --- |",
    ...JUDGE_DIMENSIONS.map((dimension) => d(DIMENSION_LABELS[dimension], m.disputes.byDimension[dimension])),
    "",
    "### 按严重程度（评审给的）",
    "",
    "| 严重程度 | 评审对 | 复核对 | 拿不准 | 未回答 | 合计 |",
    "| --- | --- | --- | --- | --- | --- |",
    d(SEVERITY_LABELS.major, m.disputes.bySeverity.major),
    d(SEVERITY_LABELS.minor, m.disputes.bySeverity.minor),
    "",
    "## 复核同意的抽查",
    "",
    m.agreedSample.total === 0 ? "没有抽查（--agreed-sample 为 0）。" : `抽查 ${m.agreedSample.total} 条双方都同意的问题：研究者同意 ${m.agreedSample.agree}，不同意 ${m.agreedSample.disagree}，拿不准 ${m.agreedSample.unsure}，未回答 ${m.agreedSample.unanswered}；同意率 ${pct(m.agreedSample.rate)}。`,
    "",
    "## 评审精确率估计",
    "",
    `估计：评审提出的问题里，经研究者认可成立的比例 ≈ ${pct(m.precisionEstimate.point)}（上界 ${pct(m.precisionEstimate.upper)}）。`,
    "",
    `- 公式：（复核同意的问题数 × 抽查同意率 + 研究者站在评审一边的分歧数）/ 全部问题数 =（${c.reviewerAgreed} × ${pct(c.agreedSampleRate)} + ${c.judgeSidedDisputes}）/ ${c.issues}。`,
    `- 分歧一共 ${c.disputesTotal} 条，其中 ${c.disputesNotJudged} 条研究者没有裁定为评审对或复核对（拿不准或未回答）。点估计把它们按“评审错”算，上界把它们按“评审对”算。`,
    "- 这只是估计：抽查样本很小，复核同意的问题里没被抽到的部分按抽查同意率外推；复核模型本身也可能出错。",
    "",
    "## 研究者不认同评审的条目",
    "",
  ];
  if (m.disagreedWithJudge.length === 0) lines.push("没有。", "");
  else lines.push("| 编号 | 来源 | 位置 | 维度 | 严重程度 | 摘要原句 | 评审说明 | 研究者备注 |", "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ...m.disagreedWithJudge.map((entry) => `| ${entry.id} | ${entry.group === "dispute" ? "复核分歧" : "复核同意的抽查"} | ${cell(entry.locationLabel)} | ${DIMENSION_LABELS[entry.dimension]} | ${SEVERITY_LABELS[entry.severity as "major" | "minor"] ?? entry.severity} | ${cell(entry.summaryQuote)} | ${cell(entry.explanation)} | ${cell(entry.note) || "-"} |`), "");
  return `${lines.join("\n")}\n`;
}

export function renderCalibration(calibration: Calibration): string {
  const header = ["| 分组 | 同意 | 不同意 | 拿不准 | 一致率 |", "| --- | --- | --- | --- | --- |"];
  const lines = [
    "# 评审判定核对报告",
    "",
    `核对页面 ${calibration.pageId}；共 ${calibration.total} 条，已判断 ${calibration.answered} 条，未判断 ${calibration.unanswered} 条。生成时间 ${calibration.generatedAt}。`,
    "",
    "一致率 = 同意 /（同意 + 不同意）。“拿不准”不计入一致率，单独列出。一致率说的是：研究者认为评审提出的问题站得住的比例。",
    "",
    "## 总体",
    "",
    ...header,
    row("全部", calibration.overall),
    "",
    "## 按维度",
    "",
    ...header,
    ...JUDGE_DIMENSIONS.map((dimension) => row(DIMENSION_LABELS[dimension], calibration.byDimension[dimension])),
    "",
    "## 按严重程度",
    "",
    ...header,
    row(SEVERITY_LABELS.major, calibration.bySeverity.major),
    row(SEVERITY_LABELS.minor, calibration.bySeverity.minor),
    "",
    "## 按来源",
    "",
    ...header,
    ...Object.entries(calibration.byOrigin).map(([kind, tally]) => row(ORIGIN_LABELS[kind] ?? kind, tally)),
    "",
    "## 不同意的判定",
    "",
  ];
  if (calibration.disagreements.length === 0) lines.push("没有。", "");
  else {
    lines.push("| 编号 | 位置 | 维度 | 严重程度 | 来源 | 摘要原句 | 评审说明 | 研究者备注 |", "| --- | --- | --- | --- | --- | --- | --- | --- |",
      ...calibration.disagreements.map((entry) => `| ${entry.id} | ${cell(entry.locationLabel)} | ${DIMENSION_LABELS[entry.dimension]} | ${SEVERITY_LABELS[entry.severity as "major" | "minor"] ?? entry.severity} | ${ORIGIN_LABELS[entry.origin] ?? entry.origin} | ${cell(entry.summaryQuote)} | ${cell(entry.explanation)} | ${cell(entry.note) || "-"} |`), "");
  }
  lines.push("## 拿不准的判定", "");
  if (calibration.unsureCards.length === 0) lines.push("没有。", "");
  else lines.push("| 编号 | 位置 | 研究者备注 |", "| --- | --- | --- |", ...calibration.unsureCards.map((entry) => `| ${entry.id} | ${cell(entry.locationLabel)} | ${cell(entry.note) || "-"} |`), "");
  if (calibration.meta) return renderMetaCalibration(calibration);
  return `${lines.join("\n")}\n`;
}

export async function runReviewImport(options: { resultsPath: string; outDir: string }, deps: { now?: () => Date } = {}): Promise<{ files: string[]; calibration: Calibration }> {
  const outDir = path.resolve(options.outDir);
  const manifestFile = path.join(outDir, "review.json");
  if (!existsSync(manifestFile)) throw new Error(`${manifestFile} not found; --out must be the directory the review page was generated into (review.json lists its cards)`);
  const manifest = JSON.parse(await readFile(manifestFile, "utf8")) as { pageId: string; cards: ReviewCard[]; mode?: string; metaAggregate?: MetaAggregateSummary };
  let parsed: { pageId?: unknown; results?: unknown };
  try { parsed = JSON.parse(await readFile(options.resultsPath, "utf8")); }
  catch { throw new Error(`${options.resultsPath} is not readable JSON`); }
  if (!Array.isArray(parsed.results)) throw new Error("the results file has no results array");
  if (parsed.pageId !== manifest.pageId) throw new Error(`页面编号不符：结果来自 ${JSON.stringify(parsed.pageId)}，而这个核对页是 ${manifest.pageId}（pageId）`);
  const byId = new Map(manifest.cards.map((card) => [card.cardId, card] as const));
  const answers = new Map<string, { decision: Decision; note: string }>();
  for (const [index, entry] of (parsed.results as Array<{ cardId?: unknown; decision?: unknown; note?: unknown }>).entries()) {
    const card = typeof entry?.cardId === "string" ? byId.get(entry.cardId) : undefined;
    if (!card) throw new Error(`results[${index}].cardId ${JSON.stringify(entry?.cardId)} does not belong to this review page`);
    const allowed = decisionsFor(card);
    if (!allowed.includes(entry.decision as Decision)) throw new Error(`results[${index}].decision ${JSON.stringify(entry.decision)} is not allowed on this card; it must be one of ${allowed.join(", ")}`);
    if (answers.has(entry.cardId)) throw new Error(`results[${index}].cardId ${entry.cardId} 重复出现 (duplicate)`);
    answers.set(entry.cardId, { decision: entry.decision as Decision, note: typeof entry.note === "string" ? entry.note : "" });
  }
  const calibration = computeCalibration(manifest.cards, manifest.pageId, answers, (deps.now ?? (() => new Date()))().toISOString(), manifest.mode === "meta" ? manifest.metaAggregate : undefined);
  const files = [path.join(outDir, "calibration.md"), path.join(outDir, "calibration.json")];
  await writeAtomic(files[0]!, renderCalibration(calibration));
  await writeAtomic(files[1]!, `${JSON.stringify(calibration, null, 2)}\n`);
  return { files, calibration };
}

export interface ReviewArgs extends Partial<SummaryReviewOptions> { importPath?: string; help?: boolean }

export function parseSummaryReviewArgs(argv: readonly string[]): ReviewArgs {
  const parsed: ReviewArgs = {};
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index]!;
    if (flag === "--help" || flag === "-h") return { help: true };
    const take = (): string => {
      const value = argv[index + 1];
      if (value === undefined || value.startsWith("--")) throw new Error(`${flag} needs a value`);
      index += 1;
      return value;
    };
    switch (flag) {
      case "--judgments": parsed.judgmentsDir = take(); break;
      case "--meta": parsed.metaDir = take(); break;
      case "--agreed-sample": {
        const value = take();
        if (!/^\d+$/.test(value)) throw new Error("--agreed-sample must be a whole number, 0 or more");
        parsed.agreedSample = Number(value);
        break;
      }
      case "--extra": parsed.extraDir = take(); break;
      case "--sources": parsed.sourcesDir = take(); break;
      case "--vault": parsed.vaultRoot = take(); break;
      case "--config": parsed.configPath = take(); break;
      case "--daily-dir": parsed.dailyDir = take(); break;
      case "--papers-dir": parsed.papersDir = take(); break;
      case "--seed": parsed.seed = take(); break;
      case "--out": parsed.outDir = take(); break;
      case "--import": parsed.importPath = take(); break;
      case "--count": {
        const value = take();
        if (!/^\d+$/.test(value) || Number(value) < 1) throw new Error("--count must be a whole number of at least 1");
        parsed.count = Number(value);
        break;
      }
      default: throw new Error(`unknown option: ${flag}`);
    }
  }
  if (!parsed.outDir) throw new Error("--out <directory> is required");
  if (parsed.importPath) {
    if (parsed.judgmentsDir || parsed.extraDir || parsed.metaDir || parsed.count !== undefined) throw new Error("--import only reads the results file and the review.json in --out; do not combine it with --judgments, --extra, --meta or --count");
  } else if (parsed.metaDir) {
    if (parsed.extraDir) throw new Error("--meta takes its issues from the meta-review directory; do not combine it with --extra");
  } else {
    if (!parsed.judgmentsDir) throw new Error("--judgments <judge output directory> is required (or --meta <meta-review directory>)");
    if (parsed.agreedSample !== undefined) throw new Error("--agreed-sample only applies with --meta");
  }
  return parsed;
}
