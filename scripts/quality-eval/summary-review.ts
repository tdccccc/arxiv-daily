import { existsSync } from "node:fs";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
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
type Decision = (typeof DECISIONS)[number];
const DECISION_LABELS: Record<Decision, string> = { agree: "同意", disagree: "不同意", unsure: "拿不准" };

/** What a reader sees for a location: a daily field by its Chinese label, a note section by its own heading. */
export function locationLabel(kind: "daily" | "note", key: string): string {
  return kind === "daily" ? (FIELD_LABELS as Record<string, string>)[key] ?? key : key;
}

export type ReviewOrigin =
  | { kind: "historical"; judgeDir: string }
  | { kind: "regenerated"; judgeDir: string; generator: string; provider: string; generationPromptHash: string; summariesDir: string };

export interface ReviewCard {
  cardId: string;
  pool: "judgments" | "extra";
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
}

export interface BuildCardsOptions {
  judgmentsDir: string;
  extraDir?: string;
  /** Cached sources, for the source context around each quote. */
  sourcesDir?: string;
  /** Only to recover the text of historical summaries the judgment did not keep. */
  vaultRoot?: string;
  configPath?: string;
  dailyDir?: string;
  papersDir?: string;
  count: number;
  seed: string;
}

export interface BuiltCards { cards: ReviewCard[]; pageId: string; shortfall: number; available: { judgments: number; extra: number } }

const sha = async (text: string): Promise<string> => (await import("node:crypto")).createHash("sha256").update(text).digest("hex");

interface Candidate { record: JudgmentRecord; index: number; card: Omit<ReviewCard, "summaryText" | "sourceContext" | "sourceContextNote"> }

function candidatesOf(records: readonly JudgmentRecord[], dir: string, pool: ReviewCard["pool"]): Candidate[] {
  const tag = pool === "judgments" ? "j" : "x";
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

/** Choose the issues to show a reviewer and gather what the page needs around each one. */
export async function buildReviewCards(options: BuildCardsOptions): Promise<BuiltCards> {
  const load = async (dir: string, pool: ReviewCard["pool"]) => {
    const resolved = path.resolve(dir);
    return candidatesOf((await loadJudgments(resolved)).records, resolved, pool);
  };
  const main = await load(options.judgmentsDir, "judgments");
  const extra = options.extraDir ? await load(options.extraDir, "extra") : [];
  if (main.length + extra.length === 0) throw new Error("no issues to review: the judgments contain no issues");
  const fromMain = pickSpread(main, options.count, options.seed);
  const fromExtra = pickSpread(extra, options.count - fromMain.length, options.seed);
  const chosen = [...fromMain, ...fromExtra];

  let corpus: SummaryCorpus | null = null;
  const needsVault = options.vaultRoot !== undefined || options.configPath !== undefined;
  if (needsVault && chosen.some((candidate) => !candidate.record.summary && candidate.card.origin.kind === "historical")) {
    const layout = await resolveVaultLayout(options);
    corpus = await loadSummaryCorpus(layout);
  }
  const sourceCache = new Map<string, string | null>();
  const cards: ReviewCard[] = [];
  for (const { record, card } of chosen) {
    let summaryText: string | null = record.summary?.locations.find((location) => location.key === card.location)?.text ?? null;
    if (summaryText === null && card.origin.kind === "regenerated") summaryText = await regeneratedText(card.origin.summariesDir, card.kind, card.id, card.location);
    if (summaryText === null && corpus) summaryText = vaultText(corpus, card.kind, card.id, card.location, record.item.date);

    let sourceContext: string | null = null;
    let sourceContextNote: ReviewCard["sourceContextNote"] = null;
    if (card.sourceQuote !== null) {
      if (!options.sourcesDir) sourceContextNote = "no-sources";
      else {
        if (!sourceCache.has(card.id)) {
          const cached = await readSourceRecord(options.sourcesDir, card.id);
          sourceCache.set(card.id, cached ? sourceTextOf(cached) : null);
        }
        const text = sourceCache.get(card.id);
        sourceContext = text ? contextAround(text, card.sourceQuote) : null;
        if (sourceContext === null) sourceContextNote = text ? "cached-source-lacks-quote" : "no-sources";
      }
    }
    cards.push({ ...card, summaryText, sourceContext, sourceContextNote });
  }
  const pageId = (await sha(JSON.stringify(cards.map((card) => [card.cardId, card.origin.judgeDir, card.summaryQuote])))).slice(0, 16);
  return { cards, pageId, shortfall: Math.max(0, options.count - cards.length), available: { judgments: main.length, extra: extra.length } };
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

function renderCard(card: ReviewCard, position: number, total: number): string {
  const sourceBlock = card.sourceQuote === null
    ? `<p class="absent">原文未提及（评审认为摘要的这个说法在原文里没有依据）</p>`
    : `<blockquote class="quote">${escapeHtml(card.sourceQuote)}</blockquote>${card.sourceContext
      ? `<p class="label">原文上下文</p><div class="text">…${highlighted(card.sourceContext, card.sourceQuote)}…</div>`
      : `<p class="note">${card.sourceContextNote === "no-sources" ? "没有提供原文缓存，无法显示上下文。" : "当前缓存的原文里找不到这句话，无法显示上下文（原文可能已更新）。"}</p>`}`;
  return `<article class="card" id="card-${position}" data-card-id="${escapeHtml(card.cardId)}">
<header><span class="count">${position} / ${total}</span><h2>${escapeHtml(card.title)}</h2><p class="id">${escapeHtml(card.id)} · ${card.kind === "daily" ? "日报条目" : "详细笔记"}</p><p class="origin origin-${card.origin.kind}">被评摘要来源：${escapeHtml(originLabel(card.origin))}</p></header>
<p class="chips"><span class="chip">位置：${escapeHtml(locationLabel(card.kind, card.location))}</span><span class="chip">维度：${DIMENSION_LABELS[card.dimension]}</span><span class="chip sev-${card.severity}">${SEVERITY_LABELS[card.severity]}</span></p>
<h3>摘要原句</h3>
<blockquote class="quote">${escapeHtml(card.summaryQuote)}</blockquote>
${card.summaryText === null ? `<p class="note">找不到这一段摘要的全文，只能显示上面的原句。</p>` : `<p class="label">所在的完整内容</p><div class="text">${highlighted(card.summaryText, card.summaryQuote)}</div>`}
<h3>原文依据</h3>
${sourceBlock}
<h3>评审说明</h3>
<p class="text">${escapeHtml(card.explanation)}</p>
<div class="actions" role="group" aria-label="你的判断">${DECISIONS.map((decision) => `<button type="button" class="decide decide-${decision}" data-decision="${decision}" aria-pressed="false">${DECISION_LABELS[decision]}</button>`).join("")}</div>
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
<p class="intro">下面每一条是评审模型提出的一个问题。请对照“原文依据”和摘要，判断评审说得对不对：对就点“同意”，评审看错了或说错了点“不同意”，一时判断不了点“拿不准”，需要时写备注。进度保存在这台设备的浏览器里，全部完成后点“导出结果”（或“复制结果”）把结果发回。页面编号 ${escapeHtml(options.pageId)}，生成于 ${escapeHtml(options.createdAt)}。</p>
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

export async function runSummaryReview(options: SummaryReviewOptions, deps: { now?: () => Date } = {}): Promise<{ files: string[]; pageId: string; cards: number; shortfall: number }> {
  if (options.sourcesDir) await requireDirectory(options.sourcesDir, "--sources");
  const built = await buildReviewCards(options);
  const createdAt = (deps.now ?? (() => new Date()))().toISOString();
  const outDir = path.resolve(options.outDir);
  await mkdir(outDir, { recursive: true });
  const files = [path.join(outDir, "review.html"), path.join(outDir, "review.json")];
  await writeAtomic(files[0]!, renderReviewPage(built.cards, { pageId: built.pageId, createdAt }));
  await writeAtomic(files[1]!, `${JSON.stringify({ schemaVersion: 1, pageId: built.pageId, createdAt, seed: options.seed, requested: options.count, available: built.available, cards: built.cards }, null, 2)}\n`);
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
}

const emptyTally = (): Tally => ({ agree: 0, disagree: 0, unsure: 0, rate: null });
function finish(tally: Tally): Tally {
  return { ...tally, rate: tally.agree + tally.disagree === 0 ? null : tally.agree / (tally.agree + tally.disagree) };
}

export function computeCalibration(cards: readonly ReviewCard[], pageId: string, answers: ReadonlyMap<string, { decision: Decision; note: string }>, generatedAt: string): Calibration {
  const overall = emptyTally();
  const byDimension = Object.fromEntries(JUDGE_DIMENSIONS.map((dimension) => [dimension, emptyTally()])) as Record<JudgeDimension, Tally>;
  const bySeverity = { major: emptyTally(), minor: emptyTally() };
  const byOrigin: Record<string, Tally> = {};
  const disagreements: Calibration["disagreements"] = [];
  const unsureCards: Calibration["unsureCards"] = [];
  let answered = 0;
  for (const card of cards) {
    byOrigin[card.origin.kind] ??= emptyTally();
    const answer = answers.get(card.cardId);
    if (!answer) continue;
    answered += 1;
    for (const tally of [overall, byDimension[card.dimension], bySeverity[card.severity], byOrigin[card.origin.kind]!]) tally[answer.decision] += 1;
    if (answer.decision === "disagree") {
      disagreements.push({ cardId: card.cardId, id: card.id, title: card.title, location: card.location, locationLabel: locationLabel(card.kind, card.location), dimension: card.dimension, severity: card.severity, origin: card.origin.kind, summaryQuote: card.summaryQuote, explanation: card.explanation, note: answer.note });
    } else if (answer.decision === "unsure") unsureCards.push({ cardId: card.cardId, id: card.id, location: card.location, locationLabel: locationLabel(card.kind, card.location), note: answer.note });
  }
  return {
    schemaVersion: 1, generatedAt, pageId, total: cards.length, answered, unanswered: cards.length - answered,
    overall: finish(overall),
    byDimension: Object.fromEntries(JUDGE_DIMENSIONS.map((dimension) => [dimension, finish(byDimension[dimension])])) as Record<JudgeDimension, Tally>,
    bySeverity: { major: finish(bySeverity.major), minor: finish(bySeverity.minor) },
    byOrigin: Object.fromEntries(Object.entries(byOrigin).map(([kind, tally]) => [kind, finish(tally)])),
    disagreements, unsureCards,
  };
}

const rateText = (tally: Tally): string => (tally.rate === null ? "n/a" : `${(tally.rate * 100).toFixed(1)}%（${tally.agree}/${tally.agree + tally.disagree}）`);
const row = (label: string, tally: Tally): string => `| ${label} | ${tally.agree} | ${tally.disagree} | ${tally.unsure} | ${rateText(tally)} |`;
const ORIGIN_LABELS: Record<string, string> = { historical: "历史摘要", regenerated: "重新生成" };

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
  return `${lines.join("\n")}\n`;
}

export async function runReviewImport(options: { resultsPath: string; outDir: string }, deps: { now?: () => Date } = {}): Promise<{ files: string[]; calibration: Calibration }> {
  const outDir = path.resolve(options.outDir);
  const manifestFile = path.join(outDir, "review.json");
  if (!existsSync(manifestFile)) throw new Error(`${manifestFile} not found; --out must be the directory the review page was generated into (review.json lists its cards)`);
  const manifest = JSON.parse(await readFile(manifestFile, "utf8")) as { pageId: string; cards: ReviewCard[] };
  let parsed: { pageId?: unknown; results?: unknown };
  try { parsed = JSON.parse(await readFile(options.resultsPath, "utf8")); }
  catch { throw new Error(`${options.resultsPath} is not readable JSON`); }
  if (!Array.isArray(parsed.results)) throw new Error("the results file has no results array");
  if (parsed.pageId !== manifest.pageId) throw new Error(`页面编号不符：结果来自 ${JSON.stringify(parsed.pageId)}，而这个核对页是 ${manifest.pageId}（pageId）`);
  const ids = new Set(manifest.cards.map((card) => card.cardId));
  const answers = new Map<string, { decision: Decision; note: string }>();
  for (const [index, entry] of (parsed.results as Array<{ cardId?: unknown; decision?: unknown; note?: unknown }>).entries()) {
    if (typeof entry?.cardId !== "string" || !ids.has(entry.cardId)) throw new Error(`results[${index}].cardId ${JSON.stringify(entry?.cardId)} does not belong to this review page`);
    if (!DECISIONS.includes(entry.decision as Decision)) throw new Error(`results[${index}].decision ${JSON.stringify(entry.decision)} must be one of ${DECISIONS.join(", ")}`);
    if (answers.has(entry.cardId)) throw new Error(`results[${index}].cardId ${entry.cardId} 重复出现 (duplicate)`);
    answers.set(entry.cardId, { decision: entry.decision as Decision, note: typeof entry.note === "string" ? entry.note : "" });
  }
  const calibration = computeCalibration(manifest.cards, manifest.pageId, answers, (deps.now ?? (() => new Date()))().toISOString());
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
    if (parsed.judgmentsDir || parsed.extraDir || parsed.count !== undefined) throw new Error("--import only reads the results file and the review.json in --out; do not combine it with --judgments, --extra or --count");
  } else if (!parsed.judgmentsDir) throw new Error("--judgments <judge output directory> is required");
  return parsed;
}
