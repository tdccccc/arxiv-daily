import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test, { after, before } from "node:test";
import { loadQualityEval } from "../quality-eval/load.mjs";

const directories = [];
let qe;

before(async () => { qe = await loadQualityEval(); });
after(async () => { await Promise.all(directories.map(directory => rm(directory, { recursive: true, force: true }))); });

const temp = async prefix => {
  const directory = await mkdtemp(path.join(tmpdir(), `quality-eval-${prefix}-`));
  directories.push(directory);
  return directory;
};

const NOW = new Date("2026-10-29T12:00:00Z");
const FIELD_LABELS = ["研究问题", "方法设计", "核心结果", "研究价值", "适用边界"];
const ZH_HEADINGS = ["研究问题", "方法设计", "核心结果", "主要结论", "关键图表", "贡献与创新点", "适用边界", "学术价值判断"];
const DAILY_IDS = ["2610.00001", "2610.00002", "2610.00003", "2610.00004", "2610.00005"];
const NOTE_ID = "2607.00001";
const GENERATOR = { llm: { provider: "openai", model: "gen-model", baseUrl: "https://gen.example.invalid/v1", apiKey: "gen-key", thinkingMode: false, reasoningEffort: "medium" }, topics: [{ tag: "galaxy-cluster", name: "Galaxy Cluster" }], summaryLanguage: "zh" };
const JUDGE_TOML = '[llm]\napi_key = "judge-key-1"\nbase_url = "https://judge.example.invalid/v1"\nmodel = "judge-model"\nprovider = "openai"\n';

async function fixture() {
  const vault = await temp("vault");
  const sources = await temp("sources");
  await mkdir(path.join(vault, "arxiv-daily", "daily"), { recursive: true });
  await mkdir(path.join(vault, "arxiv-daily", "papers"), { recursive: true });
  const block = id => [`### Paper ${id}`, "- **作者**: A. Author et al.", `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`, ...FIELD_LABELS.map(label => `- **${label}**: ${id} 的历史${label}。`), ""];
  await writeFile(path.join(vault, "arxiv-daily", "daily", "2026-10-15.md"), ["---", "date: 2026-10-15", "---", "", "# t", "", "## Photo-z", "", ...DAILY_IDS.flatMap(block)].join("\n"));
  await writeFile(path.join(vault, "arxiv-daily", "papers", `${NOTE_ID}.md`), [
    "---", 'title: "Note"', 'authors: "N. Author et al."', `arxiv_id: "${NOTE_ID}"`, "primary_topic: galaxy-cluster", "published: 2026-07-10", "---", "", "# Note", "",
    ...ZH_HEADINGS.flatMap(heading => [`## ${heading}`, "", `${heading}的历史正文。`, ""]),
  ].join("\n"));
  for (const id of [...DAILY_IDS, NOTE_ID]) {
    // 2610.00002 has a revised version (v2); the others are v1.
    await writeFile(path.join(sources, `${id}.json`), JSON.stringify({
      schemaVersion: 1, id, fetchedAt: "2026-10-28T10:00:00.000Z", sourcePath: "html", arxivVersion: id === "2610.00002" ? "v2" : "v1",
      budgets: { sectionCharLimit: 16000, paperCharLimit: 100000 },
      content: { abstract: "a", abstractConclusion: `## Abstract\nAbstract of ${id}.`, fullSections: `## Methods\nWe use 120 clusters in ${id}.`, fullTextFailure: null },
      contentHash: `hash-${id}`,
    }));
  }
  return { vault, sources };
}

const goodDaily = id => JSON.stringify({ id, coreProblem: `${id} 新问题。`, keyMethod: "新方法。", mainResult: "新结果 120 个星团。", whyRelevant: "新价值。", limitations: "新边界。" });
const NOTE_MARKDOWN = "# Note\n\n- **arXiv**: https://arxiv.org/abs/2607.00001\n\n## 研究问题\n\n新的问题正文。\n\n## 核心结果\n\n新的结果：120 个星团。\n";
const idIn = messages => /ID: (\S+)/.exec(messages.find(message => message.role === "user").content)?.[1];
const regenLlm = (script) => ({
  calls: [],
  async call(messages, options) {
    this.calls.push(messages);
    const isNote = messages[0].content.includes("详细的中文论文总结");
    const scripted = script?.(this.calls.length, messages, isNote);
    options?.onMetrics?.({ logicalCalls: 1, attempts: 1, elapsedMs: 1, inputTokens: 1, outputTokens: 1, totalTokens: 2, usageComplete: true });
    if (scripted instanceof Error) throw scripted;
    return scripted ?? (isNote ? NOTE_MARKDOWN : goodDaily(idIn(messages)));
  },
});

/** A regenerate output directory made by the real command with a fake model. */
async function regenerated(state, script) {
  const outDir = path.join(await temp("regen"), "r");
  await qe.runSummaryRegenerate({ vaultRoot: state.vault, sourcesDir: state.sources, outDir, daily: 5, notes: 1, seed: "s", yes: true },
    { llm: regenLlm(script), now: () => new Date("2026-10-29T09:00:00Z"), log: () => undefined, loadGeneratorConfig: async () => GENERATOR });
  return outDir;
}
const editRecord = async (dir, kind, id, change) => {
  const file = path.join(dir, "summaries", `${kind}-${id}.json`);
  const text = JSON.stringify(change(JSON.parse(await readFile(file, "utf8"))));
  await writeFile(file, text);
  // Model a new committed generation result; tampering tests deliberately bypass this helper.
  const runFile = path.join(dir, "run.json");
  const run = JSON.parse(await readFile(runFile, "utf8"));
  if (run.resultManifest) {
    const entry = run.resultManifest.find(item => item.kind === kind && item.id === id);
    if (entry) entry.recordHash = qe.hashText(text);
    await writeFile(runFile, JSON.stringify(run));
  }
};
const editRun = async (dir, change) => {
  const file = path.join(dir, "run.json");
  await writeFile(file, JSON.stringify(change(JSON.parse(await readFile(file, "utf8")))));
};
const manifestFor = async (dir, items) => Promise.all(items.map(async ({ kind, id }) => ({
  kind, id, recordHash: qe.hashText(await readFile(path.join(dir, "summaries", `${kind}-${id}.json`), "utf8")),
})));
const withoutManifest = run => { const { resultManifest, ...legacy } = run; return legacy; };

const OK = JSON.stringify({ verdicts: { accuracy: "ok", completeness: "ok", clarity: "ok" }, issues: [] });
const judgeLlm = (reply = () => OK) => ({ calls: [], async call(messages, options) { this.calls.push(messages); options?.onMetrics?.({ logicalCalls: 1, attempts: 1, elapsedMs: 1, usageComplete: true }); return reply(this.calls.length, messages); } });

async function judgeSetup(state, summariesDir) {
  const judgeConfigPath = path.join(await temp("jc"), "judge.toml");
  await writeFile(judgeConfigPath, JUDGE_TOML);
  const outDir = path.join(await temp("out"), "judge");
  return { outDir, options: { judgeConfigPath, sourcesDir: state.sources, outDir, summariesDir } };
}
const runJudge = async (setup, more = {}, llm = judgeLlm()) => {
  const log = [];
  const result = await qe.runSummaryJudge({ ...setup.options, ...more }, { llm, now: () => NOW, log: line => log.push(line), loadGeneratorConfig: async () => { throw new Error("the CLI config must not be read"); } });
  return { result, llm, log };
};
const readJudgment = async (outDir, kind, id) => JSON.parse(await readFile(path.join(outDir, "judgments", `${kind}-${id}.json`), "utf8"));
const section = (md, heading) => { const start = md.indexOf(`\n## ${heading}`); assert.ok(start >= 0, heading); const end = md.indexOf("\n## ", start + 1); return md.slice(start, end < 0 ? undefined : end); };

test("regenerate records carry the authors the judge needs to rebuild the input", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  assert.equal((JSON.parse(await readFile(path.join(dir, "summaries", "daily-2610.00001.json"), "utf8"))).item.authors, "A. Author et al.");
});

test("items are built from the regenerated records, with the raw structured fields as locations", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  const prepared = await qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources });
  assert.deepEqual(prepared.sample, { daily: DAILY_IDS, notes: [NOTE_ID] });
  assert.equal(prepared.items.length, 6);
  const daily = prepared.items.find(item => item.id === "2610.00001");
  assert.deepEqual(daily.locations.map(location => location.key), qe.SUMMARY_FIELDS);
  assert.equal(daily.locations[0].text, "2610.00001 新问题。");
  assert.equal(daily.judgedText, qe.SUMMARY_FIELDS.map(key => `[${key}] ${daily.locations.find(location => location.key === key).text}`).join("\n"));
  assert.ok(!daily.judgedText.includes("历史"), "the historical text is not shown");

  // The input is the generator's own input: the same paper data as rebuilding from the record.
  const record = await qe.readSourceRecord(state.sources, "2610.00001");
  const rebuilt = await qe.rebuildDailyInput(record, { title: "Paper 2610.00001", authors: "A. Author et al.", summaryLanguage: "zh" });
  assert.equal(daily.paperData, rebuilt.messages[1].content);
  assert.equal(daily.messages[1].content, `${daily.paperData}\n\n<summary_under_review>\n${daily.judgedText}\n</summary_under_review>`);
  assert.equal(daily.context.sourceText, daily.paperData);

  const note = prepared.items.find(item => item.kind === "note");
  assert.deepEqual(note.locations.map(location => location.key), ["研究问题", "核心结果"]);
  assert.equal(note.locations[1].text, "新的结果：120 个星团。");
  assert.ok(note.messages[0].content.includes("「Galaxy Cluster」"));
  assert.ok(note.messages[1].content.includes("## 核心结果\n新的结果：120 个星团。"));
  assert.equal(note.version, "current");
});

test("the origin of each item names the generator, prompt hash and time", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  const prepared = await qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources });
  const item = prepared.items[0];
  assert.equal(item.origin.kind, "regenerated");
  assert.deepEqual(item.origin.generator, { provider: "openai", model: "gen-model", thinkingMode: false, reasoningEffort: "medium" });
  assert.match(item.origin.generationPromptHash, /^[0-9a-f]{64}$/);
  assert.equal(item.origin.generatedAt, "2026-10-29T09:00:00.000Z");
  assert.equal(item.origin.summariesDir, path.resolve(dir));
  assert.match(item.summaryHash, /^[0-9a-f]{64}$/);
});

test("fallback, error, source-changed and missing results are skipped with a reason", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  await editRecord(dir, "daily", "2610.00001", record => ({ ...record, status: "fallback", output: { kind: "fallback", reasonCode: "validation-exhausted", attempts: 3 } }));
  await editRecord(dir, "daily", "2610.00002", record => ({ ...record, status: "error", output: null, error: "HTTP 500" }));
  await editRecord(dir, "daily", "2610.00003", record => ({ ...record, item: { ...record.item, source: { ...record.item.source, contentHash: "stale" } } }));
  await rm(path.join(dir, "summaries", "daily-2610.00004.json"));
  const prepared = await qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources });
  const skipped = Object.fromEntries(prepared.skipped.map(entry => [entry.id, entry]));
  assert.equal(skipped["2610.00001"].reason, "fallback");
  assert.match(skipped["2610.00001"].detail, /validation-exhausted/);
  assert.equal(skipped["2610.00002"].reason, "error");
  assert.equal(skipped["2610.00003"].reason, "source-changed");
  assert.equal(skipped["2610.00004"].reason, "not-generated");
  assert.deepEqual(prepared.items.map(item => item.id).sort(), ["2607.00001", "2610.00005"]);
  assert.deepEqual(prepared.shortfall, { daily: 4, notes: 0 });
  assert.deepEqual(prepared.requested, { daily: 5, notes: 1 });
});

test("an unreadable regenerate directory is a clear error", async () => {
  const state = await fixture();
  await assert.rejects(qe.prepareRegeneratedJudgeInput({ summariesDir: await temp("empty"), sourcesDir: state.sources }), /run\.json/);
  const dir = await regenerated(state);
  await rm(path.join(dir, "run.json"));
  await assert.rejects(qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources }), /run\.json/);
});

test("an interrupted new generator run only judges results committed in its manifest", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  const generator = { provider: "openai", model: "new-generator", thinkingMode: false, reasoningEffort: "medium" };
  await editRecord(dir, "daily", DAILY_IDS[0], record => ({ ...record, generator }));
  const resultManifest = await manifestFor(dir, [{ kind: "daily", id: DAILY_IDS[0] }]);
  await editRun(dir, run => ({ ...run, generator, finishedAt: null, counts: null, resultManifest }));

  const prepared = await qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources });
  assert.deepEqual(prepared.items.map(item => item.id), [DAILY_IDS[0]]);
  assert.equal(prepared.items[0].origin.generator.model, "new-generator");
  assert.equal(prepared.generator.model, "new-generator");
  assert.equal(prepared.skipped.length, 5);
  assert.ok(prepared.skipped.every(item => item.reason === "not-generated"));
});

test("an empty generation manifest never falls back to previous output files", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  await editRun(dir, run => ({ ...run, resultManifest: [], finishedAt: null }));
  const prepared = await qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources });
  assert.equal(prepared.items.length, 0);
  assert.equal(prepared.skipped.length, 6);
  assert.ok(prepared.skipped.every(item => item.reason === "not-generated"));
});

test("a forced regeneration interrupted by errors does not judge unattempted old summaries", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  const rerun = await qe.runSummaryRegenerate({
    vaultRoot: state.vault, sourcesDir: state.sources, outDir: dir, daily: 5, notes: 1, seed: "s", yes: true, regenerate: true,
  }, {
    llm: regenLlm(call => call === 1 ? undefined : new Error("HTTP 500")), now: () => NOW, log: () => undefined,
    loadGeneratorConfig: async () => GENERATOR,
  });
  assert.equal(rerun.stopped, "errors");
  assert.equal(rerun.counts.generated, 1);
  assert.equal(rerun.counts.remaining, 2);
  const prepared = await qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources });
  assert.deepEqual(prepared.items.map(item => item.id), [DAILY_IDS[0]]);
  assert.deepEqual(prepared.skipped.filter(item => item.reason === "not-generated").map(item => item.id), [DAILY_IDS[4], NOTE_ID]);
});

test("a changed record cannot be judged under its old generation manifest hash", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  const resultManifest = await manifestFor(dir, [{ kind: "daily", id: DAILY_IDS[0] }]);
  await editRun(dir, run => ({ ...run, resultManifest }));
  const file = path.join(dir, "summaries", `daily-${DAILY_IDS[0]}.json`);
  const record = JSON.parse(await readFile(file, "utf8"));
  await writeFile(file, JSON.stringify({ ...record, output: { ...record.output, summary: { ...record.output.summary, mainResult: "改写后未提交的结果" } } }));
  await assert.rejects(qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources }), /manifest.*hash|hash.*manifest/i);
});

test("invalid generation manifests are rejected instead of treated as legacy directories", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  const entry = (await manifestFor(dir, [{ kind: "daily", id: DAILY_IDS[0] }]))[0];
  for (const resultManifest of [null, {}, [entry, entry], [{ ...entry, id: "2610.99999" }], [{ ...entry, recordHash: "bad" }]]) {
    await editRun(dir, run => ({ ...run, resultManifest }));
    await assert.rejects(qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources }), /manifest/i);
  }
});

test("legacy generation outputs remain readable but cannot claim a different generator", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  await editRun(dir, withoutManifest);
  assert.equal((await qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources })).items.length, 6);
  await editRun(dir, run => ({ ...run, generator: { ...run.generator, model: "new-generator" } }));
  const prepared = await qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources });
  assert.equal(prepared.items.length, 0);
  assert.equal(prepared.skipped.length, 6);
  assert.ok(prepared.skipped.every(item => /generator/i.test(item.detail)));
});

test("legacy generator settings compare effective mode and reject known language mismatches", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  await editRun(dir, run => ({ ...withoutManifest(run), generator: { ...run.generator, reasoningEffort: "high" } }));
  // Effort has no effect with thinking disabled.
  assert.equal((await qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources })).items.length, 6);
  await editRecord(dir, "daily", DAILY_IDS[0], record => ({ ...record, summaryLanguage: "en" }));
  let prepared = await qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources });
  assert.equal(prepared.items.length, 5);
  assert.match(prepared.skipped.find(item => item.id === DAILY_IDS[0]).detail, /language/i);
  await editRun(dir, run => ({ ...run, generator: { ...run.generator, thinkingMode: true } }));
  prepared = await qe.prepareRegeneratedJudgeInput({ summariesDir: dir, sourcesDir: state.sources });
  assert.equal(prepared.items.length, 0);
  assert.ok(prepared.skipped.every(item => /generator/i.test(item.detail)));
});

test("the judge run uses the regenerated summaries, never reads a CLI config, and records the source block", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  const setup = await judgeSetup(state, dir);
  const { result, llm } = await runJudge(setup, { yes: true });
  assert.equal(result.counts.judged, 6);
  const sent = llm.calls.find(messages => messages[1].content.includes("2610.00001 新问题"));
  assert.ok(sent && !sent[1].content.includes("历史"));
  const record = await readJudgment(setup.outDir, "daily", "2610.00001");
  assert.equal(record.origin.kind, "regenerated");
  assert.equal(record.origin.generator.model, "gen-model");
  assert.match(record.summaryHash, /^[0-9a-f]{64}$/);
  const run = JSON.parse(await readFile(path.join(setup.outDir, "run.json"), "utf8"));
  assert.equal(run.generatorModel, "gen-model");
  assert.deepEqual(run.sample, { daily: DAILY_IDS, notes: [NOTE_ID] });
  assert.equal(run.summaries.dir, path.resolve(dir));
});

test("the judge must still differ from the generator named in the regenerate run", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  const setup = await judgeSetup(state, dir);
  await writeFile(setup.options.judgeConfigPath, JUDGE_TOML.replace("judge-model", "Gen-Model"));
  await assert.rejects(runJudge(setup, { yes: true }), /different model/);
});

test("sampling options cannot be combined with --summaries", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  const setup = await judgeSetup(state, dir);
  await assert.rejects(runJudge(setup, { daily: 3 }), /--summaries/);
  assert.equal(qe.parseSummaryJudgeArgs(["--judge-config", "/j", "--sources", "/s", "--out", "/o", "--summaries", "/r"]).summariesDir, "/r");
});

test("regenerated text that changed is judged again; unchanged is reused", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  const setup = await judgeSetup(state, dir);
  await runJudge(setup, { yes: true });
  assert.equal((await runJudge(setup, { yes: true })).llm.calls.length, 0);
  await editRecord(dir, "daily", "2610.00001", record => ({ ...record, output: { kind: "structured", summary: { ...record.output.summary, mainResult: "改写后的结果。" } } }));
  const again = await runJudge(setup, { yes: true });
  assert.equal(again.llm.calls.length, 1);
  assert.ok(again.llm.calls[0][1].content.includes("改写后的结果"));
});

test("the report names where the judged summaries came from", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  const setup = await judgeSetup(state, dir);
  await runJudge(setup, { yes: true });
  const md = await readFile(path.join(setup.outDir, "report.md"), "utf8");
  const block = section(md, "评审对象");
  assert.match(block, /被评摘要来源/);
  assert.match(block, /重新生成/);
  for (const text of ["gen-model", "openai", "medium", "2026-10-29T09:00:00.000Z", path.resolve(dir)]) assert.ok(block.includes(text), text);
  assert.match(block, /生成提示词哈希[^\n]*[0-9a-f]{12}/);
  assert.ok(!block.includes("原文已有新版本"), "the multi-version flag is for historical mode only");
  assert.ok(!md.includes("仅 v1 原文的条目"));
});

test("historical mode flags papers whose cached source has a newer version and tabulates the v1-only items", async () => {
  const state = await fixture();
  const setup = await judgeSetup(state, undefined);
  const bad = JSON.stringify({ verdicts: { accuracy: "major", completeness: "ok", clarity: "ok" }, issues: [{ dimension: "accuracy", severity: "major", location: "mainResult", summaryQuote: "历史核心结果", sourceQuote: null, explanation: "原文未提及。" }] });
  const llm = judgeLlm((call, messages) => (messages[1].content.includes("2610.00002") ? bad : OK));
  await qe.runSummaryJudge({ ...setup.options, vaultRoot: state.vault, daily: 5, notes: 1, seed: "s", yes: true }, { llm, now: () => NOW, log: () => undefined, loadGeneratorConfig: async () => ({ model: "gen-model", topics: GENERATOR.topics }) });
  const record = await readJudgment(setup.outDir, "daily", "2610.00002");
  assert.equal(record.origin.kind, "historical");
  assert.equal(record.origin.date, "2026-10-15");
  assert.ok(record.origin.vault.length > 0);
  const md = await readFile(path.join(setup.outDir, "report.md"), "utf8");
  const block = section(md, "评审对象");
  assert.match(block, /历史 Vault/);
  assert.match(block, /原文已有新版本[^\n]*1 条/);
  const list = md.slice(md.indexOf("## 问题清单"));
  assert.match(list, /2610\.00002[^\n]*原文已有新版本（v2），生成时可能看到的是旧版/);
  const v1 = section(md, "判定分布");
  assert.match(v1, /仅 v1 原文的条目/);
  const json = JSON.parse(await readFile(path.join(setup.outDir, "judgments.json"), "utf8"));
  assert.equal(json.aggregate.verdicts.daily.judged, 5);
  assert.equal(json.aggregate.verdictsV1Only.daily.judged, 4);
  assert.deepEqual(json.aggregate.verdictsV1Only.daily.accuracy, { ok: 4, minor: 0, major: 0 });
  assert.deepEqual(json.aggregate.verdicts.daily.accuracy, { ok: 4, minor: 0, major: 1 });
});

test("--offline with --summaries is skipped, and the report says why", async () => {
  const state = await fixture();
  const dir = await regenerated(state);
  const setup = await judgeSetup(state, dir);
  const offline = path.join(await temp("offline"), "findings.json");
  await writeFile(offline, JSON.stringify({ grounding: { run: true, daily: { results: [] }, notes: { results: [] } } }));
  await runJudge(setup, { yes: true, offlinePath: offline });
  const md = await readFile(path.join(setup.outDir, "report.md"), "utf8");
  const cross = section(md, "与离线检查对照");
  assert.match(cross, /历史/);
  assert.match(cross, /未做对照|跳过/);
  const json = JSON.parse(await readFile(path.join(setup.outDir, "judgments.json"), "utf8"));
  assert.equal(json.crossCheck, null);
  // Report-only also skips it.
  await qe.runJudgeReportOnly({ outDir: setup.outDir, offlinePath: offline }, { now: () => NOW });
  assert.match(section(await readFile(path.join(setup.outDir, "report.md"), "utf8"), "与离线检查对照"), /历史/);
});
