import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
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
const repoFile = relative => path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..", relative);

const SENTINEL_KEY = "SENTINEL-GEN-KEY-51d2";
const SENTINEL_HOST = "sentinel-generator-host.example.invalid";
const NOW = new Date("2026-10-29T10:00:00Z");
const FIELD_LABELS = ["研究问题", "方法设计", "核心结果", "研究价值", "适用边界"];
const ZH_HEADINGS = ["研究问题", "方法设计", "核心结果", "主要结论", "关键图表", "贡献与创新点", "适用边界", "学术价值判断"];
const DAILY_IDS = ["2610.00001", "2610.00002", "2610.00003"];
const NOTE_ID = "2607.00001";

async function fixture() {
  const vault = await temp("vault");
  const sources = await temp("sources");
  await mkdir(path.join(vault, "arxiv-daily", "daily"), { recursive: true });
  await mkdir(path.join(vault, "arxiv-daily", "papers"), { recursive: true });
  const block = id => [`### Paper ${id}`, "- **作者**: A. Author et al.", `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`, ...FIELD_LABELS.map(label => `- **${label}**: ${id} 的${label}。`), ""];
  await writeFile(path.join(vault, "arxiv-daily", "daily", "2026-10-15.md"), ["---", "date: 2026-10-15", "---", "", "# t", "", "## Photo-z", "", ...DAILY_IDS.flatMap(block)].join("\n"));
  await writeFile(path.join(vault, "arxiv-daily", "papers", `${NOTE_ID}.md`), [
    "---", 'title: "Note"', 'authors: "N. Author et al."', `arxiv_id: "${NOTE_ID}"`, "primary_topic: galaxy-cluster", "published: 2026-07-10", "---", "", "# Note", "",
    ...ZH_HEADINGS.flatMap(heading => [`## ${heading}`, "", `${heading}的正文。`, ""]),
  ].join("\n"));
  for (const id of [...DAILY_IDS, NOTE_ID]) {
    await writeFile(path.join(sources, `${id}.json`), JSON.stringify({
      schemaVersion: 1, id, fetchedAt: "2026-10-28T10:00:00.000Z", sourcePath: "html", arxivVersion: "v1",
      budgets: { sectionCharLimit: 16000, paperCharLimit: 100000 },
      content: { abstract: "a", abstractConclusion: `## Abstract\nAbstract of ${id}.`, fullSections: `## Methods\nWe use 120 clusters in ${id}.`, fullTextFailure: null },
      contentHash: `hash-${id}`,
    }));
  }
  return { vault, sources };
}

const goodDaily = id => JSON.stringify({ id, coreProblem: "问题。", keyMethod: "方法。", mainResult: "结果 $0.85$。", whyRelevant: "价值。", limitations: "边界。" });
const NOTE_MARKDOWN = "# Note\n\n## 研究问题\n\n问题。\n";
const idIn = messages => /ID: (\S+)/.exec(messages.find(message => message.role === "user").content)?.[1];

/** Replies by request kind: a detail request gets markdown, a daily request JSON for its own id. */
function fakeLlm(script) {
  const calls = [];
  const llm = {
    calls,
    call: async (messages, options) => {
      calls.push({ messages, options });
      const isNote = messages[0].content.includes("详细的中文论文总结");
      const scripted = script?.(calls.length, messages, isNote);
      const reply = scripted ?? (isNote ? NOTE_MARKDOWN : goodDaily(idIn(messages)));
      if (reply instanceof Error) throw reply;
      options?.onMetrics?.({ logicalCalls: 1, attempts: 1, elapsedMs: 5, inputTokens: 100, outputTokens: 10, totalTokens: 110, usageComplete: true });
      return reply;
    },
  };
  return llm;
}
const GENERATOR = { llm: { provider: "openai", model: "gen-model", baseUrl: `https://${SENTINEL_HOST}/v1`, apiKey: SENTINEL_KEY, thinkingMode: false, reasoningEffort: "medium" }, topics: [{ tag: "galaxy-cluster", name: "Galaxy Cluster" }], summaryLanguage: "zh" };

async function setup(extra = {}) {
  const { vault, sources } = await fixture();
  const outDir = path.join(await temp("out"), "regen");
  return { outDir, vault, sources, options: { vaultRoot: vault, sourcesDir: sources, outDir, daily: 3, notes: 1, seed: "seed-1", ...extra } };
}
const execute = async (state, { llm = fakeLlm(), generator = GENERATOR } = {}, more = {}) => {
  const log = [];
  const result = await qe.runSummaryRegenerate({ ...state.options, ...more }, { llm, now: () => NOW, log: line => log.push(line), loadGeneratorConfig: async () => generator });
  return { result, llm, log };
};
const read = async (outDir, kind, id) => JSON.parse(await readFile(path.join(outDir, "summaries", `${kind}-${id}.json`), "utf8"));
const listFiles = async directory => {
  const found = [];
  const walk = async current => { for (const entry of await readdir(current, { withFileTypes: true })) { const full = path.join(current, entry.name); if (entry.isDirectory()) await walk(full); else found.push(full); } };
  if (existsSync(directory)) await walk(directory);
  return found;
};

test("without --yes the command prints the plan and creates and calls nothing", async () => {
  const state = await setup();
  const { result, llm, log } = await execute(state);
  assert.equal(result.dryRun, true);
  assert.equal(llm.calls.length, 0);
  assert.equal(existsSync(state.outDir), false);
  const text = log.join("\n");
  assert.match(text, /gen-model/);
  assert.match(text, /daily 3.*note 1/);
  assert.match(text, /at most 10 requests/);
  assert.match(text, /3 attempts/);
  assert.match(text, /characters/);
  assert.ok(!text.includes(SENTINEL_KEY) && !text.includes(SENTINEL_HOST));
  assert.deepEqual({ daily: result.plan.items.daily, note: result.plan.items.note, maxRequests: result.plan.maxRequests }, { daily: 3, note: 1, maxRequests: 10 });
});

test("a daily entry is generated from the cached input and recorded with its identities", async () => {
  const state = await setup();
  const { result, llm } = await execute(state, {}, { yes: true });
  assert.deepEqual(result.counts, { generated: 4, fallback: 0, errors: 0, reused: 0, remaining: 0, skipped: 0 });
  const record = await read(state.outDir, "daily", "2610.00001");
  assert.equal(record.schemaVersion, 1);
  assert.equal(record.status, "generated");
  assert.deepEqual(record.item, {
    kind: "daily", id: "2610.00001", title: "Paper 2610.00001", authors: "A. Author et al.", date: "2026-10-15", topic: "Photo-z",
    source: { sourcePath: "html", arxivVersion: "v1", fetchedAt: "2026-10-28T10:00:00.000Z", contentHash: "hash-2610.00001" },
  });
  assert.deepEqual(record.generator, { provider: "openai", model: "gen-model", thinkingMode: false, reasoningEffort: "medium" });
  assert.match(record.generationPromptHash, /^[0-9a-f]{64}$/);
  assert.deepEqual(record.output, { kind: "structured", summary: JSON.parse(goodDaily("2610.00001")) });
  assert.equal(record.attempts, 1);
  assert.equal(record.usage.inputTokens, 100);
  assert.equal(record.usage.calls, 1);
  assert.equal(record.startedAt, NOW.toISOString());
  assert.equal(llm.calls.length, 4);
});

test("the request the model receives is exactly the rebuilt input for the same record", async () => {
  const state = await setup();
  const { llm } = await execute(state, {}, { yes: true });
  const corpus = await qe.loadSummaryCorpus({ vaultRoot: state.vault });
  for (const id of DAILY_IDS) {
    const entry = corpus.entries.find(candidate => candidate.id === id);
    const record = await qe.readSourceRecord(state.sources, id);
    const rebuilt = await qe.rebuildDailyInput(record, { title: entry.title, authors: entry.authors, summaryLanguage: "zh" });
    const sent = llm.calls.find(call => idIn(call.messages) === id);
    assert.deepEqual(sent.messages, rebuilt.messages);
    assert.equal(sent.options.temperature, 0);
    assert.equal((await read(state.outDir, "daily", id)).generationPromptHash, qe.sha256Hex(rebuilt.messages[0].content));
  }
  const note = corpus.notes[0];
  const rebuilt = await qe.rebuildDetailInput(await qe.readSourceRecord(state.sources, NOTE_ID), { title: note.title, authors: note.authors, topic: note.topic, topics: GENERATOR.topics, summaryLanguage: "zh" });
  assert.deepEqual(llm.calls.find(call => call.messages[0].content.includes("详细的中文论文总结")).messages, rebuilt.messages);
});

test("invalid JSON is retried by the pipeline's own wrapper, with its correction", async () => {
  const state = await setup({ daily: 1, notes: 0, ids: ["2610.00001"] });
  const llm = fakeLlm((call, messages) => (call === 1 ? "不是 JSON" : goodDaily(idIn(messages))));
  const { result } = await execute(state, { llm }, { yes: true });
  assert.equal(llm.calls.length, 2);
  assert.ok(llm.calls[1].messages[1].content.length > llm.calls[0].messages[1].content.length, "the second request carries correction guidance");
  const record = await read(state.outDir, "daily", "2610.00001");
  assert.equal(record.status, "generated");
  assert.equal(record.attempts, 2);
  assert.equal(record.usage.calls, 2);
  assert.equal(record.validationFailures.length, 1);
  assert.equal(result.counts.generated, 1);
});

test("exhausted validation attempts are recorded as the fallback the pipeline would write", async () => {
  const state = await setup({ daily: 1, notes: 0, ids: ["2610.00001"] });
  const llm = fakeLlm(() => "still not json");
  const { result } = await execute(state, { llm }, { yes: true });
  assert.equal(llm.calls.length, 3);
  const record = await read(state.outDir, "daily", "2610.00001");
  assert.equal(record.status, "fallback");
  assert.deepEqual(record.output, { kind: "fallback", reasonCode: "validation-exhausted", attempts: 3 });
  assert.equal(record.validationFailures.length, 3);
  assert.deepEqual([result.counts.fallback, result.counts.generated, result.counts.errors], [1, 0, 0]);
});

test("a note is generated as Markdown with the configured topic name, or the tag when there is none", async () => {
  const state = await setup({ daily: 0, notes: 1 });
  const { llm } = await execute(state, {}, { yes: true });
  const record = await read(state.outDir, "note", NOTE_ID);
  assert.equal(record.status, "generated");
  assert.deepEqual(record.output, { kind: "markdown", markdown: NOTE_MARKDOWN });
  assert.deepEqual(record.topicName, { name: "Galaxy Cluster", fromConfig: true });
  assert.equal(record.item.version, "current");
  assert.ok(llm.calls[0].messages[0].content.includes("「Galaxy Cluster」"));
  assert.equal(llm.calls[0].options.temperature, undefined);

  const bare = await setup({ daily: 0, notes: 1 });
  await execute(bare, { generator: { ...GENERATOR, topics: [] } }, { yes: true });
  const fallbackName = await read(bare.outDir, "note", NOTE_ID);
  assert.deepEqual(fallbackName.topicName, { name: "galaxy-cluster", fromConfig: false });
});

test("a rerun skips what is generated, and --regenerate does it again", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const second = await execute(state, {}, { yes: true });
  assert.equal(second.llm.calls.length, 0);
  assert.equal(second.result.counts.reused, 4);
  const third = await execute(state, {}, { yes: true, regenerate: true });
  assert.equal(third.llm.calls.length, 4);
});

test("a changed generator model, prompt or source text is generated again", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const file = path.join(state.outDir, "summaries", "daily-2610.00001.json");
  const record = JSON.parse(await readFile(file, "utf8"));
  await writeFile(file, JSON.stringify({ ...record, generationPromptHash: "0".repeat(64) }));
  assert.equal((await execute(state, {}, { yes: true })).llm.calls.length, 1);
  await writeFile(file, JSON.stringify({ ...record, item: { ...record.item, source: { ...record.item.source, contentHash: "other" } } }));
  assert.equal((await execute(state, {}, { yes: true })).llm.calls.length, 1);
  const other = { ...GENERATOR, llm: { ...GENERATOR.llm, model: "gen-model-2" } };
  assert.equal((await execute(state, { generator: other }, { yes: true })).llm.calls.length, 4);
});

test("a fallback result also counts as done on the next run", async () => {
  const state = await setup({ daily: 1, notes: 0, ids: ["2610.00001"] });
  await execute(state, { llm: fakeLlm(() => "bad") }, { yes: true });
  const again = await execute(state, {}, { yes: true });
  assert.equal(again.llm.calls.length, 0);
  assert.equal(again.result.counts.reused, 1);
});

test("generation reruns when thinking mode, effective effort or endpoint changes", async () => {
  for (const [before, after] of [
    [{ thinkingMode: false }, { thinkingMode: true }],
    [{ thinkingMode: true, reasoningEffort: "medium" }, { thinkingMode: true, reasoningEffort: "high" }],
    [{}, { baseUrl: "https://another-generator.example.invalid/v1" }],
  ]) {
    const state = await setup();
    const generator = llm => ({ ...GENERATOR, llm: { ...GENERATOR.llm, ...llm } });
    await execute(state, { generator: generator(before) }, { yes: true });
    const changed = await execute(state, { generator: generator(after) }, { yes: true });
    assert.equal(changed.llm.calls.length, 4, JSON.stringify({ before, after }));
    assert.equal(changed.result.counts.reused, 0);
    assert.equal((await execute(state, { generator: generator(after) }, { yes: true })).llm.calls.length, 0);
  }
});

test("generation reuses results after a key rotation or an inactive effort change", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const generator = { ...GENERATOR, llm: { ...GENERATOR.llm, apiKey: "rotated-secret", reasoningEffort: "high" } };
  const again = await execute(state, { generator }, { yes: true });
  assert.equal(again.llm.calls.length, 0);
  assert.equal(again.result.counts.reused, 4);
});

test("generation ignores GLM effort that is not sent to the provider", async () => {
  const state = await setup();
  const generator = { ...GENERATOR, llm: { ...GENERATOR.llm, provider: "zhipu", thinkingMode: true } };
  await execute(state, { generator }, { yes: true });
  const changed = { ...generator, llm: { ...generator.llm, reasoningEffort: "high" } };
  assert.equal((await execute(state, { generator: changed }, { yes: true })).llm.calls.length, 0);
});

test("duplicate daily papers are rejected before generating into the same result file", async () => {
  const state = await setup({ daily: 10 });
  const directory = path.join(state.vault, "arxiv-daily", "daily");
  await writeFile(path.join(directory, "2026-10-16.md"), (await readFile(path.join(directory, "2026-10-15.md"), "utf8")).replaceAll("2026-10-15", "2026-10-16"));
  const llm = fakeLlm();
  await assert.rejects(execute(state, { llm }, { yes: true }), /multiple summaries.*same paper/);
  assert.equal(llm.calls.length, 0);
  assert.equal(existsSync(state.outDir), false);
});

test("generation includes the full request in its cache identity", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const daily = path.join(state.vault, "arxiv-daily", "daily", "2026-10-15.md");
  await writeFile(daily, (await readFile(daily, "utf8")).replace("### Paper 2610.00001", "### Corrected paper title"));
  const changed = await execute(state, {}, { yes: true });
  assert.equal(changed.llm.calls.length, 1);
  assert.match(changed.llm.calls[0].messages[1].content, /Corrected paper title/);
});

test("a legacy generation record without request identity is planned again before paid calls", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const file = path.join(state.outDir, "summaries", "daily-2610.00001.json");
  const record = JSON.parse(await readFile(file, "utf8"));
  delete record.cacheIdentity;
  await writeFile(file, JSON.stringify(record));
  const preview = await execute(state);
  assert.equal(preview.result.plan.toGenerate, 1);
  assert.equal(preview.llm.calls.length, 0);
  assert.equal((await execute(state, {}, { yes: true })).llm.calls.length, 1);
});

test("an errored item is retried next run, and three errors in a row stop the run", async () => {
  const state = await setup();
  const llm = fakeLlm(() => new Error(`HTTP 500 from https://${SENTINEL_HOST}/v1 with ${SENTINEL_KEY}`));
  const first = await execute(state, { llm }, { yes: true });
  assert.equal(first.result.stopped, "errors");
  assert.equal(first.result.counts.errors, 3);
  assert.equal(first.result.counts.remaining, 1);
  const errored = await read(state.outDir, "daily", "2610.00001");
  assert.equal(errored.status, "error");
  assert.ok(errored.error.length > 0);
  const second = await execute(state, {}, { yes: true });
  assert.equal(second.result.counts.generated, 4);
});

test("an unreachable endpoint (the pipeline's transport-exhausted fallback) is an error to retry, not a result", async () => {
  const state = await setup({ daily: 1, notes: 0, ids: ["2610.00001"] });
  const llm = fakeLlm(() => new qe.LlmTransientExhaustedError(new Error(`connect failed to https://${SENTINEL_HOST}/v1`)));
  const { result } = await execute(state, { llm }, { yes: true });
  const record = await read(state.outDir, "daily", "2610.00001");
  assert.equal(record.status, "error");
  assert.match(record.error, /transport-exhausted/);
  assert.ok(!JSON.stringify(record).includes(SENTINEL_HOST));
  assert.deepEqual([result.counts.errors, result.counts.fallback], [1, 0]);
  assert.equal((await execute(state, {}, { yes: true })).result.counts.generated, 1);
});

test("a success between errors resets the count", async () => {
  const state = await setup();
  const llm = fakeLlm((call, messages, isNote) => (call === 1 || call === 2 || call === 4 ? new Error("HTTP 503") : isNote ? NOTE_MARKDOWN : goodDaily(idIn(messages))));
  const { result } = await execute(state, { llm }, { yes: true });
  assert.equal(llm.calls.length, 4);
  assert.equal(result.stopped, null);
  assert.deepEqual([result.counts.errors, result.counts.generated], [3, 1]);
});

test("the generator's key and address never reach a written file or the output", async () => {
  const state = await setup();
  const llm = fakeLlm(call => (call === 2 ? new Error(`failed at https://${SENTINEL_HOST}/v1 key ${SENTINEL_KEY}`) : undefined));
  const { log } = await execute(state, { llm }, { yes: true });
  const files = await listFiles(state.outDir);
  assert.ok(files.length >= 5, files.join(","));
  for (const file of files) {
    const text = await readFile(file, "utf8");
    assert.ok(!text.includes(SENTINEL_KEY) && !text.includes(SENTINEL_HOST), file);
  }
  assert.ok(!log.join("\n").includes(SENTINEL_KEY) && !log.join("\n").includes(SENTINEL_HOST));
});

test("run.json records the arguments without secrets, the seed and the sample", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const run = JSON.parse(await readFile(path.join(state.outDir, "run.json"), "utf8"));
  assert.equal(run.schemaVersion, 1);
  assert.equal(run.seed, "seed-1");
  assert.deepEqual(run.sample, { daily: DAILY_IDS, notes: [NOTE_ID] });
  assert.deepEqual(run.generator, { provider: "openai", model: "gen-model", thinkingMode: false, reasoningEffort: "medium" });
  assert.equal(run.counts.generated, 4);
  assert.ok(!JSON.stringify(run).includes(SENTINEL_KEY));
});

test("items without a usable source are skipped and listed", async () => {
  const state = await setup();
  await rm(path.join(state.sources, "2610.00002.json"));
  const { result } = await execute(state, {}, { yes: true });
  assert.equal(result.counts.skipped, 1);
  assert.equal(result.counts.generated, 3);
  const run = JSON.parse(await readFile(path.join(state.outDir, "run.json"), "utf8"));
  assert.deepEqual(run.skipped.map(entry => [entry.id, entry.reason]), [["2610.00002", "no-source"]]);
});

test("command-line arguments are validated", () => {
  const parsed = qe.parseSummaryRegenerateArgs(["--config", "/c.toml", "--vault", "/v", "--daily-dir", "d", "--papers-dir", "p", "--sources", "/s", "--out", "/o", "--daily", "7", "--notes", "0", "--seed", "x", "--ids", "2610.00001,2607.00001", "--from", "2026-10-01", "--to", "2026-10-31", "--include-older-notes", "--yes", "--regenerate"]);
  assert.deepEqual(parsed, { configPath: "/c.toml", vaultRoot: "/v", dailyDir: "d", papersDir: "p", sourcesDir: "/s", outDir: "/o", daily: 7, notes: 0, seed: "x", ids: ["2610.00001", "2607.00001"], from: "2026-10-01", to: "2026-10-31", includeOlderNotes: true, yes: true, regenerate: true });
  assert.equal(qe.parseSummaryRegenerateArgs(["--help"]).help, true);
  assert.throws(() => qe.parseSummaryRegenerateArgs(["--out", "/o"]), /--sources/);
  assert.throws(() => qe.parseSummaryRegenerateArgs(["--sources", "/s"]), /--out/);
  assert.throws(() => qe.parseSummaryRegenerateArgs(["--sources", "/s", "--out", "/o", "--daily", "-1"]), /--daily/);
  assert.throws(() => qe.parseSummaryRegenerateArgs(["--sources", "/s", "--out", "/o", "--bogus"]), /unknown option.*--bogus/i);
});

test("the script prints usage on bad arguments", async () => {
  const { spawnSync } = await import("node:child_process");
  const script = repoFile("scripts/quality-eval/summary-regenerate.mjs");
  const bad = spawnSync(process.execPath, [script, "--out", "/o"], { encoding: "utf8" });
  assert.equal(bad.status, 1);
  assert.match(bad.stderr, /--sources/);
  assert.match(bad.stderr, /Usage: node scripts\/quality-eval\/summary-regenerate\.mjs/);
  assert.equal(spawnSync(process.execPath, [script, "--help"], { encoding: "utf8" }).status, 0);
});
