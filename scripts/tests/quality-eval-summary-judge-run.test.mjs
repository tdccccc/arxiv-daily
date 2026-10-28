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

const SENTINEL_KEY = "SENTINEL-API-KEY-7f3a9c";
const SENTINEL_HOST = "sentinel-judge-host.example.invalid";
const SENTINEL_URL = `https://${SENTINEL_HOST}/v1`;
const NOW = new Date("2026-10-29T08:00:00Z");

const judgeToml = (overrides = {}) => [
  "[llm]", `api_key = "${overrides.apiKey ?? SENTINEL_KEY}"`, `base_url = "${overrides.baseUrl ?? SENTINEL_URL}"`,
  `model = "${overrides.model ?? "judge-model"}"`, 'provider = "openai"', ...(overrides.extra ?? []), "",
].join("\n");
const writeJudgeConfig = async (overrides) => {
  const file = path.join(await temp("judge-config"), "judge.toml");
  await writeFile(file, overrides?.raw ?? judgeToml(overrides));
  return file;
};

const FIELD_LABELS = ["研究问题", "方法设计", "核心结果", "研究价值", "适用边界"];
const ZH_HEADINGS = ["研究问题", "方法设计", "核心结果", "主要结论", "关键图表", "贡献与创新点", "适用边界", "学术价值判断"];
const DAILY_IDS = ["2610.00001", "2610.00002", "2610.00003"];
const NOTE_ID = "2607.00001";
const SKIPPED_ID = "2610.00004";

async function fixture() {
  const vault = await temp("vault");
  const sources = await temp("sources");
  await mkdir(path.join(vault, "arxiv-daily", "daily"), { recursive: true });
  await mkdir(path.join(vault, "arxiv-daily", "papers"), { recursive: true });
  const block = id => [`### Paper ${id}`, `- **作者**: A. Author et al.`, `- **arXiv**: [${id}](https://arxiv.org/abs/${id})`,
    ...FIELD_LABELS.map(label => `- **${label}**: ${id} 的${label}。`), ""];
  await writeFile(path.join(vault, "arxiv-daily", "daily", "2026-10-15.md"), ["---", "date: 2026-10-15", "---", "", "# t", "", "## Photo-z", "", ...[...DAILY_IDS, SKIPPED_ID].flatMap(block)].join("\n"));
  await writeFile(path.join(vault, "arxiv-daily", "papers", `${NOTE_ID}.md`), [
    "---", 'title: "Note"', 'authors: "N. Author et al."', `arxiv_id: "${NOTE_ID}"`, "primary_topic: galaxy-cluster", "published: 2026-07-10", "---", "", "# Note", "",
    ...ZH_HEADINGS.flatMap(heading => [`## ${heading}`, "", `${heading}的正文。`, ""]),
  ].join("\n"));
  for (const id of [...DAILY_IDS, NOTE_ID]) {
    await writeFile(path.join(sources, `${id}.json`), JSON.stringify({
      schemaVersion: 1, id, fetchedAt: "2026-10-28T10:00:00.000Z", sourcePath: "html", arxivVersion: "v1",
      budgets: { sectionCharLimit: 16000, paperCharLimit: 100000 },
      content: { abstract: "a", abstractConclusion: "## Abstract\na", fullSections: `## Methods\nWe use 120 clusters in ${id}.`, fullTextFailure: null },
      contentHash: `hash-${id}`,
    }));
  }
  return { vault, sources };
}

const OK_JUDGMENT = JSON.stringify({ verdicts: { accuracy: "ok", completeness: "ok", clarity: "ok" }, issues: [] });

/** A fake model: `script` maps call number (1-based, across the run) to a reply or a thrown error. */
function fakeLlm(script = () => OK_JUDGMENT) {
  const calls = [];
  return {
    calls,
    call: async (messages, options) => {
      calls.push({ messages, options });
      const reply = script(calls.length, messages);
      if (reply instanceof Error) throw reply;
      options?.onMetrics?.({ logicalCalls: 1, attempts: 1, elapsedMs: 5, inputTokens: 100, outputTokens: 10, totalTokens: 110, usageComplete: true });
      return reply;
    },
  };
}
const GENERATOR = { model: "generator-model", topics: [{ tag: "galaxy-cluster", name: "Galaxy Cluster" }] };

async function setup(extra = {}) {
  const { vault, sources } = await fixture();
  const judgeConfigPath = await writeJudgeConfig(extra.judge);
  const outDir = extra.outDir ?? path.join(await temp("out"), "judge");
  const options = { judgeConfigPath, vaultRoot: vault, sourcesDir: sources, outDir, daily: 4, notes: 1, seed: "seed-1", ...extra.options };
  return { options, outDir, vault, sources, judgeConfigPath };
}
const execute = async (setupResult, { llm = fakeLlm(), generator = GENERATOR } = {}, extraOptions = {}) => {
  const log = [];
  const result = await qe.runSummaryJudge({ ...setupResult.options, ...extraOptions }, {
    llm, now: () => NOW, log: line => log.push(line), loadGeneratorConfig: async () => generator,
  });
  return { result, llm, log };
};
const listFiles = async directory => {
  const found = [];
  const walk = async current => {
    for (const entry of await readdir(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) await walk(full); else found.push(full);
    }
  };
  if (existsSync(directory)) await walk(directory);
  return found;
};
const readJudgment = async (outDir, kind, id) => JSON.parse(await readFile(path.join(outDir, "judgments", `${kind}-${id}.json`), "utf8"));

test("without --yes the command prints the plan and creates and calls nothing", async () => {
  const state = await setup();
  const { result, llm, log } = await execute(state);
  assert.equal(result.dryRun, true);
  assert.equal(llm.calls.length, 0);
  assert.equal(existsSync(state.outDir), false);
  const text = log.join("\n");
  assert.match(text, /4 items|daily 3.*note 1|3 daily.*1 note/i);
  assert.match(text, /judge-model/);
  assert.match(text, /openai/);
  assert.match(text, /at most 8 requests|8 requests/);
  assert.match(text, /characters/);
  assert.match(text, /3 characters per token|characters per token/);
  assert.match(text, /2610\.00004[^\n]*|skipped/);
  assert.ok(!text.includes(SENTINEL_KEY) && !text.includes(SENTINEL_HOST));
  assert.equal(result.plan.items.daily, 3);
  assert.equal(result.plan.items.note, 1);
  assert.equal(result.plan.maxRequests, 8);
  assert.ok(result.plan.estimatedTokens > 0);
});

test("the judge must not be the generator model, however the generator is named", async () => {
  const state = await setup();
  await assert.rejects(execute(state, { generator: { ...GENERATOR, model: "  Judge-Model " } }), /different model|same model/i);
  await assert.rejects(execute(state, {}, { generatorModel: "JUDGE-MODEL" }), /different model|same model/i);
  // An explicit name wins over the loaded config.
  const ok = await execute(state, { generator: { ...GENERATOR, model: "judge-model" } }, { generatorModel: "other-model" });
  assert.equal(ok.result.dryRun, true);

  // Through a real --config file too.
  const configPath = path.join(await temp("config"), "config.toml");
  await writeFile(configPath, ["schema_version = 1", `vault_root = ${JSON.stringify(state.vault)}`, "", "[llm]", 'api_key = "other-secret"', 'base_url = "https://api.example.com/v1"', 'model = "Judge-Model"', "", "[arxiv]", 'categories = ["astro-ph"]', 'timezone = "UTC"', ""].join("\n"));
  await assert.rejects(qe.runSummaryJudge({ ...state.options, configPath }, { llm: fakeLlm(), now: () => NOW }), /different model|same model/i);
});

test("an unknown generator model is refused unless --generator-model names it", async () => {
  const state = await setup();
  await assert.rejects(execute(state, { generator: null }), /--generator-model/);
  const named = await execute(state, { generator: null }, { generatorModel: "generator-model" });
  assert.equal(named.result.dryRun, true);
});

test("a valid reply on the first try is recorded with the item, judge and prompt identity", async () => {
  const state = await setup();
  const { result, llm } = await execute(state, {}, { yes: true });
  assert.equal(llm.calls.length, 4);
  assert.deepEqual(llm.calls.map(call => call.options.temperature), [0, 0, 0, 0]);
  assert.deepEqual(result.counts, { judged: 4, invalid: 0, errors: 0, reused: 0, skipped: 1, remaining: 0 });

  const record = await readJudgment(state.outDir, "daily", "2610.00001");
  assert.equal(record.status, "judged");
  assert.deepEqual(record.item, {
    kind: "daily", id: "2610.00001", title: "Paper 2610.00001", date: "2026-10-15", topic: "Photo-z",
    source: { sourcePath: "html", arxivVersion: "v1", fetchedAt: "2026-10-28T10:00:00.000Z", contentHash: "hash-2610.00001" },
  });
  assert.equal(record.judge.model, "judge-model");
  assert.equal(record.judge.provider, "openai");
  assert.ok("thinkingMode" in record.judge && "reasoningEffort" in record.judge);
  assert.match(record.judgePromptHash, /^[0-9a-f]{64}$/);
  assert.equal(record.attempts.length, 1);
  assert.equal(record.attempts[0].raw, OK_JUDGMENT);
  assert.equal(record.attempts[0].reasons, null);
  assert.deepEqual(record.attempts[0].usage, { inputTokens: 100, outputTokens: 10, totalTokens: 110, usageComplete: true });
  assert.equal(record.invalid, false);
  assert.deepEqual(record.judgment.verdicts, { accuracy: "ok", completeness: "ok", clarity: "ok" });
  assert.equal(record.startedAt, NOW.toISOString());
  const note = await readJudgment(state.outDir, "note", NOTE_ID);
  assert.equal(note.item.version, "current");
  assert.equal(note.item.topic, "galaxy-cluster");
});

test("an invalid first reply gets one retry that carries the reasons", async () => {
  const state = await setup({ options: { daily: 1, notes: 0, ids: ["2610.00001"] } });
  const llm = fakeLlm(call => (call === 1 ? "这不是 JSON" : OK_JUDGMENT));
  const { result } = await execute(state, { llm }, { yes: true });
  assert.equal(llm.calls.length, 2);
  const retry = llm.calls[1].messages;
  assert.deepEqual(retry.map(message => message.role), ["system", "user", "assistant", "user"]);
  assert.equal(retry[2].content, "这不是 JSON");
  assert.ok(retry[3].content.includes("JSON"));
  assert.ok(retry[3].content.includes("上一次的输出没有通过程序校验"));
  const record = await readJudgment(state.outDir, "daily", "2610.00001");
  assert.equal(record.status, "judged");
  assert.equal(record.attempts.length, 2);
  assert.ok(record.attempts[0].reasons.length > 0);
  assert.equal(record.attempts[1].reasons, null);
  assert.equal(result.counts.judged, 1);
});

test("two invalid replies make the judgment invalid, with the reasons", async () => {
  const state = await setup({ options: { daily: 1, notes: 0, ids: ["2610.00001"] } });
  const bad = JSON.stringify({ verdicts: { accuracy: "major", completeness: "ok", clarity: "ok" }, issues: [] });
  const llm = fakeLlm(() => bad);
  const { result } = await execute(state, { llm }, { yes: true });
  assert.equal(llm.calls.length, 2);
  const record = await readJudgment(state.outDir, "daily", "2610.00001");
  assert.equal(record.status, "invalid");
  assert.equal(record.invalid, true);
  assert.equal(record.judgment, null);
  assert.ok(record.attempts.every(attempt => attempt.reasons.some(reason => reason.includes("verdicts.accuracy"))));
  assert.deepEqual(result.counts.invalid, 1);
});

test("a rerun skips what is already judged, and --rejudge does it again", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const second = await execute(state, {}, { yes: true });
  assert.equal(second.llm.calls.length, 0);
  assert.equal(second.result.counts.reused, 4);
  assert.equal(second.result.counts.judged, 0);
  const third = await execute(state, {}, { yes: true, rejudge: true });
  assert.equal(third.llm.calls.length, 4);
  assert.equal(third.result.counts.judged, 4);
});

test("a changed judge model, prompt or source is judged again", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const file = path.join(state.outDir, "judgments", "daily-2610.00001.json");
  const record = JSON.parse(await readFile(file, "utf8"));
  await writeFile(file, JSON.stringify({ ...record, judgePromptHash: "0".repeat(64) }));
  const afterPrompt = await execute(state, {}, { yes: true });
  assert.equal(afterPrompt.llm.calls.length, 1);

  await writeFile(file, JSON.stringify({ ...JSON.parse(await readFile(file, "utf8")), item: { ...record.item, source: { ...record.item.source, contentHash: "other" } } }));
  assert.equal((await execute(state, {}, { yes: true })).llm.calls.length, 1);

  const otherModel = { ...state, options: { ...state.options, judgeConfigPath: await writeJudgeConfig({ model: "judge-model-2" }) } };
  const afterModel = await execute(otherModel, {}, { yes: true });
  assert.equal(afterModel.llm.calls.length, 4);
});

test("an item that errored is retried on the next run, and three errors in a row stop the run", async () => {
  const state = await setup();
  const llm = fakeLlm(() => new Error(`HTTP 500 from ${SENTINEL_URL} using ${SENTINEL_KEY}`));
  const first = await execute(state, { llm }, { yes: true });
  assert.equal(llm.calls.length, 3);
  assert.equal(first.result.stopped, "errors");
  assert.deepEqual(first.result.counts, { judged: 0, invalid: 0, errors: 3, reused: 0, skipped: 1, remaining: 1 });
  const errored = await readJudgment(state.outDir, "daily", "2610.00001");
  assert.equal(errored.status, "error");
  assert.ok(errored.error.length > 0);

  const second = await execute(state, {}, { yes: true });
  assert.equal(second.llm.calls.length, 4, "errored and unattempted items are judged now");
  assert.equal(second.result.counts.judged, 4);
});

test("a success between errors resets the count", async () => {
  const state = await setup();
  const llm = fakeLlm(call => (call === 1 || call === 2 || call === 4 || call === 5 ? new Error("HTTP 503") : OK_JUDGMENT));
  const { result } = await execute(state, { llm }, { yes: true });
  assert.equal(llm.calls.length, 4);
  assert.equal(result.stopped, null);
  assert.equal(result.counts.errors, 3);
  assert.equal(result.counts.judged, 1);
});

test("the judge's key and address never reach a written file or the output", async () => {
  const state = await setup();
  const llm = fakeLlm(call => (call === 2 ? new Error(`request to ${SENTINEL_URL} failed with key ${SENTINEL_KEY}`) : OK_JUDGMENT));
  const { log } = await execute(state, { llm }, { yes: true });
  const files = await listFiles(state.outDir);
  assert.ok(files.length >= 5);
  for (const file of files) {
    const text = await readFile(file, "utf8");
    assert.ok(!text.includes(SENTINEL_KEY), file);
    assert.ok(!text.includes(SENTINEL_HOST), file);
  }
  assert.ok(!log.join("\n").includes(SENTINEL_KEY));
  assert.ok(!log.join("\n").includes(SENTINEL_HOST));
});

test("run.json records the arguments without secrets, the seed and the sample", async () => {
  const state = await setup();
  await execute(state, {}, { yes: true });
  const run = JSON.parse(await readFile(path.join(state.outDir, "run.json"), "utf8"));
  assert.equal(run.schemaVersion, 1);
  assert.equal(run.seed, "seed-1");
  assert.deepEqual(run.requested, { daily: 4, notes: 1 });
  assert.deepEqual(run.sample.daily, [...DAILY_IDS, SKIPPED_ID]);
  assert.deepEqual(run.sample.notes, [NOTE_ID]);
  assert.equal(run.judge.model, "judge-model");
  assert.equal(run.generatorModel, "generator-model");
  assert.ok(Array.isArray(run.skipped));
  assert.equal(run.counts.judged, 4);
  assert.ok(!("apiKey" in run.judge) && !("baseUrl" in run.judge));
});

test("judge config errors never echo the file", async () => {
  const broken = await writeJudgeConfig({ raw: `[llm\napi_key = "${SENTINEL_KEY}"` });
  const missing = await writeJudgeConfig({ raw: `[llm]\napi_key = "${SENTINEL_KEY}"\nmodel = "m"\n` });
  const badUrl = await writeJudgeConfig({ baseUrl: "not a url" });
  for (const judgeConfigPath of [broken, missing, badUrl, path.join(await temp("none"), "absent.toml")]) {
    const state = await setup();
    await assert.rejects(execute({ ...state, options: { ...state.options, judgeConfigPath } }), error => {
      assert.ok(!String(error.message).includes(SENTINEL_KEY), error.message);
      assert.ok(!String(error.message).includes("not a url"), error.message);
      return true;
    });
  }
});

test("command-line arguments are validated", () => {
  const parsed = qe.parseSummaryJudgeArgs([
    "--judge-config", "/j.toml", "--config", "/c.toml", "--generator-model", "g", "--vault", "/v", "--daily-dir", "d", "--papers-dir", "p", "--sources", "/s", "--out", "/o",
    "--daily", "7", "--notes", "0", "--seed", "x", "--ids", "2610.00001,2607.00001", "--from", "2026-10-01", "--to", "2026-10-31", "--include-older-notes", "--yes", "--rejudge",
  ]);
  assert.deepEqual(parsed, {
    judgeConfigPath: "/j.toml", configPath: "/c.toml", generatorModel: "g", vaultRoot: "/v", dailyDir: "d", papersDir: "p", sourcesDir: "/s", outDir: "/o",
    daily: 7, notes: 0, seed: "x", ids: ["2610.00001", "2607.00001"], from: "2026-10-01", to: "2026-10-31", includeOlderNotes: true, yes: true, rejudge: true,
  });
  assert.equal(qe.parseSummaryJudgeArgs(["--help"]).help, true);
  const base = ["--judge-config", "/j", "--sources", "/s", "--out", "/o"];
  for (const missing of ["--judge-config", "--sources", "--out"]) {
    const index = base.indexOf(missing);
    assert.throws(() => qe.parseSummaryJudgeArgs(base.filter((_, position) => position !== index && position !== index + 1)), new RegExp(missing));
  }
  assert.throws(() => qe.parseSummaryJudgeArgs([...base, "--daily", "-1"]), /--daily/);
  assert.throws(() => qe.parseSummaryJudgeArgs([...base, "--notes", "x"]), /--notes/);
  assert.throws(() => qe.parseSummaryJudgeArgs([...base, "--ids", ","]), /--ids/);
  assert.throws(() => qe.parseSummaryJudgeArgs([...base, "--bogus"]), /unknown option.*--bogus/i);
  assert.throws(() => qe.parseSummaryJudgeArgs([...base, "--from", "2026-13-01"]), /--from must be a date/);
});

test("the script prints usage on bad arguments and never needs a model for --help", async () => {
  const { spawnSync } = await import("node:child_process");
  const script = repoFile("scripts/quality-eval/summary-judge.mjs");
  const bad = spawnSync(process.execPath, [script, "--sources", "/s"], { encoding: "utf8" });
  assert.equal(bad.status, 1);
  assert.match(bad.stderr, /--judge-config/);
  assert.match(bad.stderr, /Usage: node scripts\/quality-eval\/summary-judge\.mjs/);
  assert.equal(spawnSync(process.execPath, [script, "--help"], { encoding: "utf8" }).status, 0);
});
