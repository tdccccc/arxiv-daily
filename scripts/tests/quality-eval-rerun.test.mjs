import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
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

const close = (actual, expected, message) => {
  if (expected === null) return assert.equal(actual, null, message);
  assert.ok(actual !== null && Math.abs(actual - expected) < 1e-9, `${message ?? ""} expected ${expected}, got ${actual}`);
};

const API_KEY = "sk-never-print-123";
const SECRETS = [API_KEY, "tok-abc", "hunter2", "api.example.com"];

const DIRECTIONS = [
  { number: 1, tag: "photo-z", id: "d1", text: "Photometric redshift methods.", dates: ["2026-10-14"] },
  { number: 2, tag: "photo-z", id: "d2", text: "Calibration with spectroscopic samples.", dates: ["2026-10-14"] },
  { number: 3, tag: "cluster", id: "d3", text: "Cluster surveys.", dates: ["2026-10-14"] },
];

function metaOf(overrides = {}) {
  return {
    schemaVersion: 1, createdAt: "2026-10-15T12:00:00.000Z", seed: "fixture-seed", skippedSample: 4,
    sourceDates: ["2026-10-14"], columns: qe.SHEET_COLUMNS,
    counts: { rows: 11, selected: 7, skippedSampled: 4, skippedAvailable: 20, withoutAbstract: 0, hadAbstractAtFilter: 8 },
    directions: DIRECTIONS,
    records: [{ reportDate: "2026-10-14", promptContractVersion: 4, resultContractVersion: 3, generation: { provider: "deepseek", model: "model-a" }, candidates: 100 }],
    ...overrides,
  };
}

/** [id, hadAbstract, category, modelDirections, score, label, labelDirection, note, title] */
const FIXTURE = [
  ["s1", "true", "photo-z", "1", "90", "y", "1", "", "Quoted \"title\", with comma"],
  ["s2", "true", "photo-z", "1;2", "70", "y", "2", "", "中文标题"],
  ["s3", "false", "cluster", "3", "60", "y", "1", "needs the full text", "Wrong topic"],
  ["s4", "true", "photo-z", "2", "85", "n", "", "", "Looks relevant, is not"],
  ["s5", "false", "cluster", "3", "40", "n", "", "", "Low score, not relevant"],
  ["s6", "true", "photo-z", "1", "55", "y", "", "", "Relevant, no direction labelled"],
  ["s7", "true", "photo-z", "1", "80", "", "", "", "Not labelled yet"],
  ["k1", "true", "skip", "", "0", "y", "3", "missed", "Missed cluster paper"],
  ["k2", "true", "skip", "", "0", "n", "", "", "Correctly skipped"],
  ["k3", "false", "skip", "", "0", "n", "", "", "Correctly skipped, title only"],
  ["k4", "true", "skip", "", "0", "", "", "", "Not labelled yet, skipped"],
];

function sheetText(rows) {
  return qe.encodeCsv([[...qe.SHEET_COLUMNS], ...rows.map(([id, had, category, modelDirections, score, label, labelDirection, note, title]) => {
    const row = {
      date: "2026-10-14", id, title, abstract: `Abstract of ${id}`, had_abstract_at_filter: had,
      model_category: category, model_directions: modelDirections, model_score: score,
      model_outcome: category === "skip" ? "skipped_by_model" : "in_report", label, label_direction: labelDirection, note,
    };
    return qe.SHEET_COLUMNS.map(column => row[column] ?? "");
  })]);
}

async function writeSheet(rows = FIXTURE, meta = metaOf()) {
  const directory = await temp("sheet");
  await writeFile(path.join(directory, "labels.csv"), sheetText(rows));
  await writeFile(path.join(directory, "labels.meta.json"), JSON.stringify(meta));
  return directory;
}

async function writeConfig(topicsToml = "") {
  const directory = await temp("config");
  const configPath = path.join(directory, "config.toml");
  await writeFile(configPath, [
    "schema_version = 1", `vault_root = ${JSON.stringify(directory)}`, "",
    "[llm]", `api_key = "${API_KEY}"`, 'base_url = "https://user:hunter2@api.example.com/v1?token=tok-abc"', 'model = "model-b"', 'provider = "deepseek"', "",
    "[arxiv]", 'categories = ["astro-ph.CO", "astro-ph.GA"]', 'timezone = "UTC"', "",
    topicsToml,
  ].join("\n"));
  return configPath;
}

const configTopics = (directionText = "Photometric redshift methods.") => [
  "[[arxiv.topics]]", 'name = "Photo-z"', 'tag = "photo-z"', "detail = false",
  "[[arxiv.topics.directions]]", `text = ${JSON.stringify(directionText)}`, 'origin = "manual"',
  "[[arxiv.topics.directions]]", 'text = "Calibration with spectroscopic samples."', 'origin = "manual"',
  "[[arxiv.topics]]", 'name = "Cluster"', 'tag = "cluster"', "detail = false",
  "[[arxiv.topics.directions]]", 'text = "Cluster surveys."', 'origin = "manual"', "",
].join("\n");

const paper = (id, category, directions, relevanceScore) => ({ id, category, directions, relevanceScore });
const answer = papers => JSON.stringify({ papers });

/** Run 1 omits s3, s5, s6, k2 and k3: TP s1 s2 k1, FP s4, FN s3 s6, TN s5 k2 k3. */
const RUN_1 = answer([
  paper("s1", "photo-z", ["photo-z#1"], 90), paper("s2", "photo-z", ["photo-z#2"], 70),
  paper("s4", "photo-z", ["photo-z#1"], 85), paper("k1", "cluster", ["cluster#1"], 66),
]);
/** Run 3 selects s1 s2 s3 only. */
const RUN_3 = answer([
  paper("s1", "photo-z", ["photo-z#1"], 90), paper("s2", "photo-z", ["photo-z#2"], 70), paper("s3", "cluster", ["cluster#1"], 60),
]);

/** A fake model: each call answers with the next response (a string, or an Error to throw). */
function fakeLlm(responses) {
  const calls = [];
  return {
    calls,
    call: async (messages, options) => {
      calls.push({ messages, options });
      const next = responses[Math.min(calls.length - 1, responses.length - 1)];
      if (next instanceof Error) throw next;
      options?.onMetrics?.({ logicalCalls: 1, attempts: 1, elapsedMs: 5, usageComplete: true, inputTokens: 1000, outputTokens: 50, totalTokens: 1050 });
      return next;
    },
  };
}

async function rerun(directory, llm, options = {}) {
  const outDir = path.join(await temp("report"), "report");
  const lines = [];
  const result = await qe.runEvaluate(
    { sheet: path.join(directory, "labels.csv"), outDir, yes: true, configPath: options.configPath ?? await writeConfig(), ...options },
    { llm, now: () => new Date("2026-10-15T13:00:00.000Z"), log: line => lines.push(line) },
  );
  const read = async name => readFile(path.join(outDir, name), "utf8").catch(() => null);
  const jsonText = await read("report.json");
  return { result, outDir, lines, json: jsonText === null ? null : JSON.parse(jsonText), markdown: await read("report.md"), jsonText };
}

test("without --yes the plan is printed and no model is called", async () => {
  const llm = fakeLlm([RUN_1]);
  const { result, lines, outDir } = await rerun(await writeSheet(), llm, { runs: 2, yes: false });

  assert.equal(result.dryRun, true);
  assert.equal(llm.calls.length, 0);
  assert.deepEqual(result.files, []);
  assert.deepEqual(await readdir(path.dirname(outDir)), [], "nothing is written");
  const plan = lines.join("\n");
  assert.match(plan, /2 model calls/);
  assert.match(plan, /9 papers/);
  assert.match(plan, /model-b/);
  assert.match(plan, /deepseek/);
  assert.match(plan, /--yes/);
  for (const secret of SECRETS) assert.ok(!plan.includes(secret), `${secret} must not be printed`);
});

test("the request is built by core from the sheet's direction snapshot and the labelled rows only", async () => {
  const llm = fakeLlm([RUN_1]);
  const { json } = await rerun(await writeSheet(), llm);

  assert.equal(llm.calls.length, 1, "one request for the whole labelled set");
  const [system, user] = llm.calls[0].messages;
  assert.match(system.content, /- photo-z:\n {2}- photo-z#1: Photometric redshift methods\.\n {2}- photo-z#2: Calibration with spectroscopic samples\.\n- cluster:\n {2}- cluster#1: Cluster surveys\./);
  assert.match(user.content, /astro-ph\.CO/);
  for (const id of ["s1", "s2", "s3", "s4", "s5", "s6", "k1", "k2", "k3"]) assert.match(user.content, new RegExp(`ID: ${id}\\nTitle: .*\\nAbstract: Abstract of ${id}\\n`));
  assert.doesNotMatch(user.content, /ID: s7|ID: k4/, "unlabelled rows are not sent");
  assert.equal(llm.calls[0].options.temperature, 0);
  assert.equal(json.current.topicSource, "sheet-snapshot");
  assert.equal(json.current.papers, 9);
  assert.equal(json.current.requestCharacters, system.content.length + user.content.length);
});

test("a run is scored like the recorded decisions, weighting by the sample's original strata", async () => {
  const { json } = await rerun(await writeSheet(), fakeLlm([RUN_1]));
  const run = json.runs[0];

  assert.equal(run.status, "ok");
  assert.equal(run.omitted, 5, "papers the model left out count as skipped");
  const all = run.sets.all;
  assert.deepEqual([all.counts.tp, all.counts.fp, all.counts.fn, all.counts.tn], [3, 1, 2, 3]);
  close(all.precision.value, 3 / 4);
  close(all.recall.value, 3 / 5);
  close(all.f1.value, 6 / 9);
  // k1 k2 k3 came from the skipped stratum (20 candidates, 3 labelled), whatever the new model says.
  close(all.weighted.skippedWeight, 20 / 3);
  close(all.weighted.recall.value, (2 + 20 / 3) / (2 + 20 / 3 + 2));
  close(all.weighted.precision.value, (2 + 20 / 3) / (2 + 20 / 3 + 1));
  assert.deepEqual([all.direction.correct, all.direction.n], [3, 3]);
  assert.deepEqual(run.disagreements.falseNegatives.map(row => row.id), ["s3", "s6"]);
  assert.deepEqual(run.disagreements.falsePositives.map(row => [row.id, row.modelDirections]), [["s4", [1]]]);
  const topics = Object.fromEntries(run.topics.map(topic => [topic.tag, topic]));
  close(topics["photo-z"].precision.value, 2 / 3);
  close(topics["photo-z"].recall.value, 2 / 3);
  close(topics.cluster.precision.value, 1);
  close(topics.cluster.recall.value, 1);
  assert.equal(run.usage.inputTokens, 1000);
});

test("the report sets the recorded decisions beside each rerun and names the current prompt, model and topic source", async () => {
  const { json, markdown } = await rerun(await writeSheet(), fakeLlm([RUN_1]));

  assert.equal(json.kind, "rerun");
  assert.equal(json.generatedAt, "2026-10-15T13:00:00.000Z");
  assert.equal(json.current.promptContractVersion, qe.DAILY_FILTER_PROMPT_CONTRACT_VERSION);
  assert.equal(json.current.resultContractVersion, qe.DAILY_FILTER_RESULT_CONTRACT_VERSION);
  assert.deepEqual([json.current.provider, json.current.model, json.current.mode.kind], ["deepseek", "model-b", "reasoning-thinking"]);
  assert.deepEqual(json.current.categories, ["astro-ph.CO", "astro-ph.GA"]);
  assert.deepEqual(json.source.records.map(record => [record.model, record.promptContractVersion]), [["model-a", 4]]);
  assert.deepEqual(json.comparison.map(entry => [entry.name, entry.rows]), [["recorded", 9], ["run 1", 9]]);
  close(json.comparison[0].precision.value, 4 / 6);
  close(json.comparison[0].recall.value, 4 / 5);
  close(json.comparison[1].precision.value, 3 / 4);
  close(json.comparison[1].weightedRecall.value, (2 + 20 / 3) / (2 + 20 / 3 + 2));
  assert.equal(json.agreement, null, "one successful run has nothing to agree with");
  assert.match(markdown, /^# .*重跑/m);
  assert.match(markdown, /model-b/);
  assert.match(markdown, /记录的判定/);
  assert.match(markdown, /第 1 次重跑/);
  assert.match(markdown, /仅有一次成功的运行|只有一次成功/);
  assert.ok(!markdown.includes("NaN"));
});

test("a failing run is reported in its own section while the others continue, and flip-flopping papers are listed", async () => {
  const llm = fakeLlm([RUN_1, new Error(`upstream 500 for ${API_KEY} at https://user:hunter2@api.example.com/v1?token=tok-abc`), RUN_3]);
  const { json, markdown, jsonText, result } = await rerun(await writeSheet(), llm, { runs: 3 });

  assert.equal(llm.calls.length, 3);
  assert.deepEqual(json.runs.map(run => run.status), ["ok", "failed", "ok"]);
  assert.equal(json.runs[1].failure, "model");
  assert.match(json.runs[1].error, /upstream 500/);
  assert.equal(result.allRunsFailed, false);
  for (const secret of SECRETS) {
    assert.ok(!jsonText.includes(secret) && !markdown.includes(secret), `${secret} must not be reported`);
  }

  // Selected: run 1 {s1 s2 s4 k1}, run 3 {s1 s2 s3}; identical on 6 of 9 papers.
  assert.equal(json.agreement.runs, 2);
  assert.deepEqual([json.agreement.identical.hits, json.agreement.identical.n], [6, 9]);
  assert.deepEqual(json.agreement.flipFlops.map(row => [row.id, row.selected]), [["k1", [true, false]], ["s3", [false, true]], ["s4", [true, false]]]);
  assert.deepEqual(json.agreement.flipFlops.map(row => row.label), ["y", "y", "n"]);
  assert.deepEqual(json.comparison.map(entry => entry.name), ["recorded", "run 1", "run 3"]);
  assert.match(markdown, /第 2 次重跑[\s\S]*失败/);
  assert.match(markdown, /摇摆|不稳定/);
});

test("a response that breaks the filter contract fails only its run", async () => {
  const { json, result } = await rerun(await writeSheet(), fakeLlm(["not json at all", answer([paper("nope", "skip", [], 0)]), RUN_1]), { runs: 3 });

  assert.deepEqual(json.runs.map(run => [run.status, run.failure ?? null]), [["failed", "validation"], ["failed", "validation"], ["ok", null]]);
  assert.match(json.runs[0].error, /strict JSON/);
  assert.match(json.runs[1].error, /contract/);
  assert.equal(result.allRunsFailed, false);
});

test("when every run fails the report is still written and the result says so", async () => {
  const { json, result, markdown } = await rerun(await writeSheet(), fakeLlm([new Error("down")]), { runs: 2 });

  assert.equal(result.allRunsFailed, true);
  assert.deepEqual(json.runs.map(run => run.status), ["failed", "failed"]);
  assert.equal(json.agreement, null);
  assert.deepEqual(json.comparison.map(entry => entry.name), ["recorded"]);
  assert.match(markdown, /全部.*失败/);
});

test("the CLI config's topics can replace the snapshot and a mismatch withholds direction accuracy", async () => {
  const same = await rerun(await writeSheet(), fakeLlm([RUN_1]), { topicsFromConfig: true, configPath: await writeConfig(configTopics()) });
  assert.equal(same.json.current.topicSource, "config");
  assert.equal(same.json.directionWithheld, null);
  assert.deepEqual(same.json.warnings, []);
  assert.deepEqual([same.json.runs[0].sets.all.direction.correct, same.json.runs[0].sets.all.direction.n], [3, 3]);

  const edited = await rerun(await writeSheet(), fakeLlm([RUN_1]), { topicsFromConfig: true, configPath: await writeConfig(configTopics("Photometric redshifts, rewritten.")) });
  assert.match(edited.json.directionWithheld, /Photometric redshift/);
  assert.ok(edited.json.warnings.some(warning => /方向|direction/i.test(warning)));
  assert.equal(edited.json.runs[0].sets.all.direction, null);
  assert.equal(edited.json.runs[0].topics, null);
  assert.equal(edited.json.runs[0].disagreements.directionMismatches, null);
  close(edited.json.runs[0].sets.all.precision.value, 3 / 4, "selection metrics are still reported");
  assert.match(edited.markdown, /方向列表与标注时不同/);
  assert.ok(edited.markdown.indexOf("方向列表与标注时不同") < edited.markdown.indexOf("## 第 1 次重跑"), "the warning comes before the results");
  assert.match(edited.json.warnings.join("\n"), /rewritten/);
});

test("topics come from the config only on request, and a config without topics is refused", async () => {
  await assert.rejects(rerun(await writeSheet(), fakeLlm([RUN_1]), { topicsFromConfig: true }), /no topics/i);
  const llm = fakeLlm([RUN_1]);
  await rerun(await writeSheet(), llm, { configPath: await writeConfig(configTopics("Used only if asked for")) });
  assert.doesNotMatch(llm.calls[0].messages[0].content, /Used only if asked for/);
});

test("a sheet spanning days with different direction lists needs --topics-from-config", async () => {
  const meta = metaOf({
    sourceDates: ["2026-10-13", "2026-10-14"],
    directions: DIRECTIONS.map((direction, index) => ({ ...direction, dates: index === 0 ? ["2026-10-13"] : ["2026-10-13", "2026-10-14"] })),
  });
  const llm = fakeLlm([RUN_1]);
  await assert.rejects(rerun(await writeSheet(FIXTURE, meta), llm), /different direction lists.*--topics-from-config/s);
  assert.equal(llm.calls.length, 0);
});

test("a paper that appears on two days cannot be sent as one request", async () => {
  const rows = [...FIXTURE, ["s1", "true", "photo-z", "1", "90", "y", "1", "", "Same paper again"]];
  await assert.rejects(rerun(await writeSheet(rows), fakeLlm([RUN_1])), /s1.*more than once|duplicate/i);
});

test("the API key and endpoint never reach the report", async () => {
  const { jsonText, markdown, lines } = await rerun(await writeSheet(), fakeLlm([RUN_1]), { runs: 1 });
  for (const secret of SECRETS) {
    assert.ok(!jsonText.includes(secret) && !markdown.includes(secret) && !lines.join("\n").includes(secret), `${secret} leaked`);
  }
});

test("recorded scoring still needs no config or model", async () => {
  const directory = await writeSheet();
  const outDir = path.join(await temp("report"), "report");
  const result = await qe.runEvaluate({ sheet: path.join(directory, "labels.csv"), outDir, recorded: true });
  assert.equal(result.report.kind, "recorded");
});

test("rerun arguments are validated", () => {
  assert.deepEqual(qe.parseEvaluateArgs(["--sheet", "/s", "--out", "/o", "--runs", "3", "--yes", "--config", "/c.toml", "--topics-from-config"]),
    { sheet: "/s", outDir: "/o", runs: 3, yes: true, configPath: "/c.toml", topicsFromConfig: true });
  assert.deepEqual(qe.parseEvaluateArgs(["--sheet", "/s", "--out", "/o"]), { sheet: "/s", outDir: "/o" });
  assert.throws(() => qe.parseEvaluateArgs(["--sheet", "/s", "--out", "/o", "--runs", "0"]), /--runs/);
  assert.throws(() => qe.parseEvaluateArgs(["--sheet", "/s", "--out", "/o", "--runs", "many"]), /--runs/);
  assert.throws(() => qe.parseEvaluateArgs(["--sheet", "/s", "--out", "/o", "--runs", "99"]), /--runs/);
  assert.throws(() => qe.parseEvaluateArgs(["--sheet", "/s", "--out", "/o", "--recorded", "--runs", "2"]), /--recorded.*--runs|--runs.*--recorded/);
  assert.throws(() => qe.parseEvaluateArgs(["--sheet", "/s", "--out", "/o", "--recorded", "--yes"]), /--recorded/);
});
