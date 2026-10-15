import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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

const DIRECTIONS = [
  { number: 1, tag: "photo-z", id: "d1", text: "Photometric redshift methods.", dates: ["2026-10-14"] },
  { number: 2, tag: "photo-z", id: "d2", text: "Calibration with spectroscopic samples.", dates: ["2026-10-14"] },
  { number: 3, tag: "cluster", id: "d3", text: "Cluster surveys.", dates: ["2026-10-14"] },
];

function metaOf(counts = { rows: 11, selected: 7, skippedSampled: 4, skippedAvailable: 20, withoutAbstract: 0, hadAbstractAtFilter: 8 }) {
  return {
    schemaVersion: 1, createdAt: "2026-10-15T12:00:00.000Z", seed: "fixture-seed", skippedSample: 4,
    sourceDates: ["2026-10-13", "2026-10-14"], columns: qe.SHEET_COLUMNS, counts, directions: DIRECTIONS,
    records: [
      { reportDate: "2026-10-13", promptContractVersion: 4, resultContractVersion: 3, generation: { provider: "deepseek", model: "model-a" }, candidates: 100 },
      { reportDate: "2026-10-14", promptContractVersion: 4, resultContractVersion: 3, generation: { provider: "deepseek", model: "model-a" }, candidates: 100 },
    ],
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

function sheetText(rows = FIXTURE, header = qe.SHEET_COLUMNS) {
  const lines = rows.map(([id, had, category, modelDirections, score, label, labelDirection, note, title]) => {
    const row = {
      date: "2026-10-14", id, title, abstract: `Abstract of ${id}`, had_abstract_at_filter: had,
      model_category: category, model_directions: modelDirections, model_score: score,
      model_outcome: category === "skip" ? "skipped_by_model" : "in_report", label, label_direction: labelDirection, note,
    };
    return header.map(column => row[column] ?? "");
  });
  return qe.encodeCsv([[...header], ...lines]);
}

async function writeSheet(rows = FIXTURE, meta = metaOf(), options = {}) {
  const directory = await temp("sheet");
  await writeFile(path.join(directory, "labels.csv"), options.text ?? sheetText(rows, options.header));
  if (meta) await writeFile(path.join(directory, "labels.meta.json"), JSON.stringify(meta));
  return directory;
}

async function evaluate(directory, options = {}) {
  const outDir = path.join(await temp("report"), "report");
  const result = await qe.runEvaluate({ sheet: path.join(directory, "labels.csv"), outDir, recorded: true, ...options }, { now: () => new Date("2026-10-15T13:00:00.000Z") });
  const json = JSON.parse(await readFile(path.join(outDir, "report.json"), "utf8"));
  const markdown = await readFile(path.join(outDir, "report.md"), "utf8");
  return { result, json, markdown, outDir };
}

test("metrics match a hand count on every label, with and without papers the filter saw no abstract for", async () => {
  const { json } = await evaluate(await writeSheet());

  assert.equal(json.sheet.rows, 11);
  assert.equal(json.sheet.labelled, 9);
  assert.equal(json.sheet.unlabelled, 2);
  assert.equal(json.sheet.labelledYes, 5);
  assert.equal(json.sheet.labelledNo, 4);

  // TP s1 s2 s3 s6, FP s4 s5, FN k1, TN k2 k3.
  const all = json.sets.all;
  assert.equal(all.rows, 9);
  assert.deepEqual([all.counts.tp, all.counts.fp, all.counts.fn, all.counts.tn], [4, 2, 1, 2]);
  close(all.precision.value, 4 / 6); assert.equal(all.precision.n, 6);
  close(all.recall.value, 4 / 5); assert.equal(all.recall.n, 5);
  close(all.f1.value, 8 / 11);

  // Skipped rows were down-sampled: 20 skipped candidates existed, 3 of the sampled ones are labelled.
  close(all.weighted.skippedWeight, 20 / 3);
  close(all.weighted.recall.value, 4 / (4 + 20 / 3));
  close(all.weighted.f1.value, 8 / (8 + 2 + 20 / 3));
  close(all.weighted.precision.value, 4 / 6);

  // Without the three rows the filter saw no abstract for: TP s1 s2 s6, FP s4, FN k1.
  const withAbstract = json.sets.withAbstract;
  assert.equal(withAbstract.rows, 6);
  assert.deepEqual([withAbstract.counts.tp, withAbstract.counts.fp, withAbstract.counts.fn, withAbstract.counts.tn], [3, 1, 1, 1]);
  close(withAbstract.precision.value, 3 / 4); assert.equal(withAbstract.precision.n, 4);
  close(withAbstract.recall.value, 3 / 4);
  close(withAbstract.f1.value, 3 / 4);
  assert.equal(withAbstract.weighted, null, "the number of skipped candidates that had an abstract is unknown");
});

test("direction accuracy counts a hit when a labelled direction is among the model's", async () => {
  const { json } = await evaluate(await writeSheet());

  // True positives with a labelled direction: s1 (hit), s2 (hit), s3 (miss); s6 has none labelled.
  const direction = json.sets.all.direction;
  assert.deepEqual([direction.correct, direction.n, direction.withoutLabelledDirection], [2, 3, 1]);
  close(direction.value, 2 / 3);
  assert.deepEqual([json.sets.withAbstract.direction.correct, json.sets.withAbstract.direction.n], [2, 2]);
  assert.match(json.rules.direction, /label_direction|标注的方向/);
});

test("precision and recall at score thresholds count a paper as selected only when it clears the score", async () => {
  const { json } = await evaluate(await writeSheet());
  const byThreshold = Object.fromEntries(json.sets.all.thresholds.map(entry => [entry.threshold, entry]));

  assert.deepEqual(json.sets.all.thresholds.map(entry => entry.threshold), [0, 50, 60, 70, 80, 90]);
  const expected = {
    0: { p: 4 / 6, r: 4 / 5, rw: 4 / (4 + 20 / 3) },
    50: { p: 4 / 5, r: 4 / 5, rw: 4 / (4 + 20 / 3) },
    60: { p: 3 / 4, r: 3 / 5, rw: 3 / (3 + 1 + 20 / 3) },
    70: { p: 2 / 3, r: 2 / 5, rw: 2 / (2 + 2 + 20 / 3) },
    80: { p: 1 / 2, r: 1 / 5, rw: 1 / (1 + 3 + 20 / 3) },
    90: { p: 1, r: 1 / 5, rw: 1 / (1 + 3 + 20 / 3) },
  };
  for (const [threshold, { p, r, rw }] of Object.entries(expected)) {
    close(byThreshold[threshold].precision.value, p, `precision at ${threshold}`);
    close(byThreshold[threshold].recall.value, r, `recall at ${threshold}`);
    close(byThreshold[threshold].weighted.recall.value, rw, `weighted recall at ${threshold}`);
  }
});

test("per-topic precision and recall follow the model's topic and the labelled direction's topic", async () => {
  const { json } = await evaluate(await writeSheet());
  const topics = Object.fromEntries(json.topics.map(topic => [topic.tag, topic]));

  // photo-z: model filed s1 s2 s4 s6 (3 labelled yes); labelled photo-z directions on s1 s2 s3 (model got 2).
  close(topics["photo-z"].precision.value, 3 / 4); assert.equal(topics["photo-z"].precision.n, 4);
  close(topics["photo-z"].recall.value, 2 / 3); assert.equal(topics["photo-z"].recall.n, 3);
  // cluster: model filed s3 s5 (1 labelled yes); labelled cluster direction on k1, which was skipped.
  close(topics.cluster.precision.value, 1 / 2);
  close(topics.cluster.recall.value, 0); assert.equal(topics.cluster.recall.n, 1);
  assert.equal(json.topicsWithoutDirection, 1, "s6 is labelled yes but names no direction");
});

test("disagreements list missed and wrongly selected papers with their details", async () => {
  const { json, markdown } = await evaluate(await writeSheet());
  const { falseNegatives, falsePositives, directionMismatches } = json.disagreements;

  assert.deepEqual(falseNegatives.map(row => row.id), ["k1"]);
  assert.deepEqual(falseNegatives[0], {
    row: 9, date: "2026-10-14", id: "k1", title: "Missed cluster paper", score: 0, modelCategory: "skip",
    modelDirections: [], labelDirections: [3], note: "missed", hadAbstractAtFilter: true,
  });
  assert.deepEqual(falsePositives.map(row => [row.id, row.score, row.modelDirections]), [["s4", 85, [2]], ["s5", 40, [3]]]);
  assert.deepEqual(directionMismatches.map(row => [row.id, row.modelDirections, row.labelDirections]), [["s3", [3], [1]]]);
  assert.match(markdown, /k1/);
  assert.match(markdown, /Missed cluster paper/);
  assert.match(markdown, /needs the full text/);
});

test("the report names its sources and says what was and was not run", async () => {
  const { json, markdown, result } = await evaluate(await writeSheet());

  assert.equal(json.kind, "recorded");
  assert.equal(json.generatedAt, "2026-10-15T13:00:00.000Z");
  assert.ok(path.isAbsolute(json.sheet.path) && json.sheet.path.endsWith("labels.csv"));
  assert.ok(json.sheet.metaPath.endsWith("labels.meta.json"));
  assert.equal(json.source.seed, "fixture-seed");
  assert.deepEqual(json.source.dates, ["2026-10-13", "2026-10-14"]);
  assert.deepEqual(json.source.skipped, { available: 20, sampled: 4 });
  assert.deepEqual(json.source.records.map(record => [record.reportDate, record.promptContractVersion, record.model, record.provider]), [
    ["2026-10-13", 4, "model-a", "deepseek"], ["2026-10-14", 4, "model-a", "deepseek"],
  ]);
  assert.deepEqual(result.files.map(file => path.basename(file)).sort(), ["report.json", "report.md"]);
  assert.match(markdown, /^# /m);
  assert.match(markdown, /## 主要结果/);
  assert.match(markdown, /## 分歧列表/);
  assert.match(markdown, /66\.7%/);
  assert.match(markdown, /未调用模型|没有调用模型/);
  assert.ok(!markdown.includes("NaN") && !markdown.includes("undefined"));
});

test("a metric with an empty denominator is n/a rather than NaN", async () => {
  const rows = [
    ["a", "true", "skip", "", "0", "n", "", "", "Skipped, not relevant"],
    ["b", "true", "skip", "", "0", "n", "", "", "Skipped, not relevant"],
  ];
  const { json, markdown } = await evaluate(await writeSheet(rows, metaOf({ rows: 2, selected: 0, skippedSampled: 2, skippedAvailable: 2, withoutAbstract: 0, hadAbstractAtFilter: 2 })));

  const all = json.sets.all;
  assert.deepEqual([all.precision.value, all.recall.value, all.f1.value, all.direction.value], [null, null, null, null]);
  assert.equal(all.precision.n, 0);
  assert.equal(all.recall.n, 0);
  assert.ok(all.thresholds.every(entry => entry.precision.value === null && entry.recall.value === null));
  assert.ok(!JSON.stringify(json).includes("NaN"));
  assert.ok(!markdown.includes("NaN"));
  assert.match(markdown, /n\/a/);
});

test("without the skipped counts only unweighted recall is reported", async () => {
  const meta = metaOf();
  delete meta.counts.skippedAvailable;
  const { json, markdown } = await evaluate(await writeSheet(FIXTURE, meta));

  assert.equal(json.sets.all.weighted, null);
  close(json.sets.all.recall.value, 4 / 5);
  assert.match(markdown, /仅未加权/);
});

test("a sheet where every skipped paper is labelled uses the sample size for the weight", async () => {
  // 20 skipped candidates existed; only k1 and k2 are labelled, so each stands for 10.
  const rows = FIXTURE.filter(row => !["k3", "k4"].includes(row[0]));
  const { json } = await evaluate(await writeSheet(rows));
  close(json.sets.all.weighted.skippedWeight, 10);
  close(json.sets.all.weighted.recall.value, 4 / (4 + 10));
});

test("invalid labels, directions and header are rejected with row numbers", async () => {
  const bad = (override, pattern) => async () => {
    const rows = FIXTURE.map(row => [...row]);
    override(rows);
    await assert.rejects(evaluate(await writeSheet(rows)), pattern);
  };
  await bad(rows => { rows[2][5] = "maybe"; }, /row 4: .*label.*maybe/s)();
  await bad(rows => { rows[1][6] = "9"; }, /row 3: .*direction.*9/s)();
  await bad(rows => { rows[3][6] = "1"; }, /row 5: .*label_direction.*label/s)();
  await bad(rows => { rows[6][6] = "1"; }, /row 8: .*label_direction.*label/s)();
  await bad(rows => { rows[0][6] = "one"; }, /row 2: .*label_direction.*one/s)();
  await bad(rows => { rows[0][3] = "7"; }, /row 2: .*model_directions.*7/s)();
  await bad(rows => { rows[0][1] = "maybe"; }, /row 2: .*had_abstract_at_filter/s)();
  await bad(rows => { rows[0][4] = "high"; }, /row 2: .*model_score/s)();

  // Every problem is listed, not just the first.
  const rows = FIXTURE.map(row => [...row]);
  rows[2][5] = "maybe"; rows[1][6] = "9";
  await assert.rejects(evaluate(await writeSheet(rows)), error => /row 3:/.test(error.message) && /row 4:/.test(error.message));

  await assert.rejects(evaluate(await writeSheet(FIXTURE, metaOf(), { header: qe.SHEET_COLUMNS.filter(column => column !== "label") })), /missing column.*label/i);
});

test("labels and flags typed the way a spreadsheet writes them are accepted", async () => {
  const rows = FIXTURE.map(row => [...row]);
  rows[0][5] = " Y "; rows[0][1] = "TRUE"; rows[3][5] = "N"; rows[1][6] = "2; 1"; rows[7][6] = "3,1"; rows[8][5] = "n";
  const { json } = await evaluate(await writeSheet(rows));
  assert.deepEqual([json.sets.all.counts.tp, json.sets.all.counts.fp, json.sets.all.counts.fn], [4, 2, 1]);
});

test("the sheet and meta must be readable", async () => {
  await assert.rejects(evaluate(await writeSheet(FIXTURE, null)), /labels\.meta\.json|--meta/);
  const directory = await writeSheet(FIXTURE, null);
  const elsewhere = await temp("meta");
  await writeFile(path.join(elsewhere, "meta.json"), JSON.stringify(metaOf()));
  const { json } = await evaluate(directory, { meta: path.join(elsewhere, "meta.json") });
  assert.equal(json.sheet.rows, 11);
  await assert.rejects(evaluate(await temp("empty")), /labels\.csv|ENOENT|read/i);
});

test("command-line arguments are validated", () => {
  assert.deepEqual(qe.parseEvaluateArgs(["--sheet", "/s/labels.csv", "--out", "/o", "--recorded"]), { sheet: "/s/labels.csv", outDir: "/o", recorded: true });
  assert.deepEqual(qe.parseEvaluateArgs(["--sheet", "/s", "--meta", "/m", "--out", "/o", "--recorded"]), { sheet: "/s", meta: "/m", outDir: "/o", recorded: true });
  assert.equal(qe.parseEvaluateArgs(["--help"]).help, true);
  assert.throws(() => qe.parseEvaluateArgs(["--out", "/o", "--recorded"]), /--sheet/);
  assert.throws(() => qe.parseEvaluateArgs(["--sheet", "/s", "--recorded"]), /--out/);
  assert.throws(() => qe.parseEvaluateArgs(["--sheet", "/s", "--out", "/o", "--bogus"]), /unknown option.*--bogus/i);
  assert.throws(() => qe.parseEvaluateArgs(["--sheet"]), /--sheet/);
});

test("without --recorded evaluate does not guess: the model rerun is not available yet", async () => {
  await assert.rejects(evaluate(await writeSheet(), { recorded: false }), /--recorded/);
});
