import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
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

const DIRECTIONS = [
  { number: 1, tag: "photo-z", id: "d1", text: "Photometric redshift methods.", dates: ["2026-10-14"] },
  { number: 2, tag: "photo-z", id: "d2", text: "Calibration with spectroscopic samples.", dates: ["2026-10-14"] },
  { number: 3, tag: "cluster", id: "d3", text: "Cluster surveys.", dates: ["2026-10-14"] },
];

function metaOf() {
  return {
    schemaVersion: 1, createdAt: "2026-10-15T12:00:00.000Z", seed: "fixture-seed", skippedSample: 4,
    sourceDates: ["2026-10-14"], columns: qe.SHEET_COLUMNS,
    counts: { rows: 11, selected: 7, skippedSampled: 4, skippedAvailable: 20, withoutAbstract: 0, hadAbstractAtFilter: 8 },
    directions: DIRECTIONS,
    records: [{ reportDate: "2026-10-14", promptContractVersion: 4, resultContractVersion: 3, generation: { provider: "deepseek", model: "model-a" }, candidates: 100 }],
  };
}

/** [id, hadAbstract, category, modelDirections, score, label, labelDirection, note, title] */
const FIXTURE = [
  ["2610.00001", "true", "photo-z", "1", "90", "y", "1", "", "Quoted \"title\", with comma"],
  ["2610.00002", "true", "photo-z", "1;2", "70", "y", "2", "", "中文标题"],
  ["2610.00003", "false", "cluster", "3", "60", "y", "1", "needs the full text", "Wrong topic"],
  ["2610.00004", "true", "photo-z", "2", "85", "n", "", "", "Looks relevant, is not"],
  ["2610.00005", "false", "cluster", "3", "40", "n", "", "", "Low score, not relevant"],
  ["2610.00006", "true", "photo-z", "1", "55", "y", "", "", "Relevant, no direction labelled"],
  ["2610.00007", "true", "photo-z", "1", "80", "", "", "", "Not labelled yet"],
  ["2610.00011", "true", "skip", "", "0", "y", "3", "missed", "Missed cluster paper"],
  ["2610.00012", "true", "skip", "", "0", "n", "", "", "Correctly skipped"],
  ["2610.00013", "false", "skip", "", "0", "n", "", "", "Correctly skipped, title only"],
  ["2610.00014", "true", "skip", "", "0", "", "", "", "Not labelled yet, skipped"],
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

async function writeSheet(rows = FIXTURE) {
  const directory = await temp("sheet");
  await writeFile(path.join(directory, "labels.csv"), sheetText(rows));
  await writeFile(path.join(directory, "labels.meta.json"), JSON.stringify(metaOf()));
  return directory;
}

async function writeConfig() {
  const directory = await temp("config");
  const configPath = path.join(directory, "config.toml");
  await writeFile(configPath, [
    "schema_version = 1", `vault_root = ${JSON.stringify(directory)}`, "",
    "[llm]", 'api_key = "sk-never-print-123"', 'base_url = "https://api.example.com/v1"', 'model = "model-b"', 'provider = "deepseek"', "",
    "[arxiv]", 'categories = ["astro-ph.CO"]', 'timezone = "UTC"', "",
  ].join("\n"));
  return configPath;
}

const paper = (id, category, directions, relevanceScore) => ({ id, category, directions, relevanceScore });
const answer = papers => JSON.stringify({ papers });

/** Selects 2610.00001, 2610.00002, 2610.00004 and 2610.00011. */
const RUN_1 = answer([
  paper("2610.00001", "photo-z", ["photo-z#1"], 90), paper("2610.00002", "photo-z", ["photo-z#2"], 70),
  paper("2610.00004", "photo-z", ["photo-z#1"], 85), paper("2610.00011", "cluster", ["cluster#1"], 66),
]);

function fakeLlm(responses) {
  const calls = [];
  return {
    calls,
    call: async (messages, options) => {
      calls.push({ messages, options });
      const next = responses[Math.min(calls.length - 1, responses.length - 1)];
      if (next instanceof Error) throw next;
      return next;
    },
  };
}

test("a broken output directory fails before any model call, and a dry run creates nothing", async () => {
  const directory = await writeSheet();
  const blocker = path.join(await temp("blocked"), "file");
  await writeFile(blocker, "");
  const outDir = path.join(blocker, "report");
  const options = { sheet: path.join(directory, "labels.csv"), outDir, yes: true, configPath: await writeConfig() };

  const llm = fakeLlm([RUN_1]);
  await assert.rejects(qe.runEvaluate(options, { llm }), /is not a directory/);
  assert.equal(llm.calls.length, 0, "no paid call happens before the directory is proven usable");

  const dry = fakeLlm([RUN_1]);
  await assert.rejects(qe.runEvaluate({ ...options, yes: false }, { llm: dry }), /is not a directory/);
  assert.equal(dry.calls.length, 0);
  assert.deepEqual(await readdir(path.dirname(blocker)), ["file"], "the dry run still creates nothing");
});

test("an existing report is protected unless --force is given", async () => {
  const directory = await writeSheet();
  const outDir = path.join(await temp("report"), "report");
  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "report.md"), "previous report");
  const options = { sheet: path.join(directory, "labels.csv"), outDir, yes: true, configPath: await writeConfig() };

  const llm = fakeLlm([RUN_1]);
  await assert.rejects(qe.runEvaluate(options, { llm }), /report\.md already exists.*--force/s);
  assert.equal(llm.calls.length, 0, "the run is refused before any model call");
  assert.equal(await readFile(path.join(outDir, "report.md"), "utf8"), "previous report");

  const forced = fakeLlm([RUN_1]);
  const result = await qe.runEvaluate({ ...options, force: true }, { llm: forced });
  assert.equal(forced.calls.length, 1);
  assert.notEqual(await readFile(path.join(outDir, "report.md"), "utf8"), "previous report");
  assert.deepEqual(result.files.map(file => path.basename(file)).sort(), ["report.json", "report.md"]);
});

test("the output directory is created before the model runs and no probe file remains", async () => {
  const directory = await writeSheet();
  const outDir = path.join(await temp("report"), "nested", "report");
  const llm = fakeLlm([RUN_1]);
  await qe.runEvaluate({ sheet: path.join(directory, "labels.csv"), outDir, yes: true, configPath: await writeConfig() }, { llm });

  assert.equal(llm.calls.length, 1);
  assert.deepEqual((await readdir(outDir)).sort(), ["report.json", "report.md"]);
});

test("--force is accepted by the evaluate arguments", () => {
  assert.deepEqual(qe.parseEvaluateArgs(["--sheet", "/s", "--out", "/o", "--force"]), { sheet: "/s", outDir: "/o", force: true });
  assert.deepEqual(qe.parseEvaluateArgs(["--sheet", "/s", "--out", "/o", "--recorded", "--force"]), { sheet: "/s", outDir: "/o", recorded: true, force: true });
});

/** One sheet row with every column, overridable per test. */
const SHEET_ROW = {
  date: "2026-10-14", id: "2610.10230", title: "Title", abstract: "Abstract", had_abstract_at_filter: "true",
  model_category: "skip", model_directions: "", model_score: "0", model_outcome: "skipped_by_model",
  label: "n", label_direction: "", note: "",
};

function oneRow(overrides = {}) {
  const row = { ...SHEET_ROW, ...overrides };
  return qe.encodeCsv([[...qe.SHEET_COLUMNS], qe.SHEET_COLUMNS.map(column => row[column] ?? "")]);
}

test("an arXiv id whose serial a spreadsheet shortened is rejected with its row number", () => {
  assert.throws(
    () => qe.parseSheet(oneRow({ id: "2610.1023" }), new Set()),
    error => error.name === "SheetError" && /row 2: .*id/.test(error.message),
  );
});

test("the serial width follows the year: four digits before 2015, five from then on", () => {
  for (const id of ["2610.1023", "2610.102300", "1501.1234", "1412.12345", "2613.10230"]) {
    assert.throws(() => qe.parseSheet(oneRow({ id }), new Set()), /row 2: .*id/, id);
  }
  const valid = ["0704.0001", "1412.1234", "1501.01234", "2610.10230", "2610.10230v2"];
  const rows = valid.map((id, index) => ({ date: ["2007-04-01", "2014-12-01", "2015-01-01", "2026-10-14", "2026-10-14"][index], id }));
  const text = qe.encodeCsv([[...qe.SHEET_COLUMNS], ...rows.map(row => {
    const filled = { ...SHEET_ROW, ...row };
    return qe.SHEET_COLUMNS.map(column => filled[column] ?? "");
  })]);
  assert.deepEqual(qe.parseSheet(text, new Set()).map(row => row.id), valid);
});

test("dates that are not real YYYY-MM-DD calendar days are rejected with their row number", () => {
  for (const date of ["2026/10/14", "2026-13-01", "2026-02-30", "14-10-2026", ""]) {
    assert.throws(() => qe.parseSheet(oneRow({ date }), new Set()), /row 2: .*date/, date);
  }
});

