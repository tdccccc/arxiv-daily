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
  { ref: "photo-z#1", tag: "photo-z", id: "dir-photoz", text: "Photometric redshift methods, catalogs." },
  { ref: "photo-z#2", tag: "photo-z", id: "dir-photoz-2", text: "Calibration with spectroscopic samples." },
  { ref: "cluster#1", tag: "cluster", id: "dir-cluster", text: "Cluster surveys, mass calibration." },
];
const GENERATION = {
  requestContractVersion: 2, provider: "deepseek", model: "model-a",
  endpointDigest: `sha256:${"a".repeat(64)}`, mode: { kind: "temperature", temperature: 0 },
};
const idOf = (day, index) => `26${day}.${String(index).padStart(5, "0")}`;

/** One candidate; `kind` picks the model decision. */
function candidate(id, kind, overrides = {}) {
  const base = {
    id, title: `Title of ${id}`, hadAbstract: false,
    category: "skip", directions: [], relevanceScore: 0, outcome: "skipped_by_model",
  };
  const selected = {
    photoz: { category: "photo-z", directions: [{ id: "dir-photoz", text: DIRECTIONS[0].text, tag: "photo-z" }], relevanceScore: 90, outcome: "in_report" },
    cluster: { category: "cluster", directions: [{ id: "dir-cluster", text: DIRECTIONS[2].text, tag: "cluster" }], relevanceScore: 60, outcome: "cut_by_limit" },
    both: { category: "photo-z", directions: [
      { id: "dir-photoz", text: DIRECTIONS[0].text, tag: "photo-z" },
      { id: "dir-photoz-2", text: DIRECTIONS[1].text, tag: "photo-z" },
    ], relevanceScore: 70, outcome: "excluded_ignored" },
  };
  return { ...base, ...(selected[kind] ?? {}), ...overrides };
}

async function writeRecord(vault, reportDate, candidates, options = {}) {
  const store = new qe.FilterDecisionRecordStore(new qe.NodeStorageAdapter(vault), options.output ?? qe.DEFAULT_SETTINGS.output);
  await store.save({
    reportDate, directions: options.directions ?? DIRECTIONS,
    promptContractVersion: 4, resultContractVersion: 3, generation: GENERATION, candidates,
  });
}

/** A vault with two days: day A has 2 selected and 6 skipped, day B has 1 selected and 4 skipped. */
async function fixtureVault() {
  const vault = await temp("vault");
  await writeRecord(vault, "2026-10-13", [
    candidate(idOf("10", 1), "photoz", { title: 'Quoted "title", with comma', hadAbstract: true }),
    candidate(idOf("10", 2), "both"),
    ...[3, 4, 5, 6, 7, 8].map(index => candidate(idOf("10", index), "skip")),
  ]);
  await writeRecord(vault, "2026-10-14", [
    candidate(idOf("11", 1), "cluster", { title: "中文标题 with\nnewline" }),
    ...[2, 3, 4, 5].map(index => candidate(idOf("11", index), "skip")),
  ]);
  return vault;
}

function atomEntry(id, abstract) {
  return `<entry><id>http://arxiv.org/abs/${id}v1</id><title>Title of ${id}</title><author><name>A. Author</name></author>` +
    `<summary>${abstract}</summary><published>2026-10-13T00:00:00Z</published><updated>2026-10-13T00:00:00Z</updated>` +
    `<arxiv:primary_category term="astro-ph.CO"/><category term="astro-ph.CO"/></entry>`;
}
const atomFeed = entries => `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">${entries.join("")}</feed>`;
// arXiv's parser collapses whitespace in abstracts, so newlines are exercised through titles and the CSV unit test.
const abstractOf = id => `Abstract of ${id}, with "quotes", commas, and Chinese. 星系团。`;

const absPage = text => `<html><body><blockquote class="abstract mathjax">\n<span class="descriptor">Abstract:</span>${text}\n</blockquote></body></html>`;
const absAbstractOf = id => `Abs page abstract of ${id}, with $x_1$ math. 星系团。`;
const API_DOWN = () => ({ status: 400, headers: {}, bodyText: "bad request" });

/**
 * A fake arXiv: the Atom API (`calls`) and the /abs/<id> pages (`absCalls`).
 * `respond` can override an API call and `abs` an abs page (a response, or undefined for the default);
 * by default abs pages do not exist.
 */
function fakeHttp(options = {}) {
  const calls = [];
  const absCalls = [];
  return {
    calls,
    absCalls,
    request: async request => {
      const absId = /^https:\/\/arxiv\.org\/abs\/(.+)$/.exec(request.url)?.[1];
      if (absId) {
        absCalls.push(absId);
        return options.abs?.(absId, absCalls.length) ?? { status: 404, headers: {}, bodyText: "not found" };
      }
      const ids = new URL(request.url).searchParams.get("id_list")?.split(",") ?? [];
      calls.push({ url: request.url, ids });
      const custom = options.respond?.(calls.length, ids, request);
      if (custom) return custom;
      const found = ids.filter(id => !options.missing?.has(id));
      return { status: 200, headers: {}, bodyText: atomFeed(found.map(id => atomEntry(id, abstractOf(id)))) };
    },
  };
}

/** Skips the 3 s spacing between arXiv requests so tests stay fast. */
const instantCoordinator = () => {
  let now = 0;
  return new qe.ArxivRequestCoordinator({
    monotonicNow: () => now, wallNow: () => 1_800_000_000_000 + now,
    sleep: async ms => { now += ms; },
  });
};

async function runExport(vault, outDir, options = {}, http = fakeHttp()) {
  const logger = new qe.Logger("error");
  const result = await qe.runExport({ vaultRoot: vault, outDir, seed: "seed-1", skippedSample: 4, ...options }, { http, logger, requestCoordinator: instantCoordinator() });
  return { result, http };
}

const readCsv = async outDir => qe.parseCsv(await readFile(path.join(outDir, "labels.csv"), "utf8"));

test("CSV writes a BOM and round-trips commas, quotes, newlines and Chinese text", () => {
  const rows = [
    ["id", "title", "abstract"],
    ["1", 'He said "hi", twice', "line one\nline two\r\nline three"],
    ["2", "中文，标题", ""],
    ["3", "", " leading and trailing "],
  ];
  const text = qe.encodeCsv(rows);
  assert.equal(text.charCodeAt(0), 0xfeff);
  assert.deepEqual(qe.parseCsv(text), rows.map(row => row.map(cell => cell.replace(/\r\n/g, "\n"))));
  assert.deepEqual(qe.parseCsv('a,b\n"x,1",y\n'), [["a", "b"], ["x,1", "y"]]);
  assert.deepEqual(qe.parseCsv("a,b\r\n1,2\r\n"), [["a", "b"], ["1", "2"]]);
  assert.throws(() => qe.parseCsv('a,"unterminated\n'), /unterminated|quote/i);
});

test("sampling keeps every selected candidate and a seeded sample of skipped ones", () => {
  const items = [];
  for (const date of ["2026-10-13", "2026-10-14"]) {
    for (let index = 1; index <= 30; index += 1) {
      items.push({ date, candidate: candidate(`2610.${String(index).padStart(5, "0")}`, index % 10 === 0 ? "photoz" : "skip") });
    }
  }
  const selected = items.filter(item => item.candidate.category !== "skip");
  const first = qe.sampleCandidates(items, { skippedSample: 10, seed: "s" });
  const again = qe.sampleCandidates([...items].reverse(), { skippedSample: 10, seed: "s" });
  const other = qe.sampleCandidates(items, { skippedSample: 10, seed: "t" });

  assert.equal(first.length, selected.length + 10);
  assert.equal(first.filter(item => item.candidate.category === "skip").length, 10);
  const keptKeys = new Set(first.map(item => `${item.date} ${item.candidate.id}`));
  for (const item of selected) assert.ok(keptKeys.has(`${item.date} ${item.candidate.id}`), "every selected candidate is kept");
  assert.deepEqual(first.map(item => [item.date, item.candidate.id]), again.map(item => [item.date, item.candidate.id]), "same seed, same rows regardless of input order");
  assert.notDeepEqual(first.map(item => item.candidate.id), other.map(item => item.candidate.id), "a different seed picks other skipped rows");
  const keys = first.map(item => `${item.date} ${item.candidate.id}`);
  assert.deepEqual(keys, [...keys].sort(), "rows are ordered by date then id");
  assert.equal(qe.sampleCandidates(items, { skippedSample: 1000, seed: "s" }).length, items.length);
  assert.equal(qe.sampleCandidates(items, { skippedSample: 0, seed: "s" }).length, selected.length);
});

test("export writes the label sheet, guide and metadata from a vault's decision records", async () => {
  const vault = await fixtureVault();
  const outDir = path.join(await temp("out"), "sheet");
  const { result, http } = await runExport(vault, outDir);

  assert.deepEqual((await readdir(outDir)).filter(name => name.startsWith("labels")).sort(), ["labels.csv", "labels.guide.md", "labels.meta.json"]);
  const raw = await readFile(path.join(outDir, "labels.csv"), "utf8");
  assert.equal(raw.charCodeAt(0), 0xfeff);
  const [header, ...rows] = await readCsv(outDir);
  assert.deepEqual(header, ["date", "id", "title", "abstract", "had_abstract_at_filter", "model_category", "model_directions", "model_score", "model_outcome", "label", "label_direction", "note"]);
  // 3 selected + 4 sampled skipped.
  assert.equal(rows.length, 7);
  assert.equal(result.rows, 7);
  assert.ok(rows.every(row => row.length === header.length));
  assert.deepEqual(rows.map(row => [row[0], row[1]]), [...rows.map(row => [row[0], row[1]])].sort((a, b) => (a.join(" ") < b.join(" ") ? -1 : 1)));

  const byId = Object.fromEntries(rows.map(row => [row[1], row]));
  const photoz = byId[idOf("10", 1)];
  assert.equal(photoz[0], "2026-10-13");
  assert.equal(photoz[2], 'Quoted "title", with comma');
  assert.equal(photoz[3], abstractOf(idOf("10", 1)), "abstracts keep quotes, commas and Chinese text");
  assert.deepEqual(photoz.slice(4, 9), ["true", "photo-z", "1", "90", "in_report"]);
  assert.deepEqual(photoz.slice(9), ["", "", ""], "label columns start empty");
  assert.deepEqual(byId[idOf("10", 2)].slice(4, 9), ["false", "photo-z", "1;2", "70", "excluded_ignored"]);
  assert.deepEqual(byId[idOf("11", 1)].slice(5, 9), ["cluster", "3", "60", "cut_by_limit"]);
  assert.equal(byId[idOf("11", 1)][2], "中文标题 with\nnewline");
  const skippedRow = rows.find(row => row[5] === "skip");
  assert.deepEqual([skippedRow[6], skippedRow[7], skippedRow[8]], ["", "0", "skipped_by_model"]);

  // Only the sampled rows are looked up, in one batch, and nothing else is contacted.
  assert.equal(http.calls.length, 1);
  assert.deepEqual([...http.calls[0].ids].sort(), rows.map(row => row[1]).sort());
  assert.match(http.calls[0].url, /^https:\/\/export\.arxiv\.org\/api\/query\?id_list=/);

  const guide = await readFile(path.join(outDir, "labels.guide.md"), "utf8");
  assert.match(guide, /标注说明/);
  assert.match(guide, /\| 1 \| photo-z \| Photometric redshift methods, catalogs\. \|/);
  assert.match(guide, /\| 2 \| photo-z \| Calibration with spectroscopic samples\. \|/);
  assert.match(guide, /\| 3 \| cluster \| Cluster surveys, mass calibration\. \|/);
  assert.match(guide, /label/);
  assert.match(guide, /y.*n|y\/n/);

  const meta = JSON.parse(await readFile(path.join(outDir, "labels.meta.json"), "utf8"));
  assert.equal(meta.seed, "seed-1");
  assert.deepEqual(meta.sourceDates, ["2026-10-13", "2026-10-14"]);
  assert.deepEqual(meta.directions.map(direction => [direction.number, direction.tag, direction.id, direction.text]), [
    [1, "photo-z", "dir-photoz", DIRECTIONS[0].text], [2, "photo-z", "dir-photoz-2", DIRECTIONS[1].text],
    [3, "cluster", "dir-cluster", DIRECTIONS[2].text],
  ]);
  assert.deepEqual(meta.records.map(record => [record.reportDate, record.promptContractVersion, record.resultContractVersion, record.generation.model, record.generation.provider]), [
    ["2026-10-13", 4, 3, "model-a", "deepseek"], ["2026-10-14", 4, 3, "model-a", "deepseek"],
  ]);
  assert.deepEqual(meta.counts, { rows: 7, selected: 3, skippedSampled: 4, skippedAvailable: 10, withoutAbstract: 0, hadAbstractAtFilter: 1 });
  assert.equal(meta.skippedSample, 4);
  assert.ok(!JSON.stringify(meta).includes("https://"), "no endpoint is recorded");
});

test("the same seed exports the same rows and a rerun reuses cached abstracts without refetching", async () => {
  const vault = await fixtureVault();
  const outDir = path.join(await temp("out"), "sheet");
  const first = await runExport(vault, outDir);
  const firstCsv = await readFile(path.join(outDir, "labels.csv"), "utf8");
  assert.equal(first.http.calls.length, 1);

  const second = await runExport(vault, outDir, { force: true });
  assert.equal(second.http.calls.length, 0, "every abstract came from the on-disk cache");
  assert.equal(await readFile(path.join(outDir, "labels.csv"), "utf8"), firstCsv);

  const third = await runExport(vault, outDir, { force: true, seed: "seed-2", skippedSample: 8 });
  assert.ok(third.http.calls.length <= 1);
  assert.ok(third.http.calls.every(call => call.ids.every(id => !first.http.calls[0].ids.includes(id))), "only uncached papers are requested");
  assert.ok((await readCsv(outDir)).length > 8);
});

test("rows whose abstract cannot be fetched are kept empty with a warning", async () => {
  const vault = await fixtureVault();
  const outDir = path.join(await temp("out"), "sheet");
  const missing = new Set([idOf("10", 1), idOf("11", 1)]);
  const { result } = await runExport(vault, outDir, {}, fakeHttp({ missing }));

  const [, ...rows] = await readCsv(outDir);
  for (const row of rows) assert.equal(row[3] === "", missing.has(row[1]));
  assert.equal(result.withoutAbstract, 2);
  assert.ok(result.warnings.some(warning => warning.includes(idOf("10", 1)) && warning.includes(idOf("11", 1))));
  assert.equal(JSON.parse(await readFile(path.join(outDir, "labels.meta.json"), "utf8")).counts.withoutAbstract, 2);

  // A failing endpoint loses abstracts, not the sheet.
  const failedOut = path.join(await temp("out"), "sheet");
  const failed = await runExport(vault, failedOut, {}, fakeHttp({ respond: () => ({ status: 400, headers: {}, bodyText: "bad request" }) }));
  assert.equal(failed.result.rows, 7);
  assert.equal(failed.result.withoutAbstract, 7);
  assert.ok(failed.result.warnings.length > 0);
  assert.ok((await readCsv(failedOut)).slice(1).every(row => row[3] === ""));
});

test("abstracts the API cannot supply come from the arxiv.org/abs page", async () => {
  const vault = await fixtureVault();
  const outDir = path.join(await temp("out"), "sheet");
  const http = fakeHttp({
    respond: API_DOWN,
    abs: id => ({ status: 200, headers: {}, bodyText: absPage(absAbstractOf(id)) }),
  });
  const { result } = await runExport(vault, outDir, {}, http);

  const [, ...rows] = await readCsv(outDir);
  assert.equal(rows.length, 7);
  for (const row of rows) assert.equal(row[3], absAbstractOf(row[1]), "the same extraction the daily pipeline applies, 'Abstract:' label removed");
  assert.deepEqual([...http.absCalls].sort(), rows.map(row => row[1]).sort(), "one abs request per paper");
  assert.equal(result.withoutAbstract, 0);
  assert.ok(result.warnings.some(warning => /abstract fetch failed/.test(warning)), "the API failure is still reported");
  assert.ok(!result.warnings.some(warning => /no abstract could be fetched/.test(warning)));
  assert.equal(JSON.parse(await readFile(path.join(outDir, "labels.meta.json"), "utf8")).counts.withoutAbstract, 0);
});

test("only papers the API missed are requested from the abs page, and a rerun requests nothing", async () => {
  const vault = await fixtureVault();
  const outDir = path.join(await temp("out"), "sheet");
  const missing = new Set([idOf("10", 1), idOf("11", 1)]);
  const absResponder = id => ({ status: 200, headers: {}, bodyText: absPage(absAbstractOf(id)) });
  const first = fakeHttp({ missing, abs: absResponder });
  await runExport(vault, outDir, {}, first);

  assert.deepEqual([...first.absCalls].sort(), [...missing].sort());
  const [, ...rows] = await readCsv(outDir);
  const byId = Object.fromEntries(rows.map(row => [row[1], row[3]]));
  assert.equal(byId[idOf("10", 1)], absAbstractOf(idOf("10", 1)));
  assert.equal(byId[idOf("10", 2)], abstractOf(idOf("10", 2)), "API abstracts are not replaced");

  const second = fakeHttp({ missing, abs: absResponder });
  const { result } = await runExport(vault, outDir, { force: true }, second);
  assert.equal(second.calls.length, 0);
  assert.equal(second.absCalls.length, 0, "abs pages are cached beside the sheet");
  assert.equal(result.withoutAbstract, 0);
});

test("an abs page without an abstract leaves the row empty, and repeated abs failures stop the fallback", async () => {
  const vault = await fixtureVault();
  const noAbstract = fakeHttp({ respond: API_DOWN, abs: () => ({ status: 200, headers: {}, bodyText: "<html><body><p>No abstract here</p></body></html>" }) });
  const out = path.join(await temp("out"), "sheet");
  const { result } = await runExport(vault, out, {}, noAbstract);
  assert.equal(result.withoutAbstract, 7);
  assert.equal(noAbstract.absCalls.length, 7);

  const failing = fakeHttp({ respond: API_DOWN, abs: () => ({ status: 400, headers: {}, bodyText: "bad request" }) });
  const failedOut = path.join(await temp("out"), "sheet");
  const failed = await runExport(vault, failedOut, {}, failing);
  assert.equal(failed.result.rows, 7);
  assert.equal(failed.result.withoutAbstract, 7);
  assert.equal(failing.absCalls.length, 3, "gives up after three consecutive abs failures instead of grinding through every paper");
  assert.ok(failed.result.warnings.some(warning => /abs page/.test(warning) && /stop/i.test(warning)));
});

test("a rate-limited request is retried with backoff", async () => {
  const vault = await fixtureVault();
  const outDir = path.join(await temp("out"), "sheet");
  const { result, http } = await runExport(vault, outDir, {}, fakeHttp({
    respond: call => (call === 1 ? { status: 429, headers: { "retry-after": "0" }, bodyText: "slow down" } : undefined),
  }));

  assert.equal(http.calls.length, 2, "one throttled attempt, then a successful retry");
  assert.equal(result.withoutAbstract, 0);
  assert.ok((await readCsv(outDir)).slice(1).every(row => row[3] !== ""));
});

test("export refuses to overwrite a label sheet unless forced", async () => {
  const vault = await fixtureVault();
  const outDir = path.join(await temp("out"), "sheet");
  await runExport(vault, outDir);
  const filled = (await readFile(path.join(outDir, "labels.csv"), "utf8")).replace(/\r?\n$/, "") + "\r\nresearcher-edit\r\n";
  await writeFile(path.join(outDir, "labels.csv"), filled);

  await assert.rejects(runExport(vault, outDir), /already exists.*--force/s);
  assert.equal(await readFile(path.join(outDir, "labels.csv"), "utf8"), filled, "the researcher's file is untouched");
  await runExport(vault, outDir, { force: true });
  assert.ok(!(await readFile(path.join(outDir, "labels.csv"), "utf8")).includes("researcher-edit"));
});

test("date range limits the records and a missing range is an error", async () => {
  const vault = await fixtureVault();
  const outDir = path.join(await temp("out"), "sheet");
  await runExport(vault, outDir, { from: "2026-10-14", to: "2026-10-14" });
  const [, ...rows] = await readCsv(outDir);
  assert.ok(rows.every(row => row[0] === "2026-10-14"));
  assert.equal(rows.length, 1 + 4);
  assert.deepEqual(JSON.parse(await readFile(path.join(outDir, "labels.meta.json"), "utf8")).sourceDates, ["2026-10-14"]);

  await assert.rejects(runExport(vault, path.join(await temp("out"), "x"), { from: "2027-01-01" }), /no filter decision records/i);
  await assert.rejects(runExport(await temp("empty"), path.join(await temp("out"), "x")), /no filter decision records/i);
});

test("a corrupt record is skipped with a warning while the others are exported", async () => {
  const vault = await fixtureVault();
  await writeFile(path.join(vault, "arxiv-daily/.index/filter-decisions/2026-10-13.json"), "{ not json");
  const outDir = path.join(await temp("out"), "sheet");
  const { result } = await runExport(vault, outDir);

  assert.deepEqual(JSON.parse(await readFile(path.join(outDir, "labels.meta.json"), "utf8")).sourceDates, ["2026-10-14"]);
  assert.ok(result.warnings.some(warning => warning.includes("2026-10-13")));
});

test("legend numbers follow the direction snapshots across days", async () => {
  const vault = await temp("vault");
  const renamed = [{ ...DIRECTIONS[0], text: "Photometric redshifts, rewritten." }, DIRECTIONS[1], DIRECTIONS[2]];
  await writeRecord(vault, "2026-10-13", [candidate(idOf("10", 1), "photoz")]);
  await writeRecord(vault, "2026-10-14", [
    candidate(idOf("11", 1), "photoz", { directions: [{ id: "dir-photoz", text: renamed[0].text, tag: "photo-z" }] }),
  ], { directions: renamed });
  const outDir = path.join(await temp("out"), "sheet");
  await runExport(vault, outDir, { skippedSample: 0 });

  const [, ...rows] = await readCsv(outDir);
  assert.deepEqual(rows.map(row => row[6]), ["1", "4"], "an edited direction text gets its own number");
  const meta = JSON.parse(await readFile(path.join(outDir, "labels.meta.json"), "utf8"));
  assert.deepEqual(meta.directions.map(direction => [direction.number, direction.text, direction.dates]), [
    [1, DIRECTIONS[0].text, ["2026-10-13"]],
    [2, DIRECTIONS[1].text, ["2026-10-13", "2026-10-14"]],
    [3, DIRECTIONS[2].text, ["2026-10-13", "2026-10-14"]],
    [4, renamed[0].text, ["2026-10-14"]],
  ]);
  assert.match(await readFile(path.join(outDir, "labels.guide.md"), "utf8"), /\| 4 \| photo-z \| Photometric redshifts, rewritten\. \|/);
});

test("a config supplies the vault root and a custom output layout", async () => {
  const vault = await temp("vault");
  const output = { ...qe.DEFAULT_SETTINGS.output, dailyDir: "research/daily", papersDir: "research/papers" };
  await writeRecord(vault, "2026-10-14", [candidate(idOf("11", 1), "photoz")], { output });
  const configPath = path.join(await temp("config"), "config.toml");
  await writeFile(configPath, [
    "schema_version = 1", `vault_root = ${JSON.stringify(vault)}`, "",
    "[llm]", 'api_key = "never-printed"', 'base_url = "https://api.example.com/v1"', 'model = "m"', "",
    "[arxiv]", 'categories = ["astro-ph"]', 'timezone = "UTC"', "",
    "[output]", 'daily_dir = "research/daily"', 'papers_dir = "research/papers"', "",
  ].join("\n"));
  const outDir = path.join(await temp("out"), "sheet");

  const { result } = await runExport(undefined, outDir, { configPath, skippedSample: 0 });
  assert.equal(result.rows, 1);
  const raw = await readFile(path.join(outDir, "labels.csv"), "utf8") + await readFile(path.join(outDir, "labels.meta.json"), "utf8");
  assert.ok(!raw.includes("never-printed"));

  // An explicit --vault wins over the config's vault root; the layout still comes from the config.
  const elsewhere = await temp("elsewhere");
  await assert.rejects(runExport(elsewhere, path.join(await temp("out"), "x"), { configPath }), /no filter decision records/i);
});

test("command-line arguments are validated", () => {
  const parsed = qe.parseExportArgs(["--vault", "/v", "--out", "/o", "--skipped", "12", "--seed", "x", "--from", "2026-10-01", "--to", "2026-10-31", "--force"]);
  assert.deepEqual(parsed, { vaultRoot: "/v", outDir: "/o", skippedSample: 12, seed: "x", from: "2026-10-01", to: "2026-10-31", force: true });
  assert.deepEqual(qe.parseExportArgs(["--out", "/o", "--config", "/c.toml"]), { outDir: "/o", configPath: "/c.toml" });
  assert.equal(qe.parseExportArgs(["--help"]).help, true);
  assert.throws(() => qe.parseExportArgs(["--vault", "/v"]), /--out/);
  assert.throws(() => qe.parseExportArgs(["--out", "/o", "--bogus"]), /unknown option.*--bogus/i);
  assert.throws(() => qe.parseExportArgs(["--out", "/o", "--skipped", "-3"]), /--skipped/);
  assert.throws(() => qe.parseExportArgs(["--out", "/o", "--from", "10/01"]), /--from/);
  assert.throws(() => qe.parseExportArgs(["--out"]), /--out/);
});
