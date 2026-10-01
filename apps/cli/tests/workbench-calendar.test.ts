import { afterEach, expect, it, vi } from "vitest";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DEFAULT_SETTINGS, PaperIndexStore, type RunState, type RunStateEntry } from "@arxiv-daily/core";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { DEFAULT_CLI_SCHEDULE, type CliRuntimeConfig } from "../src/config";
import { startWorkbench, type WorkbenchOptions } from "../src/workbench/server";

const cleanup: Array<() => Promise<unknown>> = [];
afterEach(async () => { for (const close of cleanup.splice(0).reverse()) await close(); });
const state = (status: RunStateEntry["status"], extra: Partial<RunStateEntry> = {}): RunStateEntry => ({ status, lastAttempt: 1, attempts: 1, ...extra });

async function setup(options: { timezone?: string; now?: string; records?: RunState; reports?: string[]; run?: WorkbenchOptions["run"] } = {}) {
  const root = await mkdtemp(join(tmpdir(), "arxiv-calendar-"));
  cleanup.push(() => rm(root, { recursive: true, force: true }));
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.arxiv.timezone = options.timezone || "Asia/Shanghai";
  settings.output.dailyDir = "research/daily";
  settings.output.papersDir = "research/papers";
  settings.llm.apiKey = "secret-calendar-key";
  const config: CliRuntimeConfig = { settings, vaultRoot: join(root, "vault"), cacheDir: join(root, "cache"), configPath: join(root, "config.toml"), linkStyle: "relative", scheduleIntent: { ...DEFAULT_CLI_SCHEDULE } };
  for (const directory of ["research/daily", "research/papers", "research/.index"]) await mkdir(join(config.vaultRoot, directory), { recursive: true });
  for (const date of options.reports ?? []) await writeFile(join(config.vaultRoot, `research/daily/${date}.md`), `---\ndate: ${date}\n---\n# Report ${date}\n`);
  await writeFile(join(config.vaultRoot, "research/.index/run-state.json"), JSON.stringify({ schemaVersion: 1, runState: options.records ?? {} }));
  await writeFile(config.configPath, "secret-calendar-key");
  const run = options.run ?? vi.fn(async () => 0);
  const app = await startWorkbench({ config, run, now: () => new Date(options.now ?? "2026-10-01T00:30:00.000Z") });
  cleanup.push(app.close);
  return { ...app, config, root, run, get: (path: string, init?: RequestInit) => fetch(new URL(path, app.url), init) };
}

it("serves a Monday-first leap-month calendar over real HTTP using product timezone", async () => {
  const { get, run } = await setup({ timezone: "America/Los_Angeles" });
  const response = await get("api/calendar?month=2024-02");
  expect(response.status).toBe(200);
  const calendar = await response.json();
  expect(calendar).toMatchObject({ month: "2024-02", today: "2026-09-30", timezone: "America/Los_Angeles", previousMonth: "2024-01", nextMonth: "2024-03" });
  expect(calendar.cells).toHaveLength(35);
  expect(calendar.cells.slice(0, 3)).toEqual([null, null, null]);
  expect(calendar.cells[3].date).toBe("2024-02-01");
  expect(calendar.cells[31].date).toBe("2024-02-29");
  expect(calendar.cells.slice(32)).toEqual([null, null, null]);
  expect((await (await get("api/calendar")).json()).month).toBe("2026-09");
  const december = await (await get("api/calendar?month=2025-12")).json();
  expect(december.nextMonth).toBe("2026-01");
  expect(run).not.toHaveBeenCalled();
});

it("recovers missing report counts from the existing Paper Index without replacing known totals", async () => {
  const { get, config } = await setup({ reports: ["2026-09-01", "2026-09-02", "2026-09-03"], records: { "2026-09-02": state("completed", { papersWritten: 8 }) } });
  const storage = new NodeStorageAdapter(config.vaultRoot);
  const index = new PaperIndexStore(storage, config.settings.output);
  for (const [id, date] of [["2609.10001", "2026-09-01"], ["2609.10002", "2026-09-01"], ["2609.10001", "2026-09-02"]]) {
    await index.upsertFromDailyPaper({ arxivId: id!, title: id!, authors: "An author", date: date!, arxivCategory: "cs.AI", primaryTopic: "inference", detail: false, dailyReport: `research/daily/${date}.md` });
  }
  const indexFile = join(config.vaultRoot, "research/.index/papers.json");
  const before = await readFile(indexFile, "utf8");
  const data = await (await get("api/calendar?month=2026-09")).json();
  const day = (date: string) => data.cells.find((cell: { date: string } | null) => cell?.date === date);
  expect(day("2026-09-01").papers).toBe(2);
  expect(day("2026-09-02").papers).toBe(8);
  expect(day("2026-09-03").papers).toBeNull();
  expect(await readFile(indexFile, "utf8")).toBe(before);
});

it("combines real reports and full historical run state without modifying records or exposing secrets", async () => {
  const records: RunState = {
    "2025-01-02": state("completed", { papersWritten: 4 }),
    "2025-01-03": state("completed", { papersWritten: 0 }),
    "2025-01-04": state("completed", { papersWritten: 3 }),
    "2025-01-05": state("running"),
    "2025-01-06": state("failed_transient", { error: "provider secret-calendar-key timed out" }),
    "2025-01-07": state("failed_permanent", { error: "retry limit reached" }),
    "2025-01-08": state("skipped", { error: "upstream recorded skip reason" }),
    "2025-01-09": state("pending", { error: "waiting for announcement" }),
  };
  for (let day = 1; day <= 30; day += 1) records[`2026-09-${String(day).padStart(2, "0")}`] = state("completed", { papersWritten: 0 });
  const { get, config, root, run } = await setup({ records, reports: ["2025-01-01", "2025-01-02"] });
  const files = (await readdir(root, { recursive: true, withFileTypes: true })).filter(entry => entry.isFile()).map(entry => join(entry.parentPath, entry.name));
  const before = await Promise.all(files.map(file => readFile(file, "utf8")));
  const response = await get("api/calendar?month=2025-01");
  expect(response.status).toBe(200);
  const text = await response.text();
  expect(text).not.toContain("secret-calendar-key");
  const calendar = JSON.parse(text);
  const day = (date: string) => calendar.cells.find((cell: { date: string } | null) => cell?.date === date);
  expect(day("2025-01-01")).toMatchObject({ state: "has-report", reportPath: "research/daily/2025-01-01.md", reportTitle: "Report 2025-01-01", papers: null, canGenerate: false });
  expect(day("2025-01-02")).toMatchObject({ state: "has-report", papers: 4 });
  expect(day("2025-01-03")).toMatchObject({ state: "no-matches", papers: 0, reportPath: null, canGenerate: false, actionLabel: null });
  expect(day("2025-01-04")).toMatchObject({ state: "report-missing", papers: 3, canGenerate: false });
  expect(day("2025-01-05")).toMatchObject({ state: "running", canGenerate: false });
  expect(day("2025-01-06")).toMatchObject({ state: "failed", canGenerate: true, actionLabel: "重试生成" });
  expect(day("2025-01-07")).toMatchObject({ state: "failed", canGenerate: false, message: "retry limit reached" });
  expect(day("2025-01-08")).toMatchObject({ state: "skipped", canGenerate: false, message: "upstream recorded skip reason" });
  expect(day("2025-01-09")).toMatchObject({ state: "not-generated", canGenerate: true, actionLabel: "生成日报" });
  expect(day("2025-01-10")).toMatchObject({ state: "not-generated", canGenerate: true, papers: null });
  expect(await Promise.all(files.map(file => readFile(file, "utf8")))).toEqual(before);
  expect(await readdir(join(config.vaultRoot, "research/.index"))).toEqual(["run-state.json"]);
  expect(run).not.toHaveBeenCalled();
});

it("keeps future dates unavailable but permits reading an existing future-dated report", async () => {
  const { get } = await setup({ reports: ["2026-10-05"] });
  const response = await get("api/calendar?month=2026-10");
  expect(response.status).toBe(200);
  const calendar = await response.json();
  expect(calendar.cells.find((cell: { date: string } | null) => cell?.date === "2026-10-01")).toMatchObject({ state: "not-generated", canGenerate: true });
  expect(calendar.cells.find((cell: { date: string } | null) => cell?.date === "2026-10-02")).toMatchObject({ state: "future", canGenerate: false });
  expect(calendar.cells.find((cell: { date: string } | null) => cell?.date === "2026-10-05")).toMatchObject({ state: "has-report", canGenerate: false });
});

it("rejects malformed months at the calendar HTTP boundary", async () => {
  const { get } = await setup();
  for (const month of ["", "2026-1", "2026-00", "2026-13", "2026-01-01", "../../private", "invalid"]) expect((await get(`api/calendar?month=${encodeURIComponent(month)}`)).status, month).toBe(400);
});

it("overlays an owned daily job before state persistence with a timezone-correct date", async () => {
  let finish!: (code: number) => void;
  const run = vi.fn<NonNullable<WorkbenchOptions["run"]>>((_args, _io, signal) => new Promise(resolve => { finish = resolve; signal.addEventListener("abort", () => resolve(1), { once: true }); }));
  const { get } = await setup({ run, timezone: "America/Los_Angeles" });
  const posted = await get("api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind: "daily" }) });
  expect(posted.status).toBe(202);
  const owned = (await posted.json()).run;
  expect(owned.date).toBe("2026-09-30");
  expect(run.mock.calls[0]?.[0]).toEqual(["run", "--today"]);
  const response = await get("api/calendar");
  expect(response.status).toBe(200);
  const calendar = await response.json();
  expect(calendar.cells.find((cell: { date: string } | null) => cell?.date === "2026-09-30")).toMatchObject({ state: "running", canGenerate: false });
  finish(0);
  await vi.waitFor(async () => expect((await (await get("api/runs/current")).json()).run.status).toBe("completed"));
  const paper = await get("api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind: "paper", id: "1706.03762" }) });
  expect((await paper.json()).run.date).toBeNull();
  finish(0);
});
