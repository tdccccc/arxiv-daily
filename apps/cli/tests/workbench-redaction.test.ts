// @vitest-environment happy-dom
import { afterEach, expect, it, vi } from "vitest";
import { request as httpRequest } from "node:http";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PaperIndexStore, createStorageStateStore } from "@arxiv-daily/core";
import { NodeStorageAdapter } from "@arxiv-daily/node-runtime";
import { loadCliConfig } from "../src/config";
import { startWorkbench, type WorkbenchOptions } from "../src/workbench/server";
import { readWorkbenchSettings, saveWorkbenchSettings } from "../src/workbench/settings";
import { bindSettings, settingsForm } from "../src/workbench/web/settings";

const cleanup: Array<() => Promise<unknown>> = [];
afterEach(async () => { for (const close of cleanup.splice(0).reverse()) await close(); });

const DIRECTION = "photometric redshift estimation in the local universe";

async function setup(apiKey: string, run?: WorkbenchOptions["run"]) {
  const root = await mkdtemp(join(tmpdir(), "arxiv-redaction-"));
  cleanup.push(() => rm(root, { recursive: true, force: true }));
  const configPath = join(root, "config.toml");
  const initial = await readWorkbenchSettings(configPath);
  const config = await saveWorkbenchSettings(configPath, { revision: null, values: {
    ...initial.values, vaultRoot: join(root, "vault"), apiKey, model: "fixture-model", baseUrl: "https://fixture.invalid/v1", categories: ["cs.AI"],
    topics: [{ id: "t", name: "Cosmology", tag: "cosmo", detail: false, description: DIRECTION, directions: [{ id: "d", text: DIRECTION, origin: "manual" }] }],
  } });
  const app = await startWorkbench({ config, run }); cleanup.push(app.close);
  // Node's http rather than fetch: the happy-dom environment replaces fetch.
  const send = (route: string, body?: unknown) => new Promise<{ status: number; text: string }>((resolve, reject) => {
    const payload = body === undefined ? undefined : JSON.stringify(body);
    const req = httpRequest(new URL(route, app.url), { method: payload === undefined ? "GET" : "POST", headers: payload === undefined ? {} : { "Content-Type": "application/json" } }, res => {
      let text = ""; res.setEncoding("utf8"); res.on("data", chunk => { text += chunk; }); res.on("end", () => resolve({ status: res.statusCode ?? 0, text }));
    });
    req.on("error", reject); req.end(payload);
  });
  const request = async <T>(route: string, body?: unknown): Promise<T> => {
    const { status, text } = await send(route, body);
    if (status !== 200) throw new Error(JSON.parse(text).error);
    return JSON.parse(text) as T;
  };
  return { root, configPath, config, send, request };
}

// Local servers need a non-empty key but ignore it, so the key is often an ordinary word.
it("serves documents, papers, settings and status as stored when the model key is an ordinary word", async () => {
  const { config, send } = await setup("local");
  const dailyDir = join(config.vaultRoot, config.settings.output.dailyDir);
  await mkdir(dailyDir, { recursive: true });
  await writeFile(join(dailyDir, "2026-10-01.md"), "# A local approach to calibration\n\nLocal methods work locally on non-local data.\n");
  await new PaperIndexStore(new NodeStorageAdapter(config.vaultRoot), config.settings.output)
    .upsertFromDailyPaper({ arxivId: "2610.00001", title: "A local approach to calibration", authors: "An Author", date: "2026-10-01", arxivCategory: "cs.AI", primaryTopic: "cosmo", detail: false });

  const documents = await send("api/documents");
  const entry = JSON.parse(documents.text).documents[0];
  expect(entry.title).toBe("A local approach to calibration");
  const document = await send(`api/document?path=${encodeURIComponent(entry.path)}`);
  expect(JSON.parse(document.text).html).toContain("Local methods work locally on non-local data.");
  expect(JSON.parse(document.text).title).toBe("A local approach to calibration");
  const papers = await send("api/papers");
  expect(JSON.parse(papers.text).papers[0].title).toBe("A local approach to calibration");
  const settings = await send("api/settings");
  expect(JSON.parse(settings.text).values.topics[0].directions[0].text).toBe(DIRECTION);
  expect(JSON.parse(settings.text).values.embedding.mode).toBe("local");
  const status = await send("api/status");
  expect(JSON.parse(status.text).topics[0].description).toBe(DIRECTION);
  for (const response of [documents, document, papers, settings, status]) expect(response.text).not.toContain("[REDACTED]");
});

it("keeps research directions intact when an unrelated setting is saved and the model key is an ordinary word", async () => {
  const { configPath, request } = await setup("local");
  const snapshot = await request<Awaited<ReturnType<typeof readWorkbenchSettings>>>("api/settings");
  document.body.innerHTML = settingsForm(snapshot).outerHTML;
  const form = document.querySelector("form")!;
  const binding = bindSettings(form, snapshot, request, async () => {});
  try {
    expect((form.querySelector('[name="topicDirection"]') as HTMLTextAreaElement).value).toBe(DIRECTION);
    const limit = form.elements.namedItem("maxDailyPapers") as HTMLInputElement;
    limit.value = "25"; limit.dispatchEvent(new Event("change", { bubbles: true }));
    await vi.waitFor(async () => expect((await loadCliConfig({ configPath })).settings.output.maxDailyPapers).toBe(25));
    const saved = await loadCliConfig({ configPath });
    expect(saved.settings.arxiv.topics[0]).toMatchObject({ description: DIRECTION, directions: [{ id: "d", text: DIRECTION }] });
    expect(await readFile(configPath, "utf8")).not.toContain("[REDACTED]");
  } finally { binding.dispose(); document.body.innerHTML = ""; }
});

it("still scrubs a real key from persisted run errors and from run output", async () => {
  const key = "secret-model-key";
  const { config, send } = await setup(key, async (_args, io) => {
    // Written in two chunks: only the finished output holds the whole key.
    io.stdout.write("provider echoed secret-");
    io.stdout.write("model-key and stopped\n");
    return 0;
  });
  const state = createStorageStateStore(new NodeStorageAdapter(config.vaultRoot), config.settings.output);
  await state.load();
  await state.setFailed("2026-10-02", "transient", `network error ${key}`);

  const status = await send("api/status");
  expect(JSON.parse(status.text).recentRuns[0].error).toBe("network error [REDACTED]");
  const calendar = await send("api/calendar?month=2026-10");
  expect(JSON.parse(calendar.text).cells.find((day: { date: string } | null) => day?.date === "2026-10-02").message).toBe("network error [REDACTED]");

  expect((await send("api/runs", { kind: "daily", date: "2026-10-01" })).status).toBe(202);
  await vi.waitFor(async () => expect(JSON.parse((await send("api/runs/current")).text).run.status).toBe("completed"));
  const current = await send("api/runs/current");
  expect(JSON.parse(current.text).run.output).toBe("provider echoed [REDACTED] and stopped\n");
  for (const response of [status, calendar, current]) expect(response.text).not.toContain(key);
});
