import { afterEach, describe, expect, it } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { parse, stringify } from "smol-toml";
import { connectCliLibrary, authorizeCliLibrary, inspectCliLibraryConnection } from "../src/library-connection-cmd";
import { readWorkbenchSettings, saveWorkbenchSettings } from "../src/workbench/settings";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => fs.rm(root, { recursive: true, force: true }))); });
async function fixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "workbench-settings-")); roots.push(root);
  const configPath = path.join(root, "config", "config.toml");
  const initial = await readWorkbenchSettings(configPath);
  const values = { ...initial.values, vaultRoot: path.join(root, "vault"), baseUrl: "https://api.deepseek.com/v1", provider: "deepseek", model: "test-model", apiKey: "private-test-key", categories: ["cs.AI"], timezone: "Asia/Shanghai", topics: [{ id: "focus", name: "Agents", tag: "agents", description: "Research agents", detail: false }] };
  return { configPath, values };
}
describe("workbench settings", () => {
  it("starts without config and creates private usable config without exposing secrets", async () => {
    const { configPath, values } = await fixture();
    expect(await readWorkbenchSettings(configPath)).toMatchObject({ setupRequired: true, revision: null, values: { vaultRoot: "", apiKeyConfigured: false } });
    const saved = await saveWorkbenchSettings(configPath, { revision: null, values });
    expect(saved.settings.llm.apiKey).toBe("private-test-key");
    expect(saved.settings.arxiv.topics).toEqual(values.topics);
    const view = await readWorkbenchSettings(configPath);
    expect(view).toMatchObject({ setupRequired: false, revision: saved.configRevision, values: { apiKeyConfigured: true } });
    expect(JSON.stringify(view)).not.toContain("private-test-key");
    if (process.platform !== "win32") expect((await fs.stat(configPath)).mode & 0o777).toBe(0o600);
  });
  it("preserves unknown TOML fields, library and secrets while updating settings", async () => {
    const { configPath, values } = await fixture();
    await saveWorkbenchSettings(configPath, { revision: null, values });
    const doc = parse(await fs.readFile(configPath, "utf8"));
    Object.assign(doc, { custom: { nested: "keep" }, library: { custom: "untouched" }, email: { api_key: "mail-private" }, embedding: { api_key: "embed-private" } });
    await fs.writeFile(configPath, stringify(doc));
    const view = await readWorkbenchSettings(configPath);
    await saveWorkbenchSettings(configPath, { revision: view.revision, values: { ...view.values, model: "new-model", apiKey: "" } });
    const after = parse(await fs.readFile(configPath, "utf8"));
    expect(after).toMatchObject({ custom: doc.custom, library: doc.library, email: doc.email, embedding: doc.embedding, llm: { api_key: "private-test-key", model: "new-model" } });
  });
  it("rejects stale updates and concurrent initial creation", async () => {
    const { configPath, values } = await fixture();
    const results = await Promise.allSettled([saveWorkbenchSettings(configPath, { revision: null, values }), saveWorkbenchSettings(configPath, { revision: null, values })]);
    expect(results.filter(r => r.status === "fulfilled")).toHaveLength(1);
    expect(results.find(r => r.status === "rejected")).toMatchObject({ reason: { status: 409 } });
    const view = await readWorkbenchSettings(configPath);
    await fs.appendFile(configPath, "\n# external edit\n");
    const before = await fs.readFile(configPath, "utf8");
    await expect(saveWorkbenchSettings(configPath, { revision: view.revision, values })).rejects.toMatchObject({ status: 409 });
    expect(await fs.readFile(configPath, "utf8")).toBe(before);
  });
  it.each([{ vaultRoot: "relative/path" }, { baseUrl: "file:///tmp/model" }, { baseUrl: "https://user:secret@model.test" }, { timezone: "invalid/zone" }, { dailyDir: "../escape" }, { categories: ["cs.AI", "cs.AI"] }])("rejects invalid values without writing: %j", async patch => {
    const { configPath, values } = await fixture();
    await expect(saveWorkbenchSettings(configPath, { revision: null, values: { ...values, ...patch } })).rejects.toMatchObject({ status: 400 });
    await expect(fs.stat(configPath)).rejects.toMatchObject({ code: "ENOENT" });
  });
  it("preserves an omitted key, replaces a supplied key and hides credentials in old endpoints", async () => {
    const { configPath, values } = await fixture();
    await saveWorkbenchSettings(configPath, { revision: null, values });
    let view = await readWorkbenchSettings(configPath);
    await saveWorkbenchSettings(configPath, { revision: view.revision, values: view.values });
    view = await readWorkbenchSettings(configPath);
    const saved = await saveWorkbenchSettings(configPath, { revision: view.revision, values: { ...view.values, apiKey: "replacement-key" } });
    expect(saved.settings.llm.apiKey).toBe("replacement-key");
    const doc = parse(await fs.readFile(configPath, "utf8"));
    (doc.llm as Record<string, unknown>).base_url = "https://user:private-pass@model.example/v1?token=private-token#private-fragment";
    await fs.writeFile(configPath, stringify(doc));
    const projected = JSON.stringify(await readWorkbenchSettings(configPath));
    for (const secret of ["replacement-key", "private-pass", "private-token", "private-fragment"]) expect(projected).not.toContain(secret);
  });
  it("updates the real config behind a symlink without replacing the link", async () => {
    const { configPath, values } = await fixture();
    await saveWorkbenchSettings(configPath, { revision: null, values });
    const link = path.join(path.dirname(configPath), "alias.toml");
    await fs.symlink(configPath, link);
    const view = await readWorkbenchSettings(link);
    await saveWorkbenchSettings(link, { revision: view.revision, values: { ...view.values, model: "linked-model" } });
    expect((await fs.lstat(link)).isSymbolicLink()).toBe(true);
    expect((await readWorkbenchSettings(configPath)).values.model).toBe("linked-model");
  });
  it("round trips Obsidian settings while keeping CLI cron independent", async () => {
    const { configPath, values } = await fixture();
    const saved = await saveWorkbenchSettings(configPath, { revision: null, values: { ...values,
      reasoningEffort: "none", detailProfile: "broad", linkStyle: "relative",
      schedule: { enabled: true, runAtLocal: "08:15", runUntilLocal: "17:45", tickIntervalMin: 7 },
      embedding: { mode: "remote", baseUrl: "https://embedding.example/v1", model: "embed", dimension: 512, apiKey: "embedding-private" },
      pdfParserSidecar: { enabled: true, capabilitiesUrl: "http://127.0.0.1:5555/cap", parseUrl: "http://127.0.0.1:5555/parse" },
      email: { enabled: true, mode: "self", to: "reader@example.com", fromEmail: "sender@example.com", fromName: "Reader", apiKey: "email-private", hostedToken: "hosted-private" }, logLevel: "warn"
    } });
    expect(saved.settings.llm.thinkingMode).toBe(false);
    expect(saved.settings.detailSelection).toMatchObject({ profile: "broad", softLimit: 5 });
    expect(saved.settings.output.linkStyle).toBe("relative");
    expect(saved.settings.schedule.enabled).toBe(false);
    expect(saved).toMatchObject({ workbenchSchedule: { enabled: true, tickIntervalMin: 7 } });
    expect(saved.settings.pdfParserSidecar.enabled).toBe(true);
    const view = await readWorkbenchSettings(configPath);
    expect(view.values).toMatchObject({ reasoningEffort: "none", detailProfile: "broad", linkStyle: "relative", schedule: { enabled: true },
      embedding: { mode: "remote", model: "embed", dimension: 512, apiKeyConfigured: true }, email: { apiKeyConfigured: true, hostedTokenConfigured: true }, logLevel: "warn" });
    for (const secret of ["embedding-private", "email-private", "hosted-private"]) expect(JSON.stringify(view)).not.toContain(secret);
    const again = await saveWorkbenchSettings(configPath, { revision: view.revision, values: view.values });
    expect(again.settings.embedding.apiKey).toBe("embedding-private");
    expect(again.settings.email.apiKey).toBe("email-private");
    expect(again.settings.email.hostedToken).toBe("hosted-private");
  });
  it("saves incomplete setup for incremental editing", async () => {
    const { configPath, values } = await fixture();
    await expect(saveWorkbenchSettings(configPath, { revision: null, values: { ...values, apiKey: "", model: "", baseUrl: "", categories: [], topics: [] } })).resolves.toMatchObject({ settings: { llm: { apiKey: "" }, arxiv: { topics: [] } } });
  });
  it.each([
    { reasoningEffort: "invalid" }, { detailProfile: "invalid" }, { linkStyle: "invalid" }, { logLevel: "invalid" },
    { schedule: { enabled: true, runAtLocal: "25:00", runUntilLocal: "18:00", tickIntervalMin: 20 } },
    { schedule: { enabled: true, runAtLocal: "18:00", runUntilLocal: "09:00", tickIntervalMin: 20 } },
    { pdfParserSidecar: { enabled: true, capabilitiesUrl: "https://external.example/cap", parseUrl: "https://external.example/parse" } },
    { pdfParserSidecar: { enabled: true, capabilitiesUrl: "http://127.0.0.1:5/cap", parseUrl: "http://127.0.0.1:6/parse" } },
  ])("rejects invalid extended settings %j", async patch => {
    const { configPath, values } = await fixture();
    await expect(saveWorkbenchSettings(configPath, { revision: null, values: { ...values, ...patch } })).rejects.toMatchObject({ status: 400 });
  });
  it("preserves custom thresholds and cron, and requires new consent after endpoint expansion", async () => {
    const { configPath, values } = await fixture();
    let config = await saveWorkbenchSettings(configPath, { revision: null, values });
    const connected = await connectCliLibrary(config, path.dirname(configPath));
    config = await authorizeCliLibrary(connected, inspectCliLibraryConnection(connected).disclosure!.authorizationFingerprint);
    expect(inspectCliLibraryConnection(config).status.kind).toBe("authorized");
    const doc = parse(await fs.readFile(configPath, "utf8"));
    const schedule = { enabled: true, on: "10:30", until: "18:30", interval_hours: 4, weekdays_only: false };
    const detail = { profile: "custom", normal_threshold: 81, exceptional_threshold: 97, soft_limit: 2 };
    Object.assign(doc, { schedule, detail_selection: detail });
    await fs.writeFile(configPath, stringify(doc));
    const view = await readWorkbenchSettings(configPath);
    const saved = await saveWorkbenchSettings(configPath, { revision: view.revision, values: { ...view.values, embedding: { ...view.values.embedding, mode: "remote", baseUrl: "https://remote.example/v1" } } });
    expect(saved.settings.detailSelection).toEqual({ profile: "custom", normalThreshold: 81, exceptionalThreshold: 97, softLimit: 2 });
    expect(inspectCliLibraryConnection(saved).status.kind).toBe("authorization-invalidated");
    const after = parse(await fs.readFile(configPath, "utf8"));
    expect(after.schedule).toEqual(schedule);
    expect(after.library).toEqual(doc.library);
  });
  it("does not mistake malformed existing config for first use", async () => {
    const { configPath, values } = await fixture();
    await fs.mkdir(path.dirname(configPath), { recursive: true });
    await fs.writeFile(configPath, "broken = [");
    await expect(readWorkbenchSettings(configPath)).rejects.toMatchObject({ status: 400 });
    await expect(saveWorkbenchSettings(configPath, { revision: null, values })).rejects.toMatchObject({ status: 409 });
    expect(await fs.readFile(configPath, "utf8")).toBe("broken = [");
  });
});
