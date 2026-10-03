import { afterEach, describe, expect, it } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { parse, stringify } from "smol-toml";
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
  it.each([{ vaultRoot: "relative/path" }, { baseUrl: "file:///tmp/model" }, { baseUrl: "https://user:secret@model.test" }, { timezone: "invalid/zone" }, { topics: [] }, { dailyDir: "../escape" }, { categories: [] }, { categories: ["cs.AI", "cs.AI"] }])("rejects invalid values without writing: %j", async patch => {
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
  it("does not mistake malformed existing config for first use", async () => {
    const { configPath, values } = await fixture();
    await fs.mkdir(path.dirname(configPath), { recursive: true });
    await fs.writeFile(configPath, "broken = [");
    await expect(readWorkbenchSettings(configPath)).rejects.toMatchObject({ status: 400 });
    await expect(saveWorkbenchSettings(configPath, { revision: null, values })).rejects.toMatchObject({ status: 409 });
    expect(await fs.readFile(configPath, "utf8")).toBe("broken = [");
  });
});
