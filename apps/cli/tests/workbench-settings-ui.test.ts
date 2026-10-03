// @vitest-environment happy-dom
import { afterEach, expect, it, vi } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readWorkbenchSettings, saveWorkbenchSettings } from "../src/workbench/settings";
import { mountWorkbench } from "../src/workbench/web/app";
const values = { vaultRoot: "/notes", baseUrl: "https://api.example/v1", provider: "openai", model: "test", apiKeyConfigured: true, categories: ["cs.AI"], timezone: "Asia/Shanghai", summaryLanguage: "zh", topics: [{ id: "existing-topic", name: "Models", tag: "models", description: "Inference", detail: true }], dailyDir: "daily", papersDir: "papers" };
const json = (v: unknown, status = 200) => new Response(JSON.stringify(v), { status });
let dispose = () => {};
afterEach(() => { dispose(); document.body.innerHTML = ""; vi.restoreAllMocks(); });
function setup(first = false, fail = false, configPath?: string) {
  history.replaceState({}, "", "/capability/");
  let configured = !first;
  const fetcher = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
    const path = String(url).split("?")[0];
    if (path === "api/status") return json(configured ? { configPath: "/config.toml", vaultRoot: "/notes", categories: ["cs.AI"], output: { summaryLanguage: "zh" }, llm: { ready: true, provider: "openai", model: "test", keyConfigured: true }, topics: [], emailEnabled: false } : { setupRequired: true });
    if (path === "api/settings") {
      if (init?.method === "POST") {
        if (fail) return json({ error: "配置已被其他窗口修改，请重新加载" }, 409);
        if (configPath) {
          try { await saveWorkbenchSettings(configPath, JSON.parse(String(init.body))); }
          catch (error) { return json({ error: String(error) }, 400); }
        }
        configured = true;
      }
      return json({ setupRequired: !configured, revision: configured ? "revision-1" : null, configPath: "/config.toml", values });
    }
    if (path === "api/preferences") return json({ sidebarWidth: null, sidebarCollapsed: false });
    if (!configured) return json({ error: "请先设置", setupRequired: true }, 409);
    if (path === "api/runs/current") return json({ run: null });
    if (path === "api/calendar") return json({ month: "2026-10", today: "2026-10-03", timezone: "Asia/Shanghai", previousMonth: "2026-09", nextMonth: "2026-11", cells: [] });
    if (path === "api/papers") return json({ papers: [], total: 0, nextOffset: null, topics: [], counts: {}, day: null });
    throw new Error(`Unexpected ${path}`);
  });
  const root = document.createElement("div"); document.body.append(root);
  dispose = mountWorkbench(root, { fetch: fetcher });
  return { root, fetcher };
}
function input(root: HTMLElement, name: string, value: string) { root.querySelector<HTMLInputElement>(`[name="${name}"]`)!.value = value; }
function submit(root: HTMLElement) { root.querySelector(".settings-form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); }
it("opens first-run setup before data requests and refreshes the workspace after saving", async () => {
  const directory = await mkdtemp(join(tmpdir(), "workbench-settings-ui-"));
  const configPath = join(directory, "config.toml");
  const { root, fetcher } = setup(true, false, configPath);
  await vi.waitFor(() => expect(root.querySelector(".settings-form")).toBeTruthy());
  expect(root.querySelector("dialog")?.textContent).toContain("首次使用");
  expect(fetcher.mock.calls.some(([u]) => /api\/(calendar|papers|runs)/.test(String(u)))).toBe(false);
  expect(root.querySelector<HTMLInputElement>('[name="apiKey"]')?.type).toBe("password");
  expect(root.querySelector<HTMLInputElement>('[name="apiKey"]')?.value).toBe("");
  expect(root.querySelector("dialog")?.textContent).toContain("独立");
  input(root, "vaultRoot", "/new-notes"); input(root, "apiKey", "new-secret");
  root.querySelector<HTMLButtonElement>('[data-settings="add-topic"]')!.click();
  expect(root.querySelectorAll(".settings-topic")).toHaveLength(2);
  const second = root.querySelectorAll<HTMLElement>(".settings-topic")[1]!;
  input(second, "topicName", "Vision"); input(second, "topicTag", "vision"); input(second, "topicDescription", "Video");
  submit(root);
  await vi.waitFor(() => expect(root.querySelector(".paper-workspace")).toBeTruthy());
  expect(root.querySelector("dialog")).toBeNull();
  const body = JSON.parse(String(fetcher.mock.calls.find(([, i]) => i?.method === "POST")![1]!.body));
  expect(body.revision).toBeNull(); expect(body.values.vaultRoot).toBe("/new-notes"); expect(body.values.apiKey).toBe("new-secret");
  expect(body.values.topics).toHaveLength(2);
  expect(body.values.topics[0].id).toBe("existing-topic");
  expect(body.values.topics[1].id).toEqual(expect.any(String));
  expect(new Set(body.values.topics.map((t: { id: string }) => t.id)).size).toBe(2);
  expect((await readWorkbenchSettings(configPath)).values.topics).toEqual(body.values.topics);
  await rm(directory, { recursive: true, force: true });
  expect(root.querySelector<HTMLElement>(".connection-banner")!.hidden).toBe(true);
});
it("keeps edits and a blank unchanged key on save conflicts", async () => {
  const { root, fetcher } = setup(false, true);
  await vi.waitFor(() => expect(root.querySelector(".paper-workspace")).toBeTruthy());
  root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
  await vi.waitFor(() => expect(root.querySelector(".settings-form")).toBeTruthy());
  input(root, "model", "edited-model"); submit(root);
  await vi.waitFor(() => expect(root.querySelector(".settings-form [role=alert]")?.textContent).toContain("其他窗口"));
  expect(root.querySelector<HTMLInputElement>('[name="model"]')!.value).toBe("edited-model");
  expect(root.querySelector<HTMLButtonElement>('.settings-form [type="submit"]')!.disabled).toBe(false);
  const body = JSON.parse(String(fetcher.mock.calls.find(([, i]) => i?.method === "POST")![1]!.body));
  expect(body.revision).toBe("revision-1"); expect(body.values.apiKey).toBeUndefined();
});
it("switches settings groups without losing edits or renumbering topic identities", async () => {
  const { root } = setup();
  await vi.waitFor(() => expect(root.querySelector(".paper-workspace")).toBeTruthy());
  root.querySelector<HTMLButtonElement>('[data-action="settings"]')!.click();
  await vi.waitFor(() => expect(root.querySelector(".settings-form")).toBeTruthy());
  const nav = root.querySelector<HTMLElement>('[aria-label="设置分组"]');
  expect(nav).toBeTruthy();
  nav!.querySelector<HTMLButtonElement>('[data-settings-group="model"]')!.click();
  expect(root.querySelector<HTMLElement>('#settings-model')!.hidden).toBe(false);
  expect(root.querySelector<HTMLElement>('#settings-records')!.hidden).toBe(true);
  input(root, "model", "retained-model");
  nav!.querySelector<HTMLButtonElement>('[data-settings-group="discovery"]')!.click();
  root.querySelector<HTMLButtonElement>('[data-settings="add-topic"]')!.click();
  root.querySelector<HTMLButtonElement>('[data-settings="add-topic"]')!.click();
  const ids = Array.from(root.querySelectorAll<HTMLElement>('.settings-topic')).map(row => row.dataset.topicId);
  expect(new Set(ids).size).toBe(3);
  root.querySelectorAll<HTMLElement>('.settings-topic')[1]!.querySelector<HTMLButtonElement>('[data-settings="remove-topic"]')!.click();
  expect(Array.from(root.querySelectorAll<HTMLElement>('.settings-topic')).map(row => row.dataset.topicId)).toEqual([ids[0], ids[2]]);
  nav!.querySelector<HTMLButtonElement>('[data-settings-group="model"]')!.click();
  expect(root.querySelector<HTMLInputElement>('[name="model"]')!.value).toBe("retained-model");
});
