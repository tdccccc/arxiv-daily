// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { mountWorkbench } from "../src/workbench/web/app";

const daily = { path: "arxiv-daily/daily/2026-10-01.md", kind: "daily", title: "2026-10-01 · 研究日报", date: "2026-10-01", authors: "", arxivId: "", size: 20, modifiedAt: "2026-10-01" };
const paper = { ...daily, path: "arxiv-daily/papers/2609.12345.md", kind: "papers", title: "Efficient inference", authors: "Ada", arxivId: "2609.12345" };
const status = { configPath: "/test/config.toml", vaultRoot: "/test/vault", output: { dailyDirectory: "/test/vault/daily", papersDirectory: "/test/vault/papers", summaryLanguage: "zh", linkStyle: "relative" }, llm: { provider: "openai", model: "test-model", ready: true, keyConfigured: true }, topics: [{ name: "Inference", tag: "inference", description: "Fast models", detail: true }], categories: ["cs.AI"], dailyReady: true, emailEnabled: false, paperCount: 1, recentRuns: [] };
const json = (value: unknown, code = 200) => new Response(JSON.stringify(value), { status: code, headers: { "Content-Type": "application/json" } });
const disposers: Array<() => void> = [];
afterEach(() => { disposers.splice(0).forEach(dispose => dispose()); document.body.innerHTML = ""; vi.restoreAllMocks(); });

function setup(override?: (url: URL, init?: RequestInit) => Response | Promise<Response> | undefined, pollIntervalMs = 10) {
  window.history.replaceState({}, "", "/capability/");
  const root = document.createElement("div");
  document.body.append(root);
  const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), window.location.href);
    const custom = override?.(url, init);
    if (custom) return await custom;
    if (url.pathname.endsWith("api/status")) return json(status);
    if (url.pathname.endsWith("api/runs/current")) return json({ run: null });
    if (url.pathname.endsWith("api/documents")) return json({ documents: url.searchParams.get("kind") === "papers" ? [paper] : [daily], total: 1, nextOffset: null, counts: { daily: 1, papers: 1 } });
    if (url.pathname.endsWith("api/document")) {
      const entry = url.searchParams.get("path") === paper.path ? paper : daily;
      return json({ ...entry, metadata: {}, headings: [{ id: "results", title: "结果", level: 2 }], html: `<h2 id="results">结果</h2><p>Saved research</p><a href="?document=${encodeURIComponent(paper.path)}#results">阅读全文</a>`, related: [], originalUrl: entry.kind === "papers" ? "https://arxiv.org/abs/2609.12345" : null, pdfUrl: null });
    }
    throw new Error(`Unexpected fetch: ${url}`);
  });
  disposers.push(mountWorkbench(root, { fetch: fetcher, searchDelayMs: 0, pollIntervalMs }));
  return { root, fetcher };
}
const button = (root: HTMLElement, label: string) => Array.from(root.querySelectorAll<HTMLButtonElement>("button")).find(item => item.textContent?.includes(label) || item.getAttribute("aria-label") === label)!;

describe("reading workbench UI", () => {
  it("opens saved Markdown, navigates document links and shows existing settings without generating", async () => {
    const { root, fetcher } = setup();
    await vi.waitFor(() => expect(root.querySelector("article")?.textContent).toContain("Saved research"));
    root.querySelector<HTMLAnchorElement>("article a")!.click();
    await vi.waitFor(() => expect(root.querySelector(".document-title")?.textContent).toBe(paper.title));
    expect(new URL(window.location.href).searchParams.get("document")).toBe(paper.path);
    expect(window.location.hash).toBe("#results");
    expect(root.querySelector<HTMLAnchorElement>('[data-source="original"]')?.rel).toContain("noopener");
    button(root, "设置").click();
    expect(root.querySelector('[role="dialog"]')?.textContent).toContain("/test/config.toml");
    expect(root.querySelector('[role="dialog"]')?.textContent).toContain("test-model");
    expect(fetcher.mock.calls.every(([, init]) => !init?.method || init.method === "GET")).toBe(true);
  });

  it("keeps the newest search result when older responses finish later", async () => {
    let completeOld!: (response: Response) => void;
    const { root } = setup(url => {
      if (!url.pathname.endsWith("api/documents")) return;
      if (url.searchParams.get("q") === "old") return new Promise(resolve => { completeOld = resolve; });
      if (url.searchParams.get("q") === "new") return json({ documents: [{ ...paper, title: "New result" }], total: 1, nextOffset: null, counts: { daily: 1, papers: 1 } });
    });
    await vi.waitFor(() => expect(root.querySelector("article")?.textContent).toContain("Saved research"));
    const input = root.querySelector<HTMLInputElement>('input[type="search"]')!;
    input.value = "old"; input.dispatchEvent(new Event("input", { bubbles: true }));
    await vi.waitFor(() => expect(completeOld).toBeTypeOf("function"));
    input.value = "new"; input.dispatchEvent(new Event("input", { bubbles: true }));
    await vi.waitFor(() => expect(root.querySelector(".document-list")?.textContent).toContain("New result"));
    completeOld(json({ documents: [{ ...daily, title: "Old result" }], total: 1, nextOffset: null, counts: { daily: 1, papers: 1 } }));
    await new Promise(resolve => setTimeout(resolve, 20));
    expect(root.querySelector(".document-list")?.textContent).not.toContain("Old result");
  });

  it("presents missing documents and disconnected services with a retry action", async () => {
    const { root } = setup(url => url.pathname.endsWith("api/document") ? json({ error: "文件已移动或删除" }, 404) : undefined);
    await vi.waitFor(() => expect(root.querySelector(".reading-pane")?.textContent).toContain("文件已移动或删除"));
    expect(button(root, "重试读取")).toBeTruthy();
    const another = setup(() => Promise.reject(new TypeError("Failed to fetch")));
    await vi.waitFor(() => expect(another.root.textContent).toContain("无法连接本地工作台"));
    expect(button(another.root, "重新连接")).toBeTruthy();
  });

  it("submits generation explicitly, shows completion output and refreshes the saved list", async () => {
    let started = false;
    let reads = 0;
    const run = { id: "job-1", label: "生成今日日报", status: "running", output: "Starting", exitCode: null, startedAt: "2026-10-01", finishedAt: null };
    const { root, fetcher } = setup((url, init) => {
      if (url.pathname.endsWith("api/runs") && init?.method === "POST") { started = true; return json({ run }, 202); }
      if (url.pathname.endsWith("api/runs/current") && started) return json({ run: { ...run, status: "completed", output: "Saved <script>ignored</script>", exitCode: 0, finishedAt: "2026-10-01" } });
      if (url.pathname.endsWith("api/documents")) reads += 1;
    });
    await vi.waitFor(() => expect(root.querySelector("article")?.textContent).toContain("Saved research"));
    button(root, "生成").click();
    expect(started).toBe(false);
    root.querySelector<HTMLFormElement>(".generation-form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    await vi.waitFor(() => expect(root.querySelector(".run-tray")?.textContent).toContain("已完成"));
    expect(root.querySelector(".run-tray")?.textContent).toContain("Saved <script>ignored</script>");
    expect(root.querySelector(".run-tray script")).toBeNull();
    expect(reads).toBeGreaterThan(1);
    const submitted = fetcher.mock.calls.find(([url, init]) => String(url).endsWith("api/runs") && init?.method === "POST");
    expect(JSON.parse(String(submitted?.[1]?.body))).toEqual({ kind: "daily" });
  });

  it("does not let an earlier idle poll erase a newly started task", async () => {
    let finishIdle!: (response: Response) => void;
    const run = { id: "new-job", label: "新任务", status: "running", output: "Working", exitCode: null, startedAt: "2026-10-01", finishedAt: null };
    const { root } = setup((url, init) => {
      if (url.pathname.endsWith("api/runs/current")) return new Promise(resolve => { finishIdle = resolve; });
      if (url.pathname.endsWith("api/runs") && init?.method === "POST") return json({ run }, 202);
    }, 10000);
    await vi.waitFor(() => expect(root.querySelector("article")?.textContent).toContain("Saved research"));
    button(root, "生成").click();
    root.querySelector<HTMLFormElement>(".generation-form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    await vi.waitFor(() => expect(root.querySelector(".run-tray")?.textContent).toContain("新任务"));
    finishIdle(json({ run: null }));
    await new Promise(resolve => setTimeout(resolve, 20));
    expect(root.querySelector<HTMLElement>(".run-tray")?.hidden).toBe(false);
    expect(root.querySelector(".run-tray")?.textContent).toContain("正在运行");
  });

  it("cancels the displayed task using its ID and reports the confirmed cancelled state", async () => {
    const run = { id: "cancel-job", label: "生成日报", status: "running", output: "Working", exitCode: null, startedAt: "2026-10-01", finishedAt: null };
    let cancelled = false;
    const { root, fetcher } = setup((url, init) => {
      if (url.pathname.endsWith("api/runs/current")) return json({ run: { ...run, status: cancelled ? "cancelled" : "running" } });
      if (url.pathname.endsWith("api/runs/cancel") && init?.method === "POST") { cancelled = true; return json({ run: { ...run, status: "cancelled" } }); }
    });
    await vi.waitFor(() => expect(button(root, "取消任务")).toBeTruthy());
    button(root, "取消任务").click();
    await vi.waitFor(() => expect(root.querySelector(".run-tray")?.textContent).toContain("已取消"));
    const request = fetcher.mock.calls.find(([url]) => String(url).endsWith("api/runs/cancel"));
    expect(JSON.parse(String(request?.[1]?.body))).toEqual({ id: "cancel-job" });
  });

  it("follows browser history and ignores an older document response", async () => {
    let finishDaily!: (response: Response) => void;
    const { root } = setup(url => url.pathname.endsWith("api/document") && url.searchParams.get("path") === daily.path ? new Promise(resolve => { finishDaily = resolve; }) : undefined);
    await vi.waitFor(() => expect(finishDaily).toBeTypeOf("function"));
    history.replaceState({}, "", `?document=${encodeURIComponent(paper.path)}`);
    window.dispatchEvent(new PopStateEvent("popstate"));
    await vi.waitFor(() => expect(root.querySelector(".document-title")?.textContent).toBe(paper.title));
    finishDaily(json({ ...daily, html: "<p>Old article</p>", headings: [], metadata: {}, related: [], originalUrl: null, pdfUrl: null }));
    await new Promise(resolve => setTimeout(resolve, 20));
    expect(root.querySelector(".document-title")?.textContent).toBe(paper.title);
    expect(root.querySelector("article")?.textContent).not.toContain("Old article");
  });

  it("moves keyboard focus to reading content and supports collection arrow keys", async () => {
    const { root } = setup();
    await vi.waitFor(() => expect(root.querySelector("article")?.textContent).toContain("Saved research"));
    root.querySelector<HTMLAnchorElement>(".skip-link")!.click();
    expect(document.activeElement).toBe(root.querySelector(".reading-pane"));
    const tab = root.querySelector<HTMLButtonElement>('[data-kind="daily"]')!;
    tab.focus();
    tab.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true, cancelable: true }));
    expect(root.querySelector('[data-kind="papers"]')?.getAttribute("aria-selected")).toBe("true");
    expect(document.activeElement).toBe(root.querySelector('[data-kind="papers"]'));
  });
});
