// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { mountWorkbench } from "../src/workbench/web/app";

const paper = { key: "arxiv:2609.12345", arxivId: "2609.12345", title: "Efficient inference", authors: ["Ada"], published: "2026-09-29", topics: ["Inference"], category: "cs.AI", status: "inbox", priority: "normal", starred: false, abstract: "Original abstract", summary: { coreProblem: "The existing problem", keyMethod: "The saved method", whyRelevant: "Relevant to efficient models" }, detailPath: null, reports: [{ path: "daily/2026-10-01.md", date: "2026-10-01", title: "研究日报", available: true }], originalUrl: "https://arxiv.org/abs/2609.12345", pdfUrl: "https://arxiv.org/pdf/2609.12345", provenance: null, novelty: null };
const day = { date: "2026-10-01", state: "has-report", reportPath: "daily/2026-10-01.md", reportTitle: "研究日报", papers: 1, message: "日报已保存。", canGenerate: false, actionLabel: null };
const status = { configPath: "/test/config.toml", vaultRoot: "/test/vault", output: { summaryLanguage: "zh" }, llm: { provider: "openai", model: "test", ready: true, keyConfigured: true }, topics: [], categories: [], emailEnabled: false };
const list = (papers = [paper], extra = {}) => ({ papers, total: papers.length, offset: 0, limit: 20, nextOffset: null, counts: { all: 1, inbox: 1, to_read: 0, read: 0, starred: 0 }, libraryCount: 1, topics: ["Inference"], day: null, ...extra });
const json = (value: unknown, code = 200) => new Response(JSON.stringify(value), { status: code, headers: { "Content-Type": "application/json" } });
const disposers: Array<() => void> = [];
afterEach(() => { disposers.splice(0).forEach(dispose => dispose()); document.body.innerHTML = ""; vi.restoreAllMocks(); });
function setup(override?: (url: URL, init?: RequestInit) => Response | Promise<Response> | undefined, route = "") {
  history.replaceState({}, "", `/paper-test/${route}`);
  const root = document.createElement("div"); document.body.append(root);
  const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), location.href);
    const custom = override?.(url, init); if (custom) return await custom;
    if (url.pathname.endsWith("api/status")) return json(status);
    if (url.pathname.endsWith("api/preferences")) return json({ sidebarWidth: 420, sidebarCollapsed: false });
    if (url.pathname.endsWith("api/calendar")) return json({ month: "2026-10", today: day.date, timezone: "Asia/Shanghai", previousMonth: "2026-09", nextMonth: "2026-11", cells: [day] });
    if (url.pathname.endsWith("api/papers")) return json(list([paper], { day: url.searchParams.get("date") ? day : null }));
    if (url.pathname.endsWith("api/paper")) return json({ paper });
    if (url.pathname.endsWith("api/runs/current")) return json({ run: null });
    if (url.pathname.endsWith("api/documents")) return json({ documents: [{ path: "papers/standalone.md", kind: "papers", title: "Unindexed saved note", date: "", arxivId: "", authors: "", size: 10 }], total: 1, nextOffset: null, counts: { daily: 1, papers: 1 } });
    if (url.pathname.endsWith("api/document")) return json({ path: url.searchParams.get("path"), kind: "daily", title: "完整研究日报", date: day.date, arxivId: "", authors: "", html: '<h2 id="all">完整 Markdown 内容</h2>', headings: [{ id: "all", title: "完整 Markdown 内容", level: 2 }], related: [], originalUrl: null, pdfUrl: null });
    throw new Error(`Unexpected request: ${url}`);
  });
  disposers.push(mountWorkbench(root, { fetch: fetcher, searchDelayMs: 0, pollIntervalMs: 10 }));
  return { root, fetcher };
}
const click = (root: HTMLElement, selector: string) => root.querySelector<HTMLButtonElement>(selector)!.click();
const ready = async (root: HTMLElement) => vi.waitFor(() => expect(root.querySelector(".reading-pane [data-paper]")?.textContent).toContain(paper.title));
const change = (root: HTMLElement, selector: string, value: string) => { const input = root.querySelector<HTMLInputElement | HTMLSelectElement>(selector)!; input.value = value; input.dispatchEvent(new Event(input.tagName === "SELECT" ? "change" : "input", { bubbles: true })); };

describe("bounded reading navigation", () => {
  it("restores list, paper, linked documents and scroll; discards forward branch", async () => {
    const { root } = setup(); await ready(root);
    const back = () => root.querySelector<HTMLButtonElement>('[data-action="history-back"]');
    const forward = () => root.querySelector<HTMLButtonElement>('[data-action="history-forward"]');
    expect(back()).not.toBeNull(); expect(back()!.disabled).toBe(true); expect(forward()!.disabled).toBe(true);
    const pane = root.querySelector<HTMLElement>(".reading-pane")!;
    pane.scrollTop = 180; click(root, '[data-paper]');
    await vi.waitFor(() => expect(root.querySelector('.paper-overview')).toBeTruthy());
    pane.scrollTop = 280; click(root, '[data-document]');
    await vi.waitFor(() => expect(root.querySelector('article')).toBeTruthy());
    pane.scrollTop = 380;
    back()!.click(); await vi.waitFor(() => expect(root.querySelector('.paper-overview')).toBeTruthy());
    expect(pane.scrollTop).toBe(280); expect(forward()!.disabled).toBe(false);
    back()!.click(); await ready(root); expect(pane.scrollTop).toBe(180); expect(back()!.disabled).toBe(true);
    forward()!.click(); await vi.waitFor(() => expect(root.querySelector('.paper-overview')).toBeTruthy());
    expect(pane.scrollTop).toBe(280);
    forward()!.click(); await vi.waitFor(() => expect(root.querySelector('article')).toBeTruthy()); expect(pane.scrollTop).toBe(380);
    back()!.click(); await vi.waitFor(() => expect(root.querySelector('.paper-overview')).toBeTruthy());
    click(root, '[data-action="back"]'); await ready(root); expect(forward()!.disabled).toBe(true);
  });
  it("restores linked document anchors, recovers from failures and ignores a stale request", async () => {
    let finishSlow!: (response: Response) => void;
    const { root } = setup(url => {
      if (!url.pathname.endsWith('api/document')) return;
      const path = url.searchParams.get('path');
      if (path === 'slow.md') return new Promise(resolve => { finishSlow = resolve; });
      if (path === 'missing.md') return json({ error: 'Missing document' }, 404);
      return json({ path, kind: 'papers', title: path, date: '', authors: '', html: '<h2 id="section">Section</h2><a href="?document=second.md#section">Linked</a><a href="?document=missing.md">Missing</a><a href="?document=slow.md">Slow</a>', headings: [{ id: 'section', title: 'Section', level: 2 }], related: [], originalUrl: null, pdfUrl: null });
    }, '?document=first.md&q=Ada&scope=to_read');
    await vi.waitFor(() => expect(root.querySelector('article')).toBeTruthy());
    const pane = root.querySelector<HTMLElement>('.reading-pane')!;
    const back = () => click(root, '[data-action="history-back"]');
    const forward = () => click(root, '[data-action="history-forward"]');
    pane.scrollTop = 111;
    click(root, '.toc-pane a'); expect(location.hash).toBe('#section');
    pane.scrollTop = 222; click(root, 'article a[href="?document=second.md#section"]');
    await vi.waitFor(() => expect(root.querySelector('.document-title')?.textContent).toBe('second.md'));
    pane.scrollTop = 333; back();
    await vi.waitFor(() => expect(root.querySelector('.document-title')?.textContent).toBe('first.md'));
    expect(location.hash).toBe('#section'); expect(pane.scrollTop).toBe(222);
    back(); await vi.waitFor(() => expect(location.hash).toBe('')); await vi.waitFor(() => expect(pane.scrollTop).toBe(111));
    forward(); await vi.waitFor(() => expect(location.hash).toBe('#section'));
    await vi.waitFor(() => expect(root.querySelector('article')).toBeTruthy());
    click(root, 'article a[href="?document=missing.md"]');
    await vi.waitFor(() => expect(pane.textContent).toContain('Missing document'));
    back(); await vi.waitFor(() => expect(root.querySelector('.document-title')?.textContent).toBe('first.md'));
    click(root, 'article a[href="?document=slow.md"]');
    await vi.waitFor(() => expect(finishSlow).toBeTypeOf('function'));
    back(); await vi.waitFor(() => expect(root.querySelector('.document-title')?.textContent).toBe('first.md'));
    finishSlow(json({ path: 'slow.md', title: 'Stale result', kind: 'papers', date: '', authors: '', html: '<p>Wrong page</p>', headings: [], related: [], originalUrl: null, pdfUrl: null }));
    await new Promise(resolve => setTimeout(resolve, 20)); expect(pane.textContent).not.toContain('Wrong page');
    expect(new URL(location.href).searchParams.get('q')).toBe('Ada'); expect(new URL(location.href).searchParams.get('scope')).toBe('to_read');
  });
  it("keeps browser back compatible and never traverses a foreign entry with the controls", async () => {
    const { root } = setup(); await ready(root);
    const go = vi.spyOn(history, 'go');
    click(root, '[data-action="history-back"]'); expect(go).not.toHaveBeenCalled();
    const pane = root.querySelector<HTMLElement>('.reading-pane')!; pane.scrollTop = 98;
    click(root, '[data-paper]'); await vi.waitFor(() => expect(root.querySelector('.paper-overview')).toBeTruthy());
    pane.scrollTop = 167; pane.dispatchEvent(new Event('scroll'));
    history.back(); await ready(root); expect(pane.scrollTop).toBe(98);
    click(root, '[data-action="history-forward"]'); await vi.waitFor(() => expect(root.querySelector('.paper-overview')).toBeTruthy()); expect(pane.scrollTop).toBe(167);
    history.replaceState({}, '', '?q=external'); window.dispatchEvent(new PopStateEvent('popstate')); await ready(root);
    expect(root.querySelector<HTMLButtonElement>('[data-action="history-back"]')!.disabled).toBe(true);
    expect(root.querySelector<HTMLButtonElement>('[data-action="history-forward"]')!.disabled).toBe(true);
  });

  it("keeps the current filters when a same-document Markdown link only supplies its path and anchor", async () => {
    const { root } = setup(url => url.pathname.endsWith('api/document') ? json({ path: 'first.md', kind: 'papers', title: 'First', date: '', authors: '', html: '<h2 id="section">Section</h2><a href="?document=first.md#section">Jump</a>', headings: [], related: [], originalUrl: null, pdfUrl: null }) : undefined, '?document=first.md&q=Ada&scope=to_read');
    await vi.waitFor(() => expect(root.querySelector('article')).toBeTruthy());
    click(root, 'article a');
    expect(new URL(location.href).searchParams.get('q')).toBe('Ada');
    expect(new URL(location.href).searchParams.get('scope')).toBe('to_read');
    expect(location.hash).toBe('#section');
  });

});
