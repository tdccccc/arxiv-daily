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

describe("right-side paper workspace", () => {
  it("starts with every indexed discovery on the right, without auto-opening or duplicating the list on the left", async () => {
    const { root, fetcher } = setup(); await ready(root);
    expect(root.dataset.view).toBe("list");
    expect(root.querySelector(".library-pane [data-paper]")).toBeNull();
    expect(root.querySelector(".library-pane [data-document]")).toBeNull();
    expect(root.querySelector(".reading-pane")?.textContent).toContain("未生成详细总结");
    expect(fetcher.mock.calls.some(([url]) => String(url).startsWith("api/document?"))).toBe(false);
    expect(fetcher.mock.calls.some(([, init]) => init?.method === "POST")).toBe(false);
  });

  it("filters by date, explicitly opens the full report, and restores query, scope, ordering, page and scroll", async () => {
    const { root, fetcher } = setup(url => url.pathname.endsWith("api/papers") ? json(list([paper], { total: 42, nextOffset: 20, offset: Number(url.searchParams.get("offset")), day: url.searchParams.get("date") ? day : null })) : undefined);
    await ready(root); click(root, '[data-calendar-date="2026-10-01"]'); await ready(root);
    await vi.waitFor(() => expect(root.querySelector('[data-action="read-day"]')).toBeTruthy());
    expect(root.dataset.view).toBe("list"); expect(root.querySelector("article")).toBeNull();
    change(root, 'input[type="search"]', "Ada"); await vi.waitFor(() => expect(location.search).toContain("q=Ada"));
    click(root, '[data-scope="to_read"]'); await ready(root); change(root, '[data-filter="sort"]', "title");
    await vi.waitFor(() => expect(root.querySelector('[data-action="next-page"]')).toBeTruthy());
    click(root, '[data-action="next-page"]');
    await vi.waitFor(() => expect(location.search).toContain("offset=20")); await ready(root);
    const pane = root.querySelector<HTMLElement>(".reading-pane")!; pane.scrollTop = 318;
    click(root, '[data-action="read-day"]');
    await vi.waitFor(() => expect(root.querySelector("article")?.textContent).toContain("完整 Markdown 内容"));
    click(root, '[data-action="back"]'); await ready(root);
    expect(root.dataset.view).toBe("list"); expect(pane.scrollTop).toBe(318);
    const params = new URL(location.href).searchParams;
    expect(Object.fromEntries(["date", "q", "scope", "sort", "offset"].map(key => [key, params.get(key)]))).toEqual({ date: day.date, q: "Ada", scope: "to_read", sort: "title", offset: "20" });
    expect(root.querySelector<HTMLInputElement>('input[type="search"]')!.value).toBe("Ada");
    expect(fetcher.mock.calls.some(([, init]) => init?.method === "POST")).toBe(false);
  });

  it("opens saved overview information and offers explicit detail generation", async () => {
    const { root, fetcher } = setup(); await ready(root); click(root, "[data-paper]");
    await vi.waitFor(() => expect(root.querySelector(".paper-overview")?.textContent).toContain("The saved method"));
    expect(root.querySelector(".paper-overview")?.textContent).toContain("Relevant to efficient models");
    expect(root.querySelector('[data-source="original"]')?.getAttribute("rel")).toContain("noopener");
    expect(root.querySelector('[data-action="generate-paper"]')).toBeTruthy();
    expect(fetcher.mock.calls.some(([, init]) => init?.method === "POST")).toBe(false);
  });

  it("shows unknown index counts explicitly while retaining the saved full report", async () => {
    const { root } = setup(url => url.pathname.endsWith("api/papers") ? json(list([], { day: { ...day, papers: null }, total: 0 })) : undefined, "?date=2026-10-01");
    await vi.waitFor(() => expect(root.querySelector(".reading-pane")?.textContent).toContain("索引"));
    expect(root.querySelector(".reading-pane")?.textContent).not.toContain("0 篇");
    click(root, '[data-action="read-day"]'); await vi.waitFor(() => expect(root.querySelector("article")).toBeTruthy());
  });

  it("retains legacy marks and saves independent favorites only after confirmation", async () => {
    let saved = { ...paper, status: "saved" }; let resolveMark!: (response: Response) => void;
    const { root, fetcher } = setup((url, init) => {
      if (url.pathname.endsWith("api/papers")) return json(list([saved]));
      if (url.pathname.endsWith("api/paper")) return json({ paper: saved });
      if (url.pathname.endsWith("api/paper/mark")) return new Promise(resolve => { resolveMark = resolve; });
    });
    await ready(root); expect(root.querySelector('[data-mark="status"]')?.textContent).toContain("已保存");
    click(root, '[data-mark="star"]');
    expect(root.querySelector('[data-mark="star"]')?.getAttribute("aria-pressed")).toBe("false");
    expect(root.querySelector<HTMLButtonElement>('[data-mark="star"]')?.disabled).toBe(true);
    saved = { ...saved, priority: "high", starred: true }; resolveMark(json({ paper: saved }));
    await vi.waitFor(() => expect(root.querySelector('[data-mark="star"]')?.getAttribute("aria-pressed")).toBe("true"));
    expect(root.querySelector<HTMLSelectElement>('[data-mark="status"]')?.value).toBe("saved");
    const post = fetcher.mock.calls.find(([url]) => String(url) === "api/paper/mark")!;
    expect(JSON.parse(String(post[1]?.body))).toEqual({ key: paper.key, action: "star", value: true, expected: "normal" });
    expect(fetcher.mock.calls.some(([url, init]) => String(url) === "api/runs" && init?.method === "POST")).toBe(false);
  });

  it("reports mark conflicts and refreshes actual persisted state instead of displaying false success", async () => {
    let actual = paper;
    const { root } = setup((url, init) => {
      if (url.pathname.endsWith("api/papers")) return json(list([actual]));
      if (url.pathname.endsWith("api/paper")) return json({ paper: actual });
      if (url.pathname.endsWith("api/paper/mark") && init?.method === "POST") { actual = { ...paper, status: "read" }; return json({ error: "标记冲突，请刷新" }, 409); }
    });
    await ready(root); change(root, '[data-mark="status"]', "to_read");
    await vi.waitFor(() => expect(root.querySelector<HTMLSelectElement>('[data-mark="status"]')?.value).toBe("read"));
    expect(root.textContent).toContain("标记冲突");
    expect(root.textContent).not.toContain("保存成功");
  });

  it("keeps standalone Markdown files accessible on the right and returns to that file list", async () => {
    const { root } = setup(); await ready(root); click(root, '[data-action="browse-documents"]');
    await vi.waitFor(() => expect(root.querySelector(".reading-pane [data-document]")?.textContent).toContain("Unindexed saved note"));
    expect(root.querySelector(".library-pane [data-document]")).toBeNull();
    expect(root.querySelector<HTMLSelectElement>('[data-filter="topic"]')?.disabled).toBe(true);
    click(root, ".reading-pane [data-document]"); await vi.waitFor(() => expect(root.querySelector("article")).toBeTruthy());
    click(root, '[data-action="back"]');
    await vi.waitFor(() => expect(root.querySelector(".reading-pane [data-document]")?.textContent).toContain("Unindexed saved note"));
  });

  it("ignores stale paper and list responses after browser navigation", async () => {
    let finishPaper!: (response: Response) => void; let finishOld!: (response: Response) => void;
    const { root } = setup(url => {
      if (url.pathname.endsWith("api/paper")) return new Promise(resolve => { finishPaper = resolve; });
      if (url.pathname.endsWith("api/papers") && url.searchParams.get("q") === "old") return new Promise(resolve => { finishOld = resolve; });
    });
    await ready(root); click(root, "[data-paper]"); await vi.waitFor(() => expect(finishPaper).toBeTypeOf("function"));
    history.replaceState({}, "", "?q=old"); window.dispatchEvent(new PopStateEvent("popstate"));
    await vi.waitFor(() => expect(finishOld).toBeTypeOf("function"));
    change(root, 'input[type="search"]', "new"); await ready(root);
    finishPaper(json({ paper: { ...paper, title: "Obsolete paper" } })); finishOld(json(list([{ ...paper, title: "Obsolete list" }])));
    await new Promise(resolve => setTimeout(resolve, 20));
    expect(root.dataset.view).toBe("list"); expect(root.querySelector(".reading-pane")?.textContent).not.toContain("Obsolete");
  });
});

it("refreshes detailed-note availability after an explicit paper generation without leaving the overview", async () => {
  let generated = false, started = false;
  const run = { id: "detail", label: "详细总结", date: null, status: "running", output: "", exitCode: null, startedAt: "2026-10-01", finishedAt: null };
  const { root, fetcher } = setup((url, init) => {
    if (url.pathname.endsWith("api/paper")) return json({ paper: { ...paper, detailPath: generated ? "papers/detail.md" : null } });
    if (url.pathname.endsWith("api/runs") && init?.method === "POST") { started = true; return json({ run }, 202); }
    if (url.pathname.endsWith("api/runs/current") && started) { generated = true; return json({ run: { ...run, status: "completed" } }); }
  });
  await ready(root); click(root, "[data-paper]");
  await vi.waitFor(() => expect(root.querySelector('[data-action="generate-paper"]')).toBeTruthy());
  click(root, '[data-action="generate-paper"]');
  expect(root.querySelector<HTMLInputElement>("#run-paper")?.value).toBe(paper.arxivId);
  expect(started).toBe(false);
  root.querySelector<HTMLFormElement>(".generation-form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
  await vi.waitFor(() => expect(root.querySelector('[data-document="papers/detail.md"]')).toBeTruthy());
  expect(root.dataset.view).toBe("reading"); expect(root.querySelector(".paper-overview")).toBeTruthy();
  expect(JSON.parse(String(fetcher.mock.calls.find(([url]) => String(url) === "api/runs")?.[1]?.body))).toEqual({ kind: "paper", id: paper.arxivId });
});

it("keeps the saved list scroll when reloading a reading route and returning", async () => {
  const { root } = setup(undefined, `?q=Ada&paper=${encodeURIComponent(paper.key)}`);
  history.replaceState({ listScroll: 222 }, "");
  window.dispatchEvent(new PopStateEvent("popstate"));
  await vi.waitFor(() => expect(root.querySelector(".paper-overview")).toBeTruthy());
  click(root, '[data-action="back"]'); await ready(root);
  expect(root.querySelector<HTMLElement>(".reading-pane")!.scrollTop).toBe(222);
});

it("keeps mobile search filters open while typing, then allows returning to the updated list", async () => {
  const { root } = setup(); await ready(root);
  click(root, '[data-action="show-filters"]');
  change(root, 'input[type="search"]', "Ada");
  await vi.waitFor(() => expect(location.search).toContain("q=Ada")); await ready(root);
  expect(root.classList.contains("show-filters")).toBe(true);
  click(root, '.library-pane [data-action="show-filters"]');
  expect(root.classList.contains("show-filters")).toBe(false);
});

it("renders Markdown and formulas in paper lists and saved overviews without changing source text", async () => {
  const scientific = { ...paper, title: String.raw`**Cosmology** $H_0$`,
    abstract: String.raw`Measurements use \(\Omega_m\). <img src=x onerror=alert(1)>`,
    summary: { ...paper.summary,
      keyMethod: String.raw`**Model** with $x^2$.

$$
\frac{a}{b}
$$

- First constraint
- Second constraint`,
      whyRelevant: String.raw`Improves $\sigma_8$ without [unsafe](javascript:alert(1)).`,
    },
  };
  const original = JSON.stringify(scientific);
  const {root,fetcher}=setup(url=>url.pathname.endsWith("api/papers")?json(list([scientific])):url.pathname.endsWith("api/paper")?json({paper:scientific}):undefined);
  await vi.waitFor(()=>expect(root.querySelector(".paper-title")).toBeTruthy());
  expect(root.querySelector('.paper-title .katex')).toBeTruthy();
  expect(root.querySelector('.paper-title strong')?.textContent).toBe('Cosmology');
  expect(root.querySelector('.paper-reason .katex')).toBeTruthy();
  click(root,'[data-paper]');
  await vi.waitFor(()=>expect(root.querySelector('.paper-overview')).toBeTruthy());
  expect(root.querySelector('.paper-overview .document-title .katex')).toBeTruthy();
  expect(root.querySelectorAll('.overview-content .katex').length).toBe(4);
  expect(root.querySelector('.overview-content .katex-display')).toBeTruthy();
  expect(root.querySelector('.overview-content strong')?.textContent).toBe('Model');
  expect(root.querySelectorAll('.overview-content li')).toHaveLength(2);
  expect(root.querySelector('.overview-content img')).toBeNull();
  expect(root.querySelector('[href^="javascript:"]')).toBeNull();
  expect(root.querySelectorAll('.overview-content pre .katex')).toHaveLength(0);
  expect(JSON.stringify(scientific)).toBe(original);
  expect(fetcher.mock.calls.some(([,init])=>init?.method==='POST')).toBe(false);
});

it("renders math in document titles and table of contents", async () => {
 const {root}=setup(url=>url.pathname.endsWith('api/document')?json({path:'daily/2026-10-01.md',kind:'daily',title:'Expansion $H_0$',date:day.date,arxivId:'',authors:'',html:'<h2 id="matter">Matter</h2>',headings:[{id:'matter',title:String.raw`Matter $\Omega_m$`,level:2}],related:[],originalUrl:null,pdfUrl:null}):undefined,'?date=2026-10-01');
 await ready(root);click(root,'[data-action="read-day"]');
 await vi.waitFor(()=>expect(root.querySelector('article')).toBeTruthy());
 expect(root.querySelector('.document-title .katex')).toBeTruthy();
 expect(root.querySelector('.toc-pane a .katex')).toBeTruthy();
});
