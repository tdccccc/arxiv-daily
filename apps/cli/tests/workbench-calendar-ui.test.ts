// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
import { mountWorkbench } from "../src/workbench/web/app";

const daily = { path: "arxiv-daily/daily/2026-10-01.md", kind: "daily", title: "十月研究日报", date: "2026-10-01", authors: "", arxivId: "", size: 20, modifiedAt: "2026-10-01" };
const paper = { ...daily, path: "arxiv-daily/papers/2609.12345.md", kind: "papers", title: "Inference paper", date: "2026-09-29" };
const status = { configPath: "/test/config.toml", vaultRoot: "/test/vault", output: { summaryLanguage: "zh" }, llm: { provider: "openai", model: "test", ready: true, keyConfigured: true }, topics: [], categories: ["cs.AI"], emailEnabled: false };
const json = (value: unknown, code = 200) => new Response(JSON.stringify(value), { status: code, headers: { "Content-Type": "application/json" } });
const states = {
  "2026-10-01": { state: "has-report", reportPath: daily.path, reportTitle: daily.title, papers: 3, message: "日报已保存。", canGenerate: false, actionLabel: null },
  "2026-10-02": { state: "no-matches", papers: 0, message: "已完成筛选，没有符合主题的论文。", canGenerate: false, actionLabel: null },
  "2026-10-03": { state: "failed", message: "网络请求失败，可以重试。", canGenerate: true, actionLabel: "重试生成" },
  "2026-10-04": { state: "failed", message: "配置无效；修复后请在终端处理。", canGenerate: false, actionLabel: null },
  "2026-10-05": { state: "report-missing", message: "已完成，但日报文件已移动或删除。", canGenerate: false, actionLabel: null },
};

function calendar(month = "2026-10") {
  const first = new Date(`${month}-01T00:00:00Z`);
  const previous = new Date(first); previous.setUTCMonth(previous.getUTCMonth() - 1);
  const next = new Date(first); next.setUTCMonth(next.getUTCMonth() + 1);
  const count = new Date(next.getTime() - 86400000).getUTCDate();
  const cells: Array<null | Record<string, unknown>> = Array.from({ length: (first.getUTCDay() + 6) % 7 }, () => null);
  for (let day = 1; day <= count; day++) {
    const date = `${month}-${String(day).padStart(2, "0")}`;
    cells.push({ date, state: date > "2026-10-15" ? "future" : "not-generated", reportPath: null, reportTitle: null, papers: null, message: date > "2026-10-15" ? "尚未到该日期。" : "尚未生成日报。", canGenerate: date <= "2026-10-15", actionLabel: date <= "2026-10-15" ? "生成日报" : null, ...states[date as keyof typeof states] });
  }
  while (cells.length % 7) cells.push(null);
  return { month, today: "2026-10-15", timezone: "Asia/Shanghai", previousMonth: previous.toISOString().slice(0, 7), nextMonth: next.toISOString().slice(0, 7), cells };
}

const disposers: Array<() => void> = [];
afterEach(() => { disposers.splice(0).forEach(dispose => dispose()); document.body.innerHTML = ""; vi.restoreAllMocks(); });
function setup(override?: (url: URL, init?: RequestInit) => Response | Promise<Response> | undefined, options: { mobile?: boolean; route?: string; pollIntervalMs?: number } = {}) {
  window.history.replaceState({}, "", `/calendar-test/${options.route || ""}`);
  if (options.mobile) vi.spyOn(window, "matchMedia").mockReturnValue({ matches: true } as MediaQueryList);
  const root = document.createElement("div"); document.body.append(root);
  const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), window.location.href);
    const custom = override?.(url, init);
    if (custom) return await custom;
    if (url.pathname.endsWith("api/calendar")) return json(calendar(url.searchParams.get("month") || undefined));
    if (url.pathname.endsWith("api/preferences")) return json({ sidebarWidth: null, sidebarCollapsed: false });
    if (url.pathname.endsWith("api/papers")) {
      const date = url.searchParams.get("date");
      return json({ papers: [], total: 0, offset: 0, limit: 20, nextOffset: null, counts: { all: 0, inbox: 0, to_read: 0, read: 0, starred: 0 }, libraryCount: 0, topics: [], day: date ? calendar(date.slice(0, 7)).cells.find(cell => cell?.date === date) : null });
    }
    if (url.pathname.endsWith("api/status")) return json(status);
    if (url.pathname.endsWith("api/runs/current")) return json({ run: null });
    if (url.pathname.endsWith("api/documents")) return json({ documents: url.searchParams.get("kind") === "papers" ? [paper] : [daily], total: 1, nextOffset: null, counts: { daily: 1, papers: 1 } });
    if (url.pathname.endsWith("api/document")) {
      const entry = url.searchParams.get("path") === paper.path ? paper : daily;
      return json({ ...entry, metadata: {}, headings: [], html: `<p>Saved ${entry.title}</p>`, related: [], originalUrl: null, pdfUrl: null });
    }
    throw new Error(`Unexpected request: ${url}`);
  });
  disposers.push(mountWorkbench(root, { fetch: fetcher, searchDelayMs: 0, pollIntervalMs: options.pollIntervalMs ?? 10000 }));
  return { root, fetcher };
}
const button = (root: HTMLElement, label: string) => Array.from(root.querySelectorAll<HTMLButtonElement>("button")).find(item => item.textContent?.trim() === label || item.getAttribute("aria-label") === label)!;
const day = (root: HTMLElement, date: string) => root.querySelector<HTMLButtonElement>(`button[data-calendar-date="${date}"]`)!;
const ready = async (root: HTMLElement) => { await vi.waitFor(() => expect(day(root, "2026-10-01")).toBeTruthy()); };
const posted = (fetcher: ReturnType<typeof setup>["fetcher"]) => fetcher.mock.calls.filter(([, init]) => init?.method === "POST");

describe("daily calendar in the reading workbench", () => {
  it("shows known, zero and unknown paper counts directly in report dates", async () => {
    const { root, fetcher } = setup(url => {
      if (!url.pathname.endsWith("api/calendar")) return;
      const data = calendar();
      Object.assign(data.cells.find(cell => cell?.date === "2026-10-01")!, { papers: 128 });
      Object.assign(data.cells.find(cell => cell?.date === "2026-10-06")!, { state: "has-report", reportPath: "old.md", papers: null });
      return json(data);
    });
    await ready(root);
    expect(day(root, "2026-10-01").textContent).toContain("128篇");
    expect(day(root, "2026-10-01").getAttribute("aria-label")).toContain("128 篇论文");
    expect(day(root, "2026-10-02").textContent).toContain("0篇");
    expect(day(root, "2026-10-06").textContent).toContain("—");
    expect(day(root, "2026-10-06").getAttribute("aria-label")).toContain("论文数未知");
    expect(day(root, "2026-10-03").textContent).not.toContain("0");
    expect(posted(fetcher)).toHaveLength(0);
  });

  it("shows a Monday-first month, today and report states and opens a report without generating", async () => {
    const { root, fetcher } = setup(); await ready(root);
    expect(root.querySelector(".calendar-weekdays")?.textContent?.replace(/\s/g, "")).toBe("一二三四五六日");
    expect(root.querySelector(".calendar-month")?.textContent).toBe("2026 年 10 月");
    expect(day(root, "2026-10-15").getAttribute("aria-current")).toBe("date");
    expect(day(root, "2026-10-01").getAttribute("aria-label")).toContain("已有日报");
    day(root, "2026-10-01").click();
    await vi.waitFor(() => expect(root.querySelector('[data-action="read-day"]')).toBeTruthy());
    root.querySelector<HTMLButtonElement>('[data-action="read-day"]')!.click();
    await vi.waitFor(() => expect(root.querySelector(".document-title")?.textContent).toBe(daily.title));
    expect(root.querySelector(".calendar-summary")?.textContent).toContain("3 篇");
    expect(posted(fetcher)).toHaveLength(0);
  });

  it("inspects no-match, permanent failure, missing and future days without offering generation or retaining the previous report", async () => {
    const { root, fetcher } = setup(); await ready(root);
    for (const [date, text] of [["2026-10-02", "没有符合主题的论文"], ["2026-10-04", "配置无效"], ["2026-10-05", "已移动或删除"], ["2026-10-16", "尚未到该日期"]]) {
      day(root, date).click();
      await vi.waitFor(() => expect(root.querySelector(".reading-pane")?.textContent).toContain(text));
      expect(root.querySelector("article")).toBeNull();
      expect(root.querySelector(".reading-pane [data-action='generate-date']")).toBeNull();
      expect(new URL(location.href).searchParams.get("date")).toBe(date);
      expect(new URL(location.href).searchParams.has("document")).toBe(false);
    }
    expect(posted(fetcher)).toHaveLength(0);
  });

  it("prefills the selected failed date and only generates on explicit form submission", async () => {
    const run = { id: "retry", kind: "daily", date: "2026-10-03", label: "生成 2026-10-03 日报", status: "running", output: "", exitCode: null, startedAt: "2026-10-15", finishedAt: null };
    const { root, fetcher } = setup((url, init) => url.pathname.endsWith("api/runs") && init?.method === "POST" ? json({ run }, 202) : undefined);
    await ready(root); day(root, "2026-10-03").click();
    await vi.waitFor(() => expect(root.querySelector(".reading-pane")?.textContent).toContain("网络请求失败"));
    root.querySelector<HTMLButtonElement>(".reading-pane [data-action='generate-date']")!.click();
    expect(root.querySelector<HTMLInputElement>("#run-date")?.value).toBe("2026-10-03");
    expect(posted(fetcher)).toHaveLength(0);
    root.querySelector<HTMLFormElement>(".generation-form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    await vi.waitFor(() => expect(posted(fetcher)).toHaveLength(1));
    expect(JSON.parse(String(posted(fetcher)[0][1]?.body))).toEqual({ kind: "daily", date: "2026-10-03" });
  });

  it("ignores a slow older month response and returns to configured today", async () => {
    let finishSeptember!: (value: Response) => void;
    const { root } = setup(url => url.pathname.endsWith("api/calendar") && url.searchParams.get("month") === "2026-09" ? new Promise(resolve => { finishSeptember = resolve; }) : undefined);
    await ready(root); button(root, "上个月").click();
    await vi.waitFor(() => expect(finishSeptember).toBeTypeOf("function"));
    button(root, "下个月").click();
    await vi.waitFor(() => expect(day(root, "2026-11-01")).toBeTruthy());
    finishSeptember(json(calendar("2026-09")));
    await new Promise(resolve => setTimeout(resolve, 20));
    expect(root.querySelector(".calendar-month")?.textContent).toBe("2026 年 11 月");
    button(root, "回到今天").click();
    await vi.waitFor(() => expect(root.querySelector(".calendar-summary")?.textContent).toContain("2026-10-15"));
    expect(day(root, "2026-10-15").getAttribute("aria-pressed")).toBe("true");
  });

  it("does not let an old month response undo a newer visible-day selection", async () => {
    let finishSeptember!: (value: Response) => void;
    const { root } = setup(url => url.pathname.endsWith("api/calendar") && url.searchParams.get("month") === "2026-09" ? new Promise(resolve => { finishSeptember = resolve; }) : undefined);
    await ready(root); button(root, "上个月").click();
    await vi.waitFor(() => expect(finishSeptember).toBeTypeOf("function"));
    day(root, "2026-10-02").click();
    finishSeptember(json(calendar("2026-09")));
    await vi.waitFor(() => expect(root.querySelector(".reading-pane")?.textContent).toContain("没有符合主题的论文"));
    expect(day(root, "2026-10-02").getAttribute("aria-pressed")).toBe("true");
    expect(root.querySelector(".calendar-month")?.textContent).toBe("2026 年 10 月");
  });

  it("refreshes date state as jobs run and finish without changing an unrelated selected day", async () => {
    let current: "idle" | "running" | "completed" = "idle";
    const run = { id: "job", kind: "daily", date: "2026-10-03", label: "生成日报", output: "", exitCode: null, startedAt: "2026-10-15", finishedAt: null };
    const { root } = setup((url, init) => {
      if (url.pathname.endsWith("api/runs") && init?.method === "POST") { current = "running"; return json({ run: { ...run, status: current } }, 202); }
      if (url.pathname.endsWith("api/runs/current")) return json({ run: current === "idle" ? null : { ...run, status: current } });
      if (url.pathname.endsWith("api/calendar") && current !== "idle") {
        const value = calendar();
        Object.assign(value.cells.find(cell => cell?.date === run.date)!, { state: current === "running" ? "running" : "no-matches", papers: current === "running" ? null : 0, message: current === "running" ? "正在生成。" : "没有符合主题的论文。", canGenerate: false, actionLabel: null });
        return json(value);
      }
    }, { pollIntervalMs: 15 });
    await ready(root); day(root, "2026-10-03").click();
    root.querySelector<HTMLButtonElement>(".calendar-summary [data-action='generate-date']")!.click();
    root.querySelector<HTMLFormElement>(".generation-form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    await vi.waitFor(() => expect(day(root, "2026-10-03").getAttribute("aria-label")).toContain("正在生成"));
    day(root, "2026-10-02").click(); current = "completed";
    await vi.waitFor(() => expect(day(root, "2026-10-03").getAttribute("aria-label")).toContain("无匹配论文"));
    expect(day(root, "2026-10-02").getAttribute("aria-pressed")).toBe("true");
    expect(new URL(location.href).searchParams.get("date")).toBe("2026-10-02");
    expect(root.querySelector(".day-reading")?.textContent).toContain("2026-10-02");
  });

  it("keeps an in-flight same-month refresh when another day is selected", async () => {
    let finishRefresh: ((response: Response) => void) | undefined;
    let holdRefresh = false;
    const { root } = setup(url => url.pathname.endsWith("api/calendar") && holdRefresh ? new Promise(resolve => { finishRefresh = resolve; }) : undefined);
    await ready(root); day(root, "2026-10-02").click();
    holdRefresh = true; button(root, "刷新文档").click();
    await vi.waitFor(() => expect(finishRefresh).toBeTypeOf("function"));
    day(root, "2026-10-03").click();
    const updated = calendar();
    Object.assign(updated.cells.find(cell => cell?.date === "2026-10-03")!, { state: "running", message: "该日期的日报任务正在运行。", canGenerate: false, actionLabel: null });
    finishRefresh!(json(updated));
    await vi.waitFor(() => expect(day(root, "2026-10-03").getAttribute("aria-label")).toContain("正在生成"));
    expect(root.querySelector(".day-reading")?.textContent).toContain("正在运行");
    expect(new URL(location.href).searchParams.get("date")).toBe("2026-10-03");
  });

  it("restores date routes through history and syncs a report's date without changing paper navigation", async () => {
    const { root, fetcher } = setup(undefined, { route: "?date=2026-10-02" }); await ready(root);
    await vi.waitFor(() => expect(root.querySelector(".day-reading")?.textContent).toContain("2026-10-02"));
    expect(root.querySelector("article")).toBeNull();
    history.replaceState({}, "", `?document=${encodeURIComponent(paper.path)}`); window.dispatchEvent(new PopStateEvent("popstate"));
    await vi.waitFor(() => expect(root.querySelector(".document-title")?.textContent).toBe(paper.title));
    history.replaceState({}, "", "?date=2026-10-03"); window.dispatchEvent(new PopStateEvent("popstate"));
    await vi.waitFor(() => expect(root.querySelector(".day-reading")?.textContent).toContain("2026-10-03"));
    history.replaceState({}, "", `?document=${encodeURIComponent(daily.path)}`); window.dispatchEvent(new PopStateEvent("popstate"));
    await vi.waitFor(() => expect(root.querySelector(".document-title")?.textContent).toBe(daily.title));
    expect(day(root, "2026-10-01").getAttribute("aria-pressed")).toBe("true");
    expect(posted(fetcher)).toHaveLength(0);
  });

  it("moves date focus with arrows and Home/End without generating or changing selection", async () => {
    const { root, fetcher } = setup(); await ready(root);
    const first = day(root, "2026-10-08"); first.focus();
    first.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true, cancelable: true }));
    expect(document.activeElement).toBe(day(root, "2026-10-09"));
    document.activeElement!.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true, cancelable: true }));
    expect(document.activeElement).toBe(day(root, "2026-10-16"));
    document.activeElement!.dispatchEvent(new KeyboardEvent("keydown", { key: "Home", bubbles: true, cancelable: true }));
    expect(document.activeElement).toBe(day(root, "2026-10-12"));
    document.activeElement!.dispatchEvent(new KeyboardEvent("keydown", { key: "End", bubbles: true, cancelable: true }));
    expect(document.activeElement).toBe(day(root, "2026-10-18"));
    expect(posted(fetcher)).toHaveLength(0);
  });

  it("opens mobile filters, selects a date into the right list, and returns there from reading", async () => {
    const { root } = setup(undefined, { mobile: true }); await ready(root);
    root.querySelector<HTMLButtonElement>('[data-action="show-filters"]')!.click();
    expect(root.classList.contains("show-filters")).toBe(true);
    const toggle = button(root, "展开日历"); expect(toggle.getAttribute("aria-expanded")).toBe("false");
    toggle.click(); expect(root.querySelector<HTMLElement>(".calendar-body")!.hidden).toBe(false);
    day(root, "2026-10-01").click();
    await vi.waitFor(() => expect(root.querySelector('[data-action="read-day"]')).toBeTruthy());
    expect(root.dataset.view).toBe("list"); expect(root.classList.contains("show-filters")).toBe(false);
    root.querySelector<HTMLButtonElement>('[data-action="read-day"]')!.click();
    await vi.waitFor(() => expect(root.querySelector("article")).toBeTruthy());
    button(root, "← 返回列表").click();
    await vi.waitFor(() => expect(root.dataset.view).toBe("list"));
    expect(root.classList.contains("show-filters")).toBe(false);
  });
});


it.each([
  ["pending", "awaiting_announcement", "等待公告发布", "Awaiting announcement"],
  ["completed", "no_updates", "当日无更新", "No updates"],
  ["completed", "no_matches", "无匹配论文", "No matching papers"],
  ["completed", "papers_written", "日报已保存", "Daily report saved"],
])("renders structured %s/%s outcomes in both interface languages", async (runStatus, outcome, zh, en) => {
  for (const [language, label] of [["zh", zh], ["en", en]]) {
    const { root } = setup(url => {
      if (url.pathname.endsWith("api/preferences")) return json({ appearance: { language, theme: "light" } });
      if (url.pathname.endsWith("api/runs/current")) return json({ run: { id: language, kind: "daily", date: "2026-10-03", label: "2026-10-03", status: runStatus, outcome, output: "", startedAt: "2026-10-03T00:00:00Z", finishedAt: "2026-10-03T00:00:01Z", exitCode: 0 } });
    });
    await vi.waitFor(() => expect(root.querySelector(".run-state")?.textContent).toBe(label));
    expect(root.querySelector('[data-action="cancel-run"]')).toBeNull();
  }
});


it.each([["zh", "等待公告发布", "当日无更新", "重新检查公告"], ["en", "Awaiting announcement", "No updates", "Check announcement again"]])("shows distinct calendar availability with a recheck action in %s", async (language, waiting, empty, recheck) => {
  const { root } = setup(url => {
    if (url.pathname.endsWith("api/preferences")) return json({ appearance: { language, theme: "light" } });
    if (url.pathname.endsWith("api/calendar")) {
      const value = calendar();
      Object.assign(value.cells.find(cell => cell?.date === "2026-10-03")!, { state: "awaiting-announcement", papers: null, canGenerate: true, actionLabel: "重新检查公告" });
      Object.assign(value.cells.find(cell => cell?.date === "2026-10-04")!, { state: "no-updates", papers: 0, canGenerate: false, actionLabel: null });
      return json(value);
    }
  });
  await ready(root);
  expect(day(root, "2026-10-03").getAttribute("aria-label")).toContain(waiting);
  expect(day(root, "2026-10-04").getAttribute("aria-label")).toContain(empty);
  expect(day(root, "2026-10-04").classList.contains("failed")).toBe(false);
  day(root, "2026-10-03").click();
  expect(root.querySelector('.calendar-summary [data-action="generate-date"]')?.textContent).toContain(recheck);
});
