// @vitest-environment happy-dom
import { afterEach, expect, it } from "vitest";
import { mountCalendar } from "../src/workbench/web/calendar";
import { mountSidebar } from "../src/workbench/web/sidebar";
import { dayHeading, overview, paperRows } from "../src/workbench/web/papers";
import type { WorkbenchPaper, WorkbenchPaperList } from "../src/workbench/papers";
import type { WorkbenchCalendarDay } from "../src/workbench/calendar";
import { setUiLanguage } from "../src/workbench/web/i18n";

const disposers: Array<() => void> = [];
afterEach(() => { disposers.splice(0).forEach(dispose => dispose()); setUiLanguage("zh"); document.body.innerHTML = ""; });
const paper = { key: "arxiv:2609.12345", arxivId: "2609.12345", title: "核心问题", authors: ["原文"], published: "2026-10-01", topics: ["待读"], category: "cs.AI", status: "inbox", priority: "normal", starred: false, abstract: "原始摘要", summary: { coreProblem: "主要结果", whyRelevant: "推荐理由" }, detailPath: null, reports: [], originalUrl: "https://arxiv.org/abs/2609.12345", pdfUrl: "https://arxiv.org/pdf/2609.12345", provenance: null, novelty: null } as WorkbenchPaper;
const day: WorkbenchCalendarDay = { date: "2026-10-01", state: "has-report", reportPath: "daily/report.md", reportTitle: "论文概览", papers: 3, message: "日报已保存，可以打开阅读。", canGenerate: false, actionLabel: null };
function host() { const root = document.createElement("div"); document.body.append(root); return root; }

it("renders English calendar controls, weekdays, states and paper counts", async () => {
  setUiLanguage("en");
  const root = host();
  const calendar = mountCalendar(root, { request: async <T>() => ({ month: "2026-10", today: day.date, timezone: "Asia/Shanghai", previousMonth: "2026-09", nextMonth: "2026-11", cells: [day] }) as T, selectDay: () => {}, updateDay: () => {}, onError: () => {} });
  disposers.push(calendar.dispose); await calendar.load();
  expect(root.querySelector('[data-calendar-action="previous"]')?.getAttribute("aria-label")).toBe("Previous month");
  expect(root.querySelector(".calendar-weekdays")?.textContent).toBe("MonTueWedThuFriSatSun");
  expect(root.querySelector(".calendar-month")?.textContent).toBe("2026-10");
  expect(root.querySelector('[data-calendar-date]')?.getAttribute("aria-label")).toContain("Report saved, 3 papers");
  expect(root.querySelector(".calendar-summary")?.textContent).toContain("3 papers");
  expect(root.textContent).not.toMatch(/[\u4e00-\u9fff]/);
});

it("translates paper controls while preserving user titles, topics, authors and abstracts", () => {
  setUiLanguage("en");
  const root = host(); root.innerHTML = overview(paper, false);
  expect(root.querySelector(".reading-kind")?.textContent).toBe("Paper overview");
  expect(root.querySelector(".document-title")?.textContent).toBe("核心问题");
  expect(root.querySelector(".document-authors")?.textContent).toBe("原文");
  expect(root.querySelector(".overview-content h2")?.textContent).toBe("Core problem");
  expect(root.querySelector(".overview-content p")?.textContent).toBe("主要结果");
  expect(root.querySelector('[data-mark="status"]')?.getAttribute("aria-label")).toBe("Reading status · 核心问题");
  expect(root.querySelector('[data-source="pdf"]')?.textContent).toBe("Read PDF ↗");
  root.innerHTML = paperRows({ papers: [paper] } as WorkbenchPaperList, new Set());
  expect(root.querySelector(".detail-availability")?.textContent).toBe("No detailed summary yet");
  expect(root.querySelector(".paper-topics")?.textContent).toContain("待读");
  setUiLanguage("zh"); root.innerHTML = overview(paper, false);
  expect(root.querySelector(".reading-kind")?.textContent).toBe("论文概览");
});

it("translates known day messages without altering raw provider diagnostics", () => {
  setUiLanguage("en"); const root = host(); root.innerHTML = dayHeading(day);
  expect(root.textContent).toContain("The daily report is saved and ready to read.");
  expect(root.textContent).toContain("Read full daily report");
  const diagnostic = "provider: 模型 unavailable <retry>";
  root.innerHTML = dayHeading({ ...day, state: "failed", message: diagnostic });
  expect(root.querySelector("p")?.textContent).toBe(diagnostic);
});

it("translates sidebar controls and only submits layout preferences", async () => {
  setUiLanguage("en"); const root = host();
  root.innerHTML = '<header class="header-actions"></header><div class="workspace"><aside class="library-pane"></aside></div>';
  const submissions: unknown[] = [];
  disposers.push(mountSidebar(root, { request: async <T>(_url: string, body?: unknown) => { if (body) submissions.push(body); return { sidebarWidth: 420, sidebarCollapsed: false, theme: "dark", language: "en" } as T; } }));
  await Promise.resolve();
  expect(root.querySelector(".sidebar-resize")?.getAttribute("aria-label")).toBe("Resize sidebar");
  expect(root.querySelector(".sidebar-toggle")?.textContent).toBe("Collapse sidebar");
  root.querySelector<HTMLButtonElement>(".sidebar-toggle")!.click();
  expect(submissions).toEqual([{ sidebarWidth: 420, sidebarCollapsed: true }]);
});
