// @vitest-environment happy-dom
import { expect, it } from "vitest";
import { overview } from "../src/workbench/web/papers";
import type { WorkbenchPaper } from "../src/workbench/papers";

const paper: WorkbenchPaper = { key: "arxiv:2609.12345", arxivId: "2609.12345", title: "Title", authors: [], published: "", topics: [], category: "", status: "inbox", priority: "normal", starred: false, abstract: "Original abstract", summary: { coreProblem: "Main argument", sourceSections: "Section 2" }, detailPath: null, reports: [], originalUrl: null, pdfUrl: null, provenance: null, novelty: null };
it("separates sources and subsequent reference content from the argument and puts generation details last", () => {
  document.body.innerHTML = overview(paper, false);
  expect(document.querySelector(".overview-content > section")?.textContent).toContain("Main argument");
  expect(document.querySelector(".reading-appendix > hr")).toBeTruthy();
  expect(document.querySelector(".reading-appendix")?.textContent).toContain("Section 2");
  expect(document.querySelector(".reading-appendix")?.textContent).toContain("Original abstract");
  expect(document.querySelector(".article-wrap")?.lastElementChild?.className).toContain("generation-footer");
  expect(document.querySelector(".generation-footer")?.textContent).toContain("未记录");
});

it("shows real daily-level tokens and both wall time and LLM time without attributing them to one paper", () => {
  document.body.innerHTML = overview({ ...paper, generation: { scope: "daily", sourcePath: "daily/2026-10-01.md", metrics: { logicalCalls: 2, attempts: 3, elapsedMs: 1800, pipelineElapsedMs: 2500, usageComplete: true, inputTokens: 1200, outputTokens: 80, totalTokens: 1280, generatedAt: "2026-10-01T19:22:33+08:00" } } }, false);
  const footer = document.querySelector(".generation-footer")!;
  expect(footer.textContent).toContain("统计范围：整份来源日报");
  const values = Object.fromEntries([...footer.querySelectorAll("dl > div")].map(row => [row.querySelector("dt")!.textContent, row.querySelector("dd")!.textContent]));
  expect(values).toMatchObject({ "输入 Token": "1200", "输出 Token": "80", "总 Token": "1280", "生成耗时": "2.5 s", "LLM 累计耗时": "1.8 s" });
  expect(footer.querySelector("time")?.getAttribute("datetime")).toBe("2026-10-01T11:22:33.000Z");
  expect(footer.querySelector("time")?.textContent).toContain("UTC");
});

it("labels partial provider usage and leaves absent total, wall time and generation timestamp unrecorded", () => {
  document.body.innerHTML = overview({ ...paper, generation: { scope: "paper", sourcePath: "papers/detail.md", metrics: { logicalCalls: 1, attempts: 1, elapsedMs: 100, usageComplete: false, inputTokens: 22 } } }, false);
  const footer = document.querySelector(".generation-footer")!;
  const values = Object.fromEntries([...footer.querySelectorAll("dl > div")].map(row => [row.querySelector("dt")!.textContent, row.querySelector("dd")!.textContent]));
  expect(values).toMatchObject({ "输入 Token": "22", "输出 Token": "未记录", "总 Token": "未记录", "生成耗时": "未记录", "生成时间": "未记录", "LLM 累计耗时": "100 ms" });
  expect(footer.textContent).toContain("Token 用量记录不完整");
  expect(footer.querySelector("time")).toBeNull();
});
