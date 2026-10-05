// @vitest-environment happy-dom
import { afterEach, expect, it, vi } from "vitest";
import { setUiLanguage } from "../src/workbench/web/i18n";
import { mountLibrary, type LibraryPaper } from "../src/workbench/web/library";

const paper: LibraryPaper = { paperKey: "arxiv:2601.00001", source: "arxiv", externalId: "2601.00001", title: "A $x^2$ result", authors: ["A. Author"], abstract: "We measure $H_0$.", published: "2026-01-01", updated: "2026-01-01", primaryCategory: "astro-ph.CO", categories: ["astro-ph.CO"], evidenceDepth: "metadata-and-abstract", filePaths: ["private/file.pdf"], pdfAvailable: true };
const page = (papers = [paper], extra = {}) => ({ papers, total: papers.length, offset: 0, nextOffset: null, summary: null, connected: true, ...extra });
const settle = async () => { await new Promise(resolve => setTimeout(resolve, 0)); };
const disposers: Array<() => void> = [];
afterEach(() => { setUiLanguage("zh"); disposers.splice(0).forEach(dispose => dispose()); document.body.innerHTML = ""; });
function setup(request: <T>(url: string, body?: unknown) => Promise<T>, onSettings = vi.fn()) {
  const root = document.createElement("div"); document.body.append(root);
  disposers.push(mountLibrary(root, { request, onSettings }));
  return root;
}
function click(root: HTMLElement, action: string) { (root.querySelector(`[data-library="${action}"]`) as HTMLButtonElement).click(); }

it("browses the catalog without invoking retrieval, renders scientific text and opens a scoped PDF", async () => {
  const request = vi.fn().mockResolvedValue(page());
  const root = setup(request);
  expect(root.textContent).toContain("正在加载文献库");
  await settle();
  expect(request).toHaveBeenCalledOnce();
  expect(request.mock.calls[0][0]).toBe("api/library?q=&offset=0&limit=20");
  expect(root.querySelectorAll(".katex").length).toBe(2);
  expect(root.textContent).toContain("A. Author");
  expect(root.textContent).toContain("astro-ph.CO");
  const link = root.querySelector("a")!;
  expect(link.getAttribute("href")).toBe("api/library/pdf?key=arxiv%3A2601.00001");
  expect(link.getAttribute("target")).toBe("_blank");
  expect(root.innerHTML).not.toContain("private/file.pdf");
});

it("filters the catalog, paginates and searches only after explicit retrieval, containing input events", async () => {
  const request = vi.fn().mockResolvedValue(page([paper], { total: 21, nextOffset: 20 }));
  const root = setup(request); await settle();
  const bubbled = vi.fn(); document.body.addEventListener("input", bubbled);
  const input = root.querySelector("input")!; input.value = "galaxy"; input.dispatchEvent(new Event("input", { bubbles: true }));
  await settle();
  expect(bubbled).not.toHaveBeenCalled();
  click(root, "browse"); await settle();
  expect(request).toHaveBeenLastCalledWith("api/library?q=galaxy&offset=0&limit=20");
  click(root, "next"); await settle();
  expect(request).toHaveBeenLastCalledWith("api/library?q=galaxy&offset=20&limit=20");
  const mode = root.querySelector("select")!; mode.value = "hybrid";
  click(root, "search"); await settle();
  expect(request).toHaveBeenLastCalledWith("api/library/search", { query: "galaxy", mode: "hybrid", limit: 20 });
  document.body.removeEventListener("input", bubbled);
});

it("distinguishes disconnected and empty libraries and provides connection settings", async () => {
  const request = vi.fn().mockResolvedValueOnce(page([], { connected: false })).mockResolvedValue(page([]));
  const onSettings = vi.fn(); const root = setup(request, onSettings); await settle();
  expect(root.textContent).toContain("尚未连接个人文献库");
  click(root, "settings"); expect(onSettings).toHaveBeenCalledOnce();
  click(root, "browse"); await settle();
  expect(root.textContent).toContain("文献库尚无论文");
});

it("shows retryable errors and ignores outdated responses and disposed views", async () => {
  let resolveOld!: (value: unknown) => void;
  const request = vi.fn().mockImplementationOnce(() => new Promise(resolve => { resolveOld = resolve; })).mockRejectedValueOnce(new Error("index unavailable")).mockResolvedValue(page([{ ...paper, title: "Latest", pdfAvailable: false }]));
  const root = setup(request);
  click(root, "browse"); await settle();
  expect(root.querySelector('[role="alert"]')?.textContent).toContain("index unavailable");
  click(root, "retry"); await settle();
  resolveOld(page([{ ...paper, title: "Stale" }])); await settle();
  expect(root.textContent).toContain("Latest"); expect(root.textContent).not.toContain("Stale");
  expect(root.querySelector("a")).toBeNull();
  disposers.pop()!(); root.innerHTML = "Elsewhere";
  clickAfterDispose(root);
  expect(root.textContent).toBe("Elsewhere");
});
function clickAfterDispose(root: HTMLElement) { root.dispatchEvent(new MouseEvent("click", { bubbles: true })); }

it("requires a query for index retrieval and safely renders untrusted metadata", async () => {
  const request = vi.fn().mockResolvedValue(page([{ ...paper, title: '<img src=x onerror="alert(1)">', abstract: "<script>alert(1)</script>", authors: ["<img src=x>"], categories: ["<script>bad</script>"] }]));
  const root = setup(request); await settle();
  click(root, "search"); await settle();
  expect(request).toHaveBeenCalledOnce();
  expect(root.querySelector("input")!.validationMessage).toContain("请输入检索内容");
  expect(root.querySelector("script, img")).toBeNull();
  expect(root.textContent).toContain("<img src=x>");
});

it("does not apply a pending request after disposal and removes its event listeners", async () => {
  let resolve!: (value: unknown) => void;
  const request = vi.fn().mockImplementation(() => new Promise(done => { resolve = done; }));
  const onSettings = vi.fn(); const root = setup(request, onSettings);
  disposers.pop()!();
  click(root, "settings"); expect(onSettings).not.toHaveBeenCalled();
  root.innerHTML = "Another view";
  resolve(page()); await settle();
  expect(root.innerHTML).toBe("Another view");
});

it("preserves the catalog query during pagination and retries the selected retrieval mode", async () => {
  const request = vi.fn().mockResolvedValueOnce(page([paper], { total: 21, nextOffset: 20 }))
    .mockResolvedValueOnce(page([paper], { total: 21, offset: 20, nextOffset: null }))
    .mockResolvedValueOnce(page([paper], { total: 21, nextOffset: 20 }))
    .mockRejectedValueOnce(new Error("Temporary embedding error"))
    .mockResolvedValue(page([]));
  const root = setup(request); await settle();
  root.querySelector("input")!.value = "unapplied edit";
  click(root, "next"); await settle();
  expect(request).toHaveBeenLastCalledWith("api/library?q=&offset=20&limit=20");
  click(root, "previous"); await settle();
  expect(request).toHaveBeenLastCalledWith("api/library?q=&offset=0&limit=20");
  root.querySelector("select")!.value = "dense";
  click(root, "search"); await settle();
  click(root, "retry"); await settle();
  expect(request).toHaveBeenLastCalledWith("api/library/search", { query: "unapplied edit", mode: "dense", limit: 20 });
  expect(root.textContent).toContain("没有匹配的文献");
  expect(root.querySelector('[data-library="next"]')).toBeNull();
});


it("uses shared English messages including pagination and keeps research text untouched", async () => {
  setUiLanguage("en");
  const root = setup(vi.fn().mockResolvedValue(page([{ ...paper, title: "星系研究" }], { total: 21, nextOffset: 20 })));
  await settle();
  expect(root.textContent).toContain("Personal library");
  expect(root.textContent).toContain("Connection and indexing");
  expect(root.textContent).toContain("Search index");
  expect(root.textContent).toContain("Catalog · 21 papers");
  expect(root.querySelector('[data-library="next"]')?.textContent).toContain("Next");
  expect(root.textContent).toContain("星系研究");
});

it("supports indexed local files without inventing arXiv metadata", async () => {
  const local: LibraryPaper = { ...paper, source: "file", paperKey: "file:abcdef", title: "Local report", externalId: "", published: "", authors: [], primaryCategory: "", categories: [] };
  const root = setup(vi.fn().mockResolvedValue(page([local]))); await settle();
  expect(root.querySelector("a")?.getAttribute("href")).toBe("api/library/pdf?key=file%3Aabcdef");
  expect(root.querySelector(".paper-row-meta")?.textContent).not.toContain("·");
});
