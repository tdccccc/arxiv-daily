import type { PersonalLibraryPaperRecord, PersonalLibraryScanSummary } from "@arxiv-daily/core";
import { t } from "./i18n";
import { h } from "./dom";
import { scientificInline } from "./papers";

export type LibraryPaper = Omit<PersonalLibraryPaperRecord, "source"> & { source: "arxiv" | "file"; pdfAvailable: boolean; score?: number };
interface LibraryPage {
  papers: LibraryPaper[];
  total: number;
  offset?: number;
  nextOffset?: number | null;
  summary?: PersonalLibraryScanSummary | null;
  connected?: boolean;
}
export interface LibraryViewOptions {
  request<T>(url: string, body?: unknown): Promise<T>;
  onSettings(): void;
  onReview?(): void;
}

export function mountLibrary(root: HTMLElement, options: LibraryViewOptions): () => void {
  let disposed = false;
  let sequence = 0;
  let currentOffset = 0;
  let nextOffset: number | null = null;
  let appliedQuery = "";
  let lastRequest: () => Promise<void>;
  const label = (source: string, ...args: Array<string | number>) => t(source, ...args);
  const button = (action: string, source: string) => h("button", { type: "button", class: "quiet-button", "data-library": action }, label(source));
  root.replaceChildren(
    h("section", { class: "library-workspace", "aria-label": label("个人文献库") },
      h("header", { class: "library-heading" },
        h("div", null, h("h1", null, label("个人文献库")), h("p", null, label("浏览本地目录记录，或按标题与摘要检索已建索引的论文。"))),
        h("div", { class: "library-actions" }, button("settings", "连接与索引"), options.onReview ? button("review", "方向审核") : null),
      ),
      h("form", { class: "library-search" },
        h("label", { class: "library-query" }, label("文献库关键词"), h("input", { type: "search", "aria-label": label("文献库关键词"), autocomplete: "off" })),
        button("browse", "筛选目录"),
        h("label", null, label("索引检索方式"),
          h("select", { "aria-label": label("索引检索方式") },
            h("option", { value: "lexical" }, label("关键词检索")),
            h("option", { value: "hybrid" }, label("混合检索")),
            h("option", { value: "dense" }, label("语义检索")),
          ),
        ),
        button("search", "检索索引"),
      ),
      h("p", { class: "library-search-note" }, label("浏览目录不调用模型；语义与混合检索使用已配置的嵌入服务。")),
      h("div", { class: "library-results", "aria-live": "polite" }),
    ),
  );
  const input = root.querySelector<HTMLInputElement>("input")!;
  const select = root.querySelector<HTMLSelectElement>("select")!;
  const results = root.querySelector<HTMLElement>(".library-results")!;

  const empty = (title: string, description: string) => h("div", { class: "list-empty" }, h("h2", null, label(title)), h("p", null, label(description)));
  function render(page: LibraryPage, searching: boolean): void {
    currentOffset = page.offset ?? 0;
    nextOffset = searching ? null : page.nextOffset ?? null;
    if (page.connected === false) { results.replaceChildren(empty("尚未连接个人文献库", "先在连接与索引中选择文献目录。")); return; }
    const summary = page.summary;
    const rows = page.papers.map(paper => h("article", { class: "paper-row library-paper" },
      h("div", { class: "paper-row-meta" }, [paper.published || t("日期未记录"), paper.categories.join(" · ") || paper.primaryCategory, paper.externalId].filter(Boolean).join(" · ")),
      h("h2", { class: "paper-title" }, scientificInline(paper.title)),
      h("p", { class: "paper-authors" }, paper.authors.join(", ")),
      h("p", { class: "library-abstract" }, paper.abstract ? scientificInline(paper.abstract) : label("暂无摘要")),
      h("div", { class: "paper-row-footer" },
        paper.pdfAvailable
          ? h("a", { href: `api/library/pdf?key=${encodeURIComponent(paper.paperKey)}`, target: "_blank", rel: "noopener noreferrer" }, `${label("阅读 PDF")} ↗`)
          : h("span", { class: "muted" }, label("本地 PDF 不可用")),
        typeof paper.score === "number" && Number.isFinite(paper.score) ? h("span", { class: "muted" }, label("相关度 {0}", paper.score.toFixed(3))) : null,
      ),
    ));
    results.replaceChildren(
      h("div", { class: "library-result-summary" },
        h("strong", null, label(searching ? "检索结果 · {0} 篇论文" : "目录 · {0} 篇论文", page.total)),
        summary ? h("span", null, label("待识别 {0} · 处理失败 {1}", summary.unresolved, summary.failed)) : null,
        summary?.truncated ? h("p", null, label("扫描达到数量上限，目录可能不完整。")) : null,
      ),
      ...(rows.length ? rows : [searching || appliedQuery ? empty("没有匹配的文献", "尝试其他关键词，或在连接与索引中检查扫描和索引状态。") : empty("文献库尚无论文", "在连接与索引中扫描目录，建立可浏览的论文目录。")]),
      ...(!searching && (currentOffset > 0 || nextOffset !== null)
        ? [h("nav", { class: "library-pagination", "aria-label": label("分页") },
            h("button", { type: "button", "data-library": "previous", disabled: currentOffset === 0 }, label("← 上一页")),
            h("span", null, `${currentOffset + 1}–${currentOffset + page.papers.length} / ${page.total}`),
            h("button", { type: "button", "data-library": "next", disabled: nextOffset === null }, label("下一页 →")),
          )]
        : []),
    );
  }
  async function load(url: string, body?: unknown): Promise<void> {
    const token = ++sequence;
    results.setAttribute("aria-busy", "true");
    results.replaceChildren(h("p", { role: "status", class: "list-empty" }, `${label("正在加载文献库")}…`));
    try {
      const page = body === undefined ? await options.request<LibraryPage>(url) : await options.request<LibraryPage>(url, body);
      if (!disposed && token === sequence) render(page, body !== undefined);
    } catch (error) {
      if (!disposed && token === sequence) results.replaceChildren(
        h("div", { role: "alert", class: "list-empty" },
          h("h2", null, label("文献库加载失败")),
          h("p", null, error instanceof Error ? error.message : String(error)),
          button("retry", "重试"),
        ),
      );
    } finally {
      if (!disposed && token === sequence) results.setAttribute("aria-busy", "false");
    }
  }
  function browse(offset: number, query: string): void {
    appliedQuery = query;
    lastRequest = () => load(`api/library?q=${encodeURIComponent(query)}&offset=${offset}&limit=20`);
    void lastRequest();
  }
  function search(): void {
    const query = input.value.trim();
    if (!query) { input.setCustomValidity(t("请输入检索内容")); input.reportValidity(); return; }
    input.setCustomValidity("");
    const body = { query, mode: select.value, limit: 20 };
    lastRequest = () => load("api/library/search", body);
    void lastRequest();
  }
  function onClick(event: MouseEvent): void {
    event.stopPropagation();
    const action = (event.target as Element).closest<HTMLElement>("[data-library]")?.dataset.library;
    if (action === "settings") options.onSettings();
    else if (action === "review") options.onReview?.();
    else if (action === "browse") browse(0, input.value.trim());
    else if (action === "search") search();
    else if (action === "previous") browse(Math.max(0, currentOffset - 20), appliedQuery);
    else if (action === "next" && nextOffset !== null) browse(nextOffset, appliedQuery);
    else if (action === "retry") void lastRequest();
  }
  const contain = (event: Event) => { event.stopPropagation(); input.setCustomValidity(""); };
  const submit = (event: Event) => { event.preventDefault(); event.stopPropagation(); browse(0, input.value.trim()); };
  root.addEventListener("click", onClick);
  root.addEventListener("input", contain);
  root.addEventListener("change", contain);
  root.addEventListener("submit", submit);
  browse(0, "");
  return () => {
    disposed = true; ++sequence;
    root.removeEventListener("click", onClick); root.removeEventListener("input", contain);
    root.removeEventListener("change", contain); root.removeEventListener("submit", submit);
  };
}
