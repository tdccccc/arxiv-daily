import { t, getUiLanguage, setUiLanguage } from "./i18n";
import { isCompletedDiscovery, DEFAULT_UI_APPEARANCE, normalizeUiAppearancePreferences, type UiAppearancePreferences } from "@arxiv-daily/core";
import { settingsForm, bindSettings, type SettingsSnapshot } from "./settings";
import type { DocumentEntry, WorkbenchDocuments } from "../documents";
import type { WorkbenchRun } from "../server";
import type { inspectProduct } from "../../inspect-cmd";
import { generationFooter } from "./generation-footer";
import { mountCalendar } from "./calendar";
import { mountSidebar } from "./sidebar";
import type { WorkbenchPaper, WorkbenchPaperList, PaperScope } from "../papers";
import { scopes, marks, paperRows, dayHeading, overview, scientificInline } from "./papers";

export interface WorkbenchClientOptions {
  fetch?: typeof fetch;
  searchDelayMs?: number;
  pollIntervalMs?: number;
  appearance?: UiAppearancePreferences;
}

type ProductStatus = Awaited<ReturnType<typeof inspectProduct>>;
type ReadingDocument = Awaited<ReturnType<WorkbenchDocuments["document"]>>;
interface DocumentList { documents: DocumentEntry[]; total: number; nextOffset: number | null; counts: { daily: number; papers: number } }

const symbols = {
  search: '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="8.5" cy="8.5" r="5.5"/><path d="m13 13 4 4"/></svg>',
  arrow: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 15 15 5M5 5h10v10"/></svg>',
};

export function mountWorkbench(root: HTMLElement, options: WorkbenchClientOptions = {}): () => void {
  const fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
  const lifetime = new AbortController();
  let disposed = false, generation = 0;
  let disposeContent: (() => void) | undefined;
  let restoreScroll: MutationObserver | undefined;
  let appearance: UiAppearancePreferences = options.appearance ?? { ...DEFAULT_UI_APPEARANCE, language: getUiLanguage() };
  const navigation = { session: `${Date.now()}-${Math.random()}`, index: 0, end: 0, scrolls: new Map<number, number>(), pending: false };
  function render(next: UiAppearancePreferences) {
    if (disposed) return;
    const scroll = root.querySelector<HTMLElement>('.reading-pane')?.scrollTop ?? 0;
    generation += 1;
    restoreScroll?.disconnect(); disposeContent?.();
    appearance = next; setUiLanguage(next.language);
    document.documentElement.lang = next.language === 'zh' ? 'zh-CN' : 'en';
    document.title = t('arxiv-daily · 阅读工作台');
    disposeContent = mountWorkbenchContent(root, { ...options, appearance: next }, navigation, async value => {
      if (value.language !== appearance.language || value.theme !== appearance.theme) render(value);
    });
    if (scroll > 0) {
      restoreScroll = new MutationObserver(() => {
        if (!root.querySelector('.markdown-body, .paper-workspace, .document-list')) return;
        const pane = root.querySelector<HTMLElement>('.reading-pane'); if (pane) pane.scrollTop = scroll;
        restoreScroll?.disconnect();
      });
      restoreScroll.observe(root, { childList: true, subtree: true });
    }
  }
  render(appearance);
  const initialGeneration = generation;
  if (!options.appearance) void fetcher('api/preferences', { signal: lifetime.signal }).then(async response => {
    if (!response.ok) return;
    const value = await response.json() as { appearance?: UiAppearancePreferences };
    if (disposed || generation !== initialGeneration || !value.appearance) return;
    const saved = normalizeUiAppearancePreferences(value.appearance);
    if (saved.language !== appearance.language || saved.theme !== appearance.theme) render(saved);
  }).catch(() => { /* Sidebar and settings provide retry UI for unavailable preferences. */ });
  return () => { disposed = true; lifetime.abort(); restoreScroll?.disconnect(); disposeContent?.(); };
}

interface ReadingNavigation { session: string; index: number; end: number; scrolls: Map<number, number>; pending: boolean }

function mountWorkbenchContent(root: HTMLElement, options: WorkbenchClientOptions, navigation: ReadingNavigation, onAppearanceSaved: (value: UiAppearancePreferences) => Promise<void>): () => void {
  const fetcher = options.fetch ?? globalThis.fetch.bind(globalThis);
  const lifetime = new AbortController();
  let disposed = false;
  let status: ProductStatus | null = null;
  let setupRequired = false;
  let kind: "all" | "daily" | "papers" = "all";
  let query = "", scope: PaperScope = "all", topic = "", sort = "published", direction = "desc", offset = 0;
  let documentsMode = false;
  let entries: DocumentEntry[] = [];
  let nextOffset: number | null = null;
  let paperList: WorkbenchPaperList | null = null;
  let activePaper: WorkbenchPaper | null = null;
  let selectedKey = "";
  let selectedPath = "";
  let selectedDate = "";
  let listScroll = 0;
  let readingReady = false;
  const scrollPositions = new Map<string, number>();
  const pendingMarks = new Set<string>();
  let listVersion = 0;
  let documentVersion = 0;
  let runVersion = 0;
  let currentRun: WorkbenchRun | null = null;
  let dismissedRun = "";
  let searchTimer: ReturnType<typeof setTimeout> | undefined;
  let pollTimer: ReturnType<typeof setTimeout> | undefined;
  let returnFocus: HTMLElement | null = null;
  let dialog: HTMLDialogElement | null = null;
  let fontSize = Math.max(14, Math.min(22, Number(preference("font-size")) || 17));

  root.className = "workbench";
  root.dataset.view = "list";
  const appearance = options.appearance ?? DEFAULT_UI_APPEARANCE;
  const systemTheme = window.matchMedia?.('(prefers-color-scheme: dark)');
  const applyTheme = () => { root.dataset.theme = appearance.theme === 'system' ? (systemTheme?.matches ? 'dark' : 'light') : appearance.theme; };
  applyTheme(); systemTheme?.addEventListener?.('change', applyTheme);
  root.style.setProperty("--reading-size", `${fontSize}px`);
  root.innerHTML = `
    <a class="skip-link" href="#reading-content">${t("跳到正文")}</a>
    <header class="app-header">
      <a class="brand" href="./" aria-label="${t("arxiv-daily 首页")}">arxiv<span class="brand-hyphen">-</span>daily</a>
      <div class="header-actions"><nav class="reading-history" aria-label="${t("阅读历史")}"><button class="quiet-button" data-action="history-back" aria-label="${t("后退")}" title="${t("后退")}" disabled>← <span>${t("后退")}</span></button><button class="quiet-button" data-action="history-forward" aria-label="${t("前进")}" title="${t("前进")}" disabled><span>${t("前进")}</span> →</button></nav><button class="quiet-button" data-action="settings">${t("设置")}</button><button class="primary-button" data-action="generate"><span aria-hidden="true">＋</span> ${t("生成")}</button></div>
    </header>
    <div class="connection-banner" role="alert" hidden></div>
    <div class="workspace">
      <aside class="library-pane" aria-label="${t("日历与筛选")}">
        <div class="library-heading"><span>${t("我的阅读")}</span><button class="quiet-button show-filters" data-action="show-filters">${t("返回列表")}</button><button class="icon-button" data-action="refresh" aria-label="${t("刷新文档")}">↻</button></div>
        <section class="calendar-panel" aria-label="${t("日报日历")}"></section>
        <label class="search-box">${symbols.search}<input type="search" aria-label="${t("搜索标题、作者、arXiv ID 或日期")}" placeholder="${t("搜索标题、作者或关键词")}" autocomplete="off"></label>
        <nav class="paper-scopes" aria-label="${t("阅读筛选")}">${Object.entries(scopes).map(([key, label]) => `<button class="scope-button" data-scope="${key}"><span>${t(label)}</span><span data-count="${key}">—</span></button>`).join("")}</nav>
        <label class="topic-filter">${t("Topic")}<select data-filter="topic" aria-label="${t("筛选主题")}"><option value="">${t("全部主题")}</option></select></label>
        <div class="navigation-footer"><button class="quiet-button" data-action="clear-date">${t("浏览全部日期")}</button><button class="quiet-button" data-action="browse-documents">${t("浏览 Markdown 文件 ↗")}</button></div>
      </aside>
      <main class="reading-pane" id="reading-content" tabindex="-1"></main>
      <aside class="toc-pane" aria-label="${t("文章目录")}"></aside>
    </div>
    <section class="run-tray" aria-label="${t("生成任务")}" hidden></section>
    <div class="dialog-host"></div>`;

  const find = <T extends HTMLElement = HTMLElement>(selector: string) => root.querySelector<T>(selector)!;
  const reading = find(".reading-pane");
  const calendar = mountCalendar(find(".calendar-panel"), {
    request,
    selectDay: day => { selectedDate = day.date; documentsMode = false; offset = 0; void showList(true); },
    updateDay: day => {
      if (selectedDate === day.date && paperList && !documentsMode) {
        paperList.day = day;
        const header = reading.querySelector(".day-reading");
        if (root.dataset.view === "list" && header) header.outerHTML = dayHeading(day);
      }
    },
    onError: reportConnection,
  });

  const disposeSidebar = mountSidebar(root, { request });

  async function request<T>(url: string, body?: unknown): Promise<T> {
    try {
      const response = await fetcher(url, {
        signal: lifetime.signal,
        ...(body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
      });
      const value = await response.json() as T & { error?: string };
      if (!response.ok) throw new Error(value.error ? t(value.error) : t("请求失败（{0}）", response.status));
      return value;
    } catch (error) {
      if (error instanceof TypeError) throw new Error(t("无法连接本地工作台。请确认启动它的终端仍在运行，再重新连接。"));
      throw error;
    }
  }

  function reportConnection(error: unknown): void {
    if (disposed) return;
    const banner = find(".connection-banner");
    banner.innerHTML = `<span>${escapeHtml(message(error))}</span><button class="quiet-button" data-action="reconnect">${t("重新连接")}</button>`;
    banner.hidden = false;
  }

  async function loadStatus(): Promise<boolean> {
    try {
      const result = await request<ProductStatus | { setupRequired: true }>("api/status");
      if (disposed) return false;
      setupRequired = "setupRequired" in result && result.setupRequired === true;
      status = setupRequired ? null : result as ProductStatus;
      return true;
    } catch (error) { reportConnection(error); return false; }
  }

  function syncMarkValues(): void {
    for (const select of Array.from(root.querySelectorAll<HTMLSelectElement>('[data-mark="status"]'))) {
      const key = select.closest<HTMLElement>("[data-key]")?.dataset.key;
      const paper = activePaper?.key === key ? activePaper : paperList?.papers.find(item => item.key === key);
      if (paper) select.value = paper.status;
    }
  }
  function listIdentity(): string { return JSON.stringify([documentsMode, kind, selectedDate, query, scope, topic, sort, direction, offset]); }
  function syncNavigation(): void {
    find<HTMLButtonElement>('[data-action="history-back"]').disabled = navigation.pending || navigation.index === 0;
    find<HTMLButtonElement>('[data-action="history-forward"]').disabled = navigation.pending || navigation.index === navigation.end;
  }
  function rememberReadingScroll(): void { if (readingReady) navigation.scrolls.set(navigation.index, reading.scrollTop); }
  function navigateHistory(delta: number): void {
    if (navigation.pending || navigation.index + delta < 0 || navigation.index + delta > navigation.end) return;
    rememberReadingScroll(); navigation.pending = true; syncNavigation(); history.go(delta);
  }
  function writeRoute(url: URL, mode: "push" | "replace"): void {
    if (mode === "push") {
      navigation.index += 1; navigation.end = navigation.index;
      for (const key of navigation.scrolls.keys()) if (key >= navigation.index) navigation.scrolls.delete(key);
    }
    history[mode === "push" ? "pushState" : "replaceState"]({ ...history.state, listScroll, readingNavigation: { session: navigation.session, index: navigation.index } }, "", url);
    syncNavigation();
  }
  function rememberScroll(): void {
    rememberReadingScroll();
    if (root.dataset.view !== "list") return;
    listScroll = reading.scrollTop;
    scrollPositions.set(listIdentity(), listScroll);
    history.replaceState({ ...history.state, listScroll }, "");
  }
  function route(mode: "push" | "replace" = "push"): void {
    const url = new URL(location.href);
    const values = { q: query, scope: scope === "all" ? "" : scope, topic, sort, direction, offset: offset ? String(offset) : "", date: selectedDate, files: documentsMode ? "1" : "", kind: documentsMode ? kind : "", document: selectedPath, paper: selectedKey };
    for (const [key, value] of Object.entries(values)) { if (value) url.searchParams.set(key, value); else url.searchParams.delete(key); }
    url.hash = "";
    writeRoute(url, mode);
  }
  function readRoute(): void {
    const params = new URL(location.href).searchParams;
    query = params.get("q") || ""; scope = Object.hasOwn(scopes, params.get("scope") || "") ? params.get("scope") as PaperScope : "all";
    topic = params.get("topic") || ""; sort = params.get("sort") || "published"; direction = params.get("direction") || "desc";
    offset = Math.max(0, Number(params.get("offset")) || 0); selectedDate = routeDate();
    documentsMode = params.get("files") === "1"; kind = params.get("kind") === "daily" ? "daily" : params.get("kind") === "papers" ? "papers" : "all";
    selectedPath = params.get("document") || ""; selectedKey = params.get("paper") || "";
    listScroll = scrollPositions.get(listIdentity()) ?? history.state?.listScroll ?? 0;
  }
  function syncFilters(): void {
    find<HTMLInputElement>('input[type="search"]').value = query;
    for (const button of Array.from(root.querySelectorAll<HTMLElement>("[data-scope]"))) button.setAttribute("aria-pressed", String(button.dataset.scope === scope && !documentsMode));
    find<HTMLSelectElement>('[data-filter="topic"]').value = topic;
    find<HTMLSelectElement>('[data-filter="topic"]').disabled = documentsMode;
  }
  function listToolbar(): string {
    return `<div class="paper-list-heading"><div><span class="day-eyebrow">${documentsMode ? t("本地研究记录") : selectedDate || t("我的文献")}</span><h1>${documentsMode ? t("Markdown 文件") : t(scopes[scope])}</h1></div><button class="quiet-button show-filters" data-action="show-filters">${t("日历与筛选")}</button></div>`;
  }
  function renderList(): void {
    if (root.dataset.view !== "list" || !paperList) return;
    const unknown = paperList.total === 0 && paperList.day?.reportPath && paperList.day.papers !== 0 && !query && !topic && scope === "all";
    reading.innerHTML = `<div class="paper-workspace">${listToolbar()}${dayHeading(paperList.day)}<div class="paper-list-tools"><span class="list-caption" aria-live="polite">${unknown ? t("日报已保存，论文索引尚无可用条目") : t("{0} 篇论文", paperList.total)}</span><label>${t("排序")} <select data-filter="sort" aria-label="${t("论文排序")}">${Object.entries({ published: t("发表日期"), title: t("标题"), priority: t("优先级"), relevance: t("相关度") }).map(([value, label]) => `<option value="${value}" ${sort === value ? "selected" : ""}>${t(label)}</option>`).join("")}</select></label><button class="quiet-button" data-action="direction" aria-label="${t("切换排序方向")}">${direction === "asc" ? t("↑ 升序") : t("↓ 降序")}</button></div><div class="paper-list">${paperRows(paperList, pendingMarks) || `<div class="list-empty"><h2>${unknown ? t("可直接阅读完整日报") : t("没有匹配的论文")}</h2><p>${unknown ? t("尚未找到对应的论文索引；原始 Markdown 仍可阅读。") : t("可调整日期或筛选条件，也可通过“生成”获取新论文。")}</p></div>`}</div>${pagination(paperList.total, paperList.nextOffset)}</div>`;
    for (const [key, count] of Object.entries(paperList.counts)) find(`[data-count="${key}"]`).textContent = String(count);
    find<HTMLSelectElement>('[data-filter="topic"]').innerHTML = `<option value="">${t("全部主题")}</option>${[...new Set([...paperList.topics, ...(topic ? [topic] : [])])].map(value => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
    syncFilters(); syncMarkValues();
  }
  function pagination(total: number, next: number | null): string {
    return `<div class="paper-pagination"><button class="quiet-button" data-action="previous-page" ${offset === 0 ? "disabled" : ""}>${t("← 上一页")}</button><span>${t("第 {0} 页", Math.floor(offset / 20) + 1)}${total ? ` / ${Math.max(1, Math.ceil(total / 20))}` : ""}</span><button class="quiet-button" data-action="next-page" ${next === null ? "disabled" : ""}>${t("下一页 →")}</button></div>`;
  }
  async function loadList(restore = false): Promise<void> {
    readingReady = false;
    const version = ++listVersion;
    const scroll = restore ? listScroll : 0;
    reading.innerHTML = `<div class="reading-loading" role="status">${t("正在读取列表…")}</div>`;
    try {
      const params = new URLSearchParams({ q: query, offset: String(offset), limit: "20" });
      if (documentsMode) {
        params.set("kind", kind);
        const result = await request<DocumentList>(`api/documents?${params}`);
        if (disposed || version !== listVersion || root.dataset.view !== "list") return;
        entries = result.documents; nextOffset = result.nextOffset;
        reading.innerHTML = `<div class="paper-workspace">${listToolbar()}<div class="collection-tabs" role="tablist" aria-label="${t("文档类型")}">${Object.entries({ all: t("全部文件"), daily: t("日报"), papers: t("论文总结") }).map(([value, label]) => `<button role="tab" data-kind="${value}" aria-selected="${kind === value}">${t(label)}</button>`).join("")}</div><div class="document-list">${entries.map(entry => `<button class="document-row" data-document="${escapeHtml(entry.path)}"><span class="document-row-meta">${escapeHtml(entry.date || t("已保存文档"))}</span><span class="document-row-title">${scientificInline(entry.title)}</span><span class="document-row-authors">${escapeHtml(entry.authors)}</span></button>`).join("") || `<div class="list-empty">${t("暂无匹配文件")}</div>`}</div>${pagination(result.total, result.nextOffset)}</div>`;
      } else {
        for (const [key, value] of Object.entries({ scope, topic, sort, direction, date: selectedDate })) if (value) params.set(key, value);
        const result = await request<WorkbenchPaperList>(`api/papers?${params}`);
        if (disposed || version !== listVersion || root.dataset.view !== "list") return;
        paperList = result; nextOffset = result.nextOffset; renderList();
      }
      reading.scrollTop = scroll; readingReady = true;
    } catch (error) {
      if (disposed || version !== listVersion || root.dataset.view !== "list") return;
      reading.innerHTML = `<div class="empty-reading"><h1>${t("列表暂时不可用")}</h1><p>${escapeHtml(message(error))}</p><button class="quiet-button" data-action="refresh">${t("重试")}</button><button class="quiet-button" data-action="browse-documents">${t("浏览 Markdown 文件")}</button></div>`;
      readingReady = true; reportConnection(error);
    }
  }
  async function showList(push = false, restore = false, keepFilters = false): Promise<void> {
    if (push) rememberScroll();
    documentVersion += 1; selectedPath = ""; selectedKey = ""; activePaper = null;
    root.dataset.view = "list"; root.classList.remove("is-reading");
    if (!keepFilters) root.classList.remove("show-filters");
    find(".toc-pane").innerHTML = "";
    if (push) route();
    syncFilters();
    await loadList(restore);
  }
  function beginReading(captureScroll: boolean): void {
    if (captureScroll) rememberScroll(); readingReady = false; listVersion += 1;
    root.dataset.view = "reading"; root.classList.add("is-reading"); root.classList.remove("show-filters");
    find(".toc-pane").innerHTML = "";
  }
  async function openDocument(path: string, historyMode: "push" | "replace" | "none" = "push", hash = "", syncCalendar = true): Promise<void> {
    beginReading(historyMode !== "none");
    const version = ++documentVersion;
    selectedPath = path; selectedKey = ""; activePaper = null;
    if (historyMode !== "none") { route(historyMode); if (hash) { const url = new URL(location.href); url.hash = hash; history.replaceState(history.state, "", url); } }
    reading.innerHTML = `<div class="reading-loading" role="status">${t("正在打开文档…")}</div>`;
    try {
      const result = await request<ReadingDocument>(`api/document?path=${encodeURIComponent(path)}`);
      if (disposed || version !== documentVersion) return;
      renderDocument(result);
      if (syncCalendar && result.kind === "daily" && /^\d{4}-\d{2}-\d{2}$/.test(result.date)) void calendar.selectDate(result.date);
      if (hash) scrollToHash(hash); else reading.scrollTop = 0;
      readingReady = true;
    } catch (error) {
      if (disposed || version !== documentVersion) return;
      reading.innerHTML = `<div class="empty-reading"><button class="quiet-button" data-action="back">${t("← 返回列表")}</button><h1>${t("暂时无法打开文档")}</h1><p>${escapeHtml(message(error))}</p><button class="primary-button" data-action="retry-document">${t("重试读取")}</button></div>`;
      readingReady = true;
      if (message(error).includes("无法连接") || message(error).includes("Cannot connect")) reportConnection(error);
    }
  }
  async function openPaper(key: string, push = true, preserveScroll = false): Promise<void> {
    const scroll = reading.scrollTop;
    beginReading(push); const version = ++documentVersion;
    selectedKey = key; selectedPath = "";
    if (push) route();
    reading.innerHTML = `<div class="reading-loading" role="status">${t("正在打开论文…")}</div>`;
    try {
      const result = await request<{ paper: WorkbenchPaper }>(`api/paper?key=${encodeURIComponent(key)}`);
      if (disposed || version !== documentVersion) return;
      activePaper = result.paper; reading.innerHTML = overview(result.paper, pendingMarks.has(key)); syncMarkValues(); reading.scrollTop = preserveScroll ? scroll : 0; readingReady = true;
    } catch (error) {
      if (disposed || version !== documentVersion) return;
      readingReady = true;
      reading.innerHTML = `<div class="empty-reading"><button class="quiet-button" data-action="back">${t("← 返回列表")}</button><h1>${t("论文暂时不可用")}</h1><p>${escapeHtml(message(error))}</p><button class="quiet-button" data-action="retry-paper">${t("重试")}</button></div>`;
    }
  }
  async function saveMark(key: string, action: "status" | "star", value: string | boolean): Promise<void> {
    const paper = activePaper?.key === key ? activePaper : paperList?.papers.find(item => item.key === key);
    if (!paper || pendingMarks.has(key)) return;
    const viewVersion = documentVersion;
    pendingMarks.add(key); listVersion += 1;
    function renderMarks(current: WorkbenchPaper): void {
      if (activePaper?.key === key) activePaper = current;
      if (paperList) paperList.papers = paperList.papers.map(item => item.key === key ? current : item);
      for (const container of Array.from(root.querySelectorAll<HTMLElement>(".paper-marks"))) if (container.dataset.key === key) container.outerHTML = marks(current, pendingMarks.has(key));
      syncMarkValues();
    }
    renderMarks(paper);
    try {
      const result = await request<{ paper: WorkbenchPaper }>("api/paper/mark", { key, action, value, expected: action === "status" ? paper.status : paper.priority });
      if (disposed) return;
      pendingMarks.delete(key); renderMarks(result.paper);
      if (activePaper?.key === key) activePaper = result.paper;
    } catch (error) {
      if (disposed) return;
      reportConnection(error);
      try {
        const result = await request<{ paper: WorkbenchPaper }>(`api/paper?key=${encodeURIComponent(key)}`);
        if (!disposed) { pendingMarks.delete(key); renderMarks(result.paper); if (activePaper?.key === key) activePaper = result.paper; }
      } catch { /* Keep the error visible; a fresh list below retries persisted state. */ }
    } finally {
      pendingMarks.delete(key);
      if (!disposed && root.dataset.view === "list" && viewVersion === documentVersion) { listScroll = reading.scrollTop; void loadList(true); }
    }
  }

  function renderDocument(entry: ReadingDocument): void {
    reading.innerHTML = `<div class="reading-toolbar"><button class="quiet-button" data-action="back">${t("← 返回列表")}</button><span class="reading-kind">${entry.kind === "daily" ? t("研究日报") : t("论文总结")}</span><div class="reading-controls"><button class="icon-button" data-action="font-down" aria-label="${t("缩小字号")}">A−</button><button class="icon-button" data-action="font-up" aria-label="${t("放大字号")}">A＋</button><a class="quiet-button" data-source="raw" href="api/raw?path=${encodeURIComponent(entry.path)}" target="_blank" rel="noopener noreferrer">Markdown ${symbols.arrow}</a></div></div>
      <div class="article-wrap"><header class="document-header"><div class="document-eyebrow">${escapeHtml(entry.date || t("已保存文档"))}${entry.arxivId ? ` <span>· arXiv:${escapeHtml(entry.arxivId)}</span>` : ""}</div><h1 class="document-title">${scientificInline(entry.title)}</h1>${entry.authors ? `<p class="document-authors">${escapeHtml(entry.authors)}</p>` : ""}<div class="document-links">${sourceLink(entry.originalUrl, t("arXiv 原文"), "original")}${sourceLink(entry.pdfUrl, t("阅读 PDF"), "pdf")}${entry.related.map(item => `<a href="?document=${encodeURIComponent(item.path)}">${t("来源日报 ·")} ${escapeHtml(item.title)}</a>`).join("")}</div></header><article class="markdown-body" aria-label="${t("文档正文")}"></article><footer class="article-footer"><span>${t("Markdown 保存在本地")}</span><span>${escapeHtml(entry.path)}</span></footer>${generationFooter(entry.generationMetrics, entry.kind === "daily" ? "daily" : "paper")}</div>`;
    // Body HTML comes from the safe reader; titles use its safe inline projection.
    find("article").innerHTML = entry.html;
    const firstHeading = find("article").querySelector("h1");
    const firstHeadingMetadata = entry.headings.find(heading => heading.level === 1);
    const sameHeading = firstHeading && firstHeadingMetadata?.id === firstHeading.id
      && firstHeadingMetadata.title.trim() === entry.title.trim();
    if (firstHeading && (sameHeading || firstHeading.textContent === entry.title)) {
      find(".document-title").id = firstHeading.id;
      firstHeading.remove();
    }
    find(".toc-pane").innerHTML = entry.headings.length ? `<div class="toc-inner"><span class="toc-label">${t("本页目录")}</span><nav>${entry.headings.map(heading => `<a href="#${encodeURIComponent(heading.id)}" class="toc-level-${heading.level}">${scientificInline(heading.title)}</a>`).join("")}</nav><span class="toc-note">${t("阅读原文，核对结论。")}</span></div>` : "";
  }

  function sourceLink(url: string | null, label: string, source: string): string {
    if (!url || !(/^(?:https?:\/\/|api\/asset\?)/i.test(url))) return "";
    return `<a data-source="${source}" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${label} ${symbols.arrow}</a>`;
  }

  function openHash(hash: string): void {
    rememberScroll();
    const url = new URL(location.href); url.hash = hash;
    writeRoute(url, "push"); scrollToHash(hash);
  }

  function scrollToHash(hash: string): void {
    let id = hash.replace(/^#/, "");
    try { id = decodeURIComponent(id); } catch { return; }
    if (id === reading.id) { root.classList.add("is-reading"); reading.focus(); return; }
    const heading = Array.from(reading.querySelectorAll<HTMLElement>("[id]")).find(item => item.id === id);
    heading?.scrollIntoView?.({ block: "start", behavior: "instant" });
  }

  function closeDialog(): void {
    dialog?.close?.();
    dialog?.remove();
    dialog = null;
    returnFocus?.focus();
  }

  function showDialog(title: string, content: string): void {
    closeDialog();
    returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dialog = document.createElement("dialog");
    dialog.className = "workbench-dialog";
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    dialog.setAttribute("aria-labelledby", "dialog-title");
    dialog.innerHTML = `<div class="dialog-heading"><h2 id="dialog-title">${title}</h2><button class="icon-button" data-action="close-dialog" aria-label="${t("关闭弹窗")}">×</button></div>${content}`;
    find(".dialog-host").append(dialog);
    if (dialog.showModal) dialog.showModal(); else dialog.setAttribute("open", "");
    dialog.addEventListener("cancel", event => { event.preventDefault(); closeDialog(); });
  }

  async function showSettings(): Promise<void> {
    showDialog(setupRequired ? t("首次使用 arXiv Daily") : t("设置"), `<p class="dialog-description">${t("正在读取设置…")}</p>`);
    const activeDialog = dialog!;
    try {
      const snapshot = await request<SettingsSnapshot>("api/settings");
      if (disposed || dialog !== activeDialog) return;
      activeDialog.querySelector(".dialog-description")!.outerHTML = settingsForm(snapshot, status?.recentRuns?.some(isCompletedDiscovery), appearance);
      bindSettings(activeDialog.querySelector("form")!, snapshot, request, async () => {
        if (disposed) return;
        closeDialog();
        find(".connection-banner").hidden = true;
        if (!await loadStatus()) return;
        await initializeWorkspace(true);
      }, acceptRun, status?.recentRuns?.some(isCompletedDiscovery), { appearance, onAppearanceSaved });
    } catch (error) {
      if (disposed || dialog !== activeDialog) return;
      activeDialog.querySelector(".dialog-description")!.textContent = message(error);
    }
  }

  function showGeneration(date = ""): void {
    if (setupRequired) { void showSettings(); return; }
    const unavailable = !status || !status.llm.ready;
    showDialog(t("生成研究内容"), `<p class="dialog-description">${t("筛选与总结由已配置的 arXiv Daily 流程完成，结果保存为 Markdown。")}</p><form class="generation-form"><fieldset class="generation-kind"><legend>${t("内容类型")}</legend><label><input type="radio" name="kind" value="daily" checked> ${t("日报")}</label><label><input type="radio" name="kind" value="paper"> ${t("单篇详细总结")}</label></fieldset><div class="daily-fields"><label class="field-label" for="run-date">${t("日报日期")} <span>${t("留空生成今天的日报")}</span></label><input id="run-date" name="date" type="date"></div><div class="paper-fields" hidden><label class="field-label" for="run-paper">${t("arXiv ID 或链接")}</label><input id="run-paper" name="paper" type="text" placeholder="${t("例如 2609.12345")}" autocomplete="off"></div>${status?.emailEnabled ? `<p class="generation-note">${t("邮件已开启：日报成功后，会按现有配置发送邮件。")}</p>` : ""}${unavailable ? `<p class="generation-note">${t("模型 API 尚未就绪，请先打开“设置”完成配置。")}</p>` : ""}<p class="form-error" role="alert" hidden></p><div class="dialog-footer"><button type="button" class="quiet-button" data-action="close-dialog">${t("取消")}</button><button class="primary-button" type="submit" ${unavailable || currentRun?.status === "running" ? "disabled" : ""}>${t("开始生成")}</button></div></form>`);
    find<HTMLInputElement>("#run-date").value = date;
  }

  function renderRun(): void {
    const tray = find(".run-tray");
    if (!currentRun || dismissedRun === currentRun.id) { tray.hidden = true; return; }
    const labels = { pending: t("等待公告发布"), running: t("正在运行"), completed: t("已完成"), failed: t("生成失败"), cancelled: t("已取消"), skipped: t("已跳过") };
    const outcomeLabels = { awaiting_announcement: "等待公告发布", no_updates: "当日无更新", no_matches: "无匹配论文", papers_written: "日报已保存" };
    const label = currentRun.outcome && (currentRun.status === "pending" || currentRun.status === "completed") ? t(outcomeLabels[currentRun.outcome]) : labels[currentRun.status];
    const keepOpen = tray.querySelector("details")?.open;
    tray.hidden = false;
    tray.innerHTML = `<div class="run-heading"><div><span class="run-state ${currentRun.status}" role="status">${label}</span><strong>${escapeHtml(runLabel(currentRun.label))}</strong></div>${currentRun.status === "running" ? `<button class="quiet-button" data-action="cancel-run">${t("取消任务")}</button>` : `<button class="icon-button" data-action="dismiss-run" aria-label="${t("关闭任务状态")}">×</button>`}</div><details ${keepOpen || currentRun.status === "failed" ? "open" : ""}><summary>${t("查看运行详情")}</summary><pre></pre></details>`;
    tray.querySelector("pre")!.textContent = t(currentRun.output) || t("等待运行输出…");
  }

  function acceptRun(run: WorkbenchRun | null): void {
    if (disposed) return;
    runVersion += 1;
    if (run && currentRun?.id === run.id && currentRun.status !== "running" && run.status === "running") return;
    const previous = currentRun;
    currentRun = run;
    root.querySelector(".settings-form")?.dispatchEvent(new CustomEvent("workbench-run", { detail: run }));
    renderRun();
    if (run?.id !== previous?.id || run?.status !== previous?.status || run?.date !== previous?.date) void calendar.refresh();
    clearTimeout(pollTimer);
    if (run?.status === "running") pollTimer = setTimeout(() => { void pollRun(); }, options.pollIntervalMs ?? 1200);
    if (run && run.status !== "running" && previous?.id === run.id && previous.status === "running") {
      if (root.dataset.view === "list") { listScroll = reading.scrollTop; void loadList(true); }
      if (selectedKey) void openPaper(selectedKey, false, true);
      else if (selectedPath) void openDocument(selectedPath, "none", location.hash, false);
    }
  }

  async function pollRun(): Promise<void> {
    const version = ++runVersion;
    try {
      const result = await request<{ run: WorkbenchRun | null }>("api/runs/current");
      if (version === runVersion) acceptRun(result.run);
    } catch (error) { if (version === runVersion) reportConnection(error); }
  }

  async function submitGeneration(form: HTMLFormElement): Promise<void> {
    const selected = new FormData(form).get("kind");
    const date = String(new FormData(form).get("date") || "");
    const id = String(new FormData(form).get("paper") || "").trim();
    const body = selected === "paper" ? { kind: "paper", id } : { kind: "daily", ...(date ? { date } : {}) };
    const submit = form.querySelector<HTMLButtonElement>('button[type="submit"]')!;
    const errorBox = form.querySelector<HTMLElement>(".form-error")!;
    if (selected === "paper" && !id) { errorBox.hidden = false; errorBox.textContent = t("请输入 arXiv ID 或链接。"); return; }
    submit.disabled = true;
    submit.textContent = t("正在启动…");
    try {
      const result = await request<{ run: WorkbenchRun }>("api/runs", body);
      if (disposed) return;
      acceptRun(result.run);
      closeDialog();
    } catch (error) {
      if (disposed) return;
      submit.disabled = false;
      submit.textContent = t("开始生成");
      errorBox.hidden = false;
      errorBox.textContent = message(error);
    }
  }

  async function cancelRun(): Promise<void> {
    if (!currentRun || currentRun.status !== "running") return;
    const button = find<HTMLButtonElement>('[data-action="cancel-run"]');
    button.disabled = true;
    button.textContent = t("正在取消…");
    try { acceptRun((await request<{ run: WorkbenchRun }>("api/runs/cancel", { id: currentRun.id })).run); }
    catch (error) { reportConnection(error); void pollRun(); }
  }

  function switchCollection(value: typeof kind, focus = false): void {
    kind = value; documentsMode = true; offset = 0;
    void showList(true).then(() => { if (focus && !disposed) root.querySelector<HTMLButtonElement>(`[data-kind="${kind}"]`)?.focus(); });
  }
  function reconnect(): void {
    find(".connection-banner").hidden = true;
    void loadStatus().then(ok => { if (ok) void initializeWorkspace(true); });
  }

  async function initializeWorkspace(refresh = false): Promise<void> {
    if (disposed) return;
    if (setupRequired) {
      reading.innerHTML = `<div class="empty-reading"><h1>${t("开始积累你的研究记录")}</h1><p>${t("先设置保存目录、模型 API 和关注主题。")}</p><button class="primary-button" data-action="settings">${t("开始设置")}</button></div>`;
      await showSettings(); return;
    }
    void pollRun();
    if (refresh) void calendar.refresh();
    else if (selectedDate) void calendar.selectDate(selectedDate); else void calendar.load();
    if (selectedPath) void openDocument(selectedPath, "none", location.hash);
    else if (selectedKey) void openPaper(selectedKey, false, true);
    else { listScroll = reading.scrollTop; void loadList(true); }
  }

  function click(event: MouseEvent): void {
    const target = event.target instanceof Element ? event.target : null;
    if (!target) return;
    const link = target.closest<HTMLAnchorElement>("a");
    if (link && !event.ctrlKey && !event.metaKey && !event.shiftKey && event.button === 0 && link.target !== "_blank") {
      const url = new URL(link.href, location.href);
      const path = url.searchParams.get("document");
      if (url.origin === location.origin && url.pathname === location.pathname && path) {
        event.preventDefault();
        if (path === selectedPath && url.hash) { openHash(url.hash); }
        else void openDocument(path, "push", url.hash);
        return;
      }
      if (link.getAttribute("href")?.startsWith("#")) { event.preventDefault(); openHash(url.hash); return; }
    }
    const markButton = target.closest<HTMLElement>('[data-mark="star"]');
    if (markButton) { const key = markButton.closest<HTMLElement>("[data-key]")!.dataset.key!; void saveMark(key, "star", markButton.getAttribute("aria-pressed") !== "true"); return; }
    const paperButton = target.closest<HTMLElement>("[data-paper]");
    if (paperButton) { void openPaper(paperButton.dataset.paper!); return; }
    const scopeButton = target.closest<HTMLElement>("[data-scope]");
    if (scopeButton) { scope = scopeButton.dataset.scope as PaperScope; documentsMode = false; offset = 0; void showList(true); return; }
    const documentButton = target.closest<HTMLElement>("[data-document]");
    if (documentButton) { void openDocument(documentButton.dataset.document!); return; }
    const tab = target.closest<HTMLElement>("[data-kind]");
    if (tab) { switchCollection(tab.dataset.kind as typeof kind); return; }
    const action = target.closest<HTMLElement>("[data-action]")?.dataset.action;
    if (action === "history-back") navigateHistory(-1);
    else if (action === "history-forward") navigateHistory(1);
    else if (action === "generate") showGeneration();
    else if (action === "generate-date") showGeneration(target.closest<HTMLElement>("[data-date]")!.dataset.date);
    else if (action === "settings") showSettings();
    else if (action === "close-dialog") closeDialog();
    else if (action === "reconnect" || action === "refresh") reconnect();
    else if (action === "next-page" && nextOffset !== null) { offset = nextOffset; void showList(true); }
    else if (action === "previous-page") { offset = Math.max(0, offset - 20); void showList(true); }
    else if (action === "direction") { direction = direction === "asc" ? "desc" : "asc"; offset = 0; void showList(true); }
    else if (action === "read-day" && paperList?.day?.reportPath) void openDocument(paperList.day.reportPath);
    else if (action === "browse-documents") { documentsMode = true; offset = 0; void showList(true); }
    else if (action === "clear-date") { selectedDate = ""; calendar.clearSelection(); offset = 0; void showList(true); }
    else if (action === "show-filters") { root.classList.toggle("show-filters"); }
    else if (action === "retry-paper" && selectedKey) void openPaper(selectedKey, false);
    else if (action === "generate-paper" && activePaper) {
      showGeneration(); find<HTMLInputElement>('input[name="kind"][value="paper"]').checked = true;
      find(".daily-fields").hidden = true; find(".paper-fields").hidden = false; find<HTMLInputElement>("#run-paper").value = activePaper.arxivId;
    }
    else if (action === "retry-document" && selectedPath) void openDocument(selectedPath, "none", location.hash);
    else if (action === "back") { void showList(true, true).then(() => reading.focus()); }
    else if (action === "cancel-run") void cancelRun();
    else if (action === "dismiss-run") { dismissedRun = currentRun?.id || ""; renderRun(); }
    else if (action === "font-up" || action === "font-down") { fontSize = Math.max(14, Math.min(22, fontSize + (action === "font-up" ? 1 : -1))); root.style.setProperty("--reading-size", `${fontSize}px`); preference("font-size", String(fontSize)); }
  }

  function input(event: Event): void {
    const target = event.target;
    if (target instanceof HTMLInputElement && target.type === "search") {
      query = target.value.trim();
      listVersion += 1;
      clearTimeout(searchTimer);
      offset = 0; documentVersion += 1;
      searchTimer = setTimeout(() => { void showList(true, false, root.classList.contains("show-filters")); }, options.searchDelayMs ?? 180);
    }
    if (target instanceof HTMLSelectElement && target.dataset.mark === "status") {
      const key = target.closest<HTMLElement>("[data-key]")!.dataset.key!; void saveMark(key, "status", target.value);
    }
    if (target instanceof HTMLSelectElement && target.dataset.filter) {
      if (target.dataset.filter === "topic") topic = target.value;
      else if (target.dataset.filter === "sort") { sort = target.value; direction = sort === "title" || sort === "priority" ? "asc" : "desc"; }
      offset = 0; void showList(true);
    }
    if (target instanceof HTMLInputElement && target.name === "kind") {
      find(".daily-fields").hidden = target.value !== "daily";
      find(".paper-fields").hidden = target.value !== "paper";
    }
  }

  function submit(event: Event): void {
    if (event.target instanceof HTMLFormElement && event.target.classList.contains("generation-form")) { event.preventDefault(); void submitGeneration(event.target); }
  }
  function keydown(event: KeyboardEvent): void {
    if (!(event.target instanceof HTMLElement) || !event.target.closest("[data-kind]")) return;
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const values = ["all", "daily", "papers"] as const;
    const next = event.key === "Home" ? 0 : event.key === "End" ? 2 : (values.indexOf(event.target.closest<HTMLElement>("[data-kind]")!.dataset.kind as typeof kind) + (event.key === "ArrowLeft" ? 2 : 1)) % 3;
    switchCollection(values[next]!, true);
  }
  function popstate(): void {
    listVersion += 1; documentVersion += 1; clearTimeout(searchTimer);
    const entry = history.state?.readingNavigation;
    if (entry?.session === navigation.session && Number.isInteger(entry.index) && entry.index >= 0 && entry.index <= navigation.end) navigation.index = entry.index;
    else {
      // Unknown browser entries are a new local boundary, never a license to leave the frame.
      navigation.index = 0; navigation.end = 0; navigation.scrolls.clear();
      history.replaceState({ ...history.state, readingNavigation: { session: navigation.session, index: 0 } }, "");
    }
    navigation.pending = false; syncNavigation();
    const scroll = navigation.scrolls.get(navigation.index);
    readRoute(); syncFilters();
    const loading = selectedPath ? openDocument(selectedPath, "none", location.hash)
      : selectedKey ? openPaper(selectedKey, false) : showList(false, true);
    const version = documentVersion;
    void loading.then(() => { if (!disposed && version === documentVersion && scroll !== undefined) reading.scrollTop = scroll; });
    if (selectedDate) void calendar.selectDate(selectedDate); else calendar.clearSelection();
  }
  root.addEventListener("click", click);
  root.addEventListener("input", input);
  root.addEventListener("change", input);
  root.addEventListener("submit", submit);
  root.addEventListener("keydown", keydown);
  window.addEventListener("popstate", popstate);
  reading.addEventListener("scroll", rememberReadingScroll);
  history.replaceState({ ...history.state, readingNavigation: { session: navigation.session, index: navigation.index } }, "");
  syncNavigation();
  readRoute(); syncFilters();
  void loadStatus().then(ok => { if (ok) void initializeWorkspace(); });
  return () => {
    disposed = true;
    calendar.dispose(); disposeSidebar();
    systemTheme?.removeEventListener?.("change", applyTheme);
    lifetime.abort();
    clearTimeout(searchTimer); clearTimeout(pollTimer);
    closeDialog();
    root.removeEventListener("click", click); root.removeEventListener("input", input); root.removeEventListener("change", input); root.removeEventListener("submit", submit);
    root.removeEventListener("keydown", keydown);
    reading.removeEventListener("scroll", rememberReadingScroll);
    window.removeEventListener("popstate", popstate);
  };
}

function preference(key: string, value?: string): string | null {
  try { if (value !== undefined) localStorage.setItem(`arxiv-daily-reader:${key}`, value); return localStorage.getItem(`arxiv-daily-reader:${key}`); } catch { return null; }
}
function escapeHtml(value: string): string { return value.replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!); }
function message(error: unknown): string { return error instanceof Error ? t(error.message) : t("操作未完成，请重试。"); }
function routeDate(): string { const date = new URL(location.href).searchParams.get("date") || ""; return /^\d{4}-\d{2}-\d{2}$/.test(date) ? date : ""; }

if (typeof document !== "undefined") {
  const app = document.getElementById("app");
  if (app) mountWorkbench(app);
}

function runLabel(label: string): string { const daily=/^(\d{4}-\d{2}-\d{2}) 日报$/.exec(label); const paper=/^(.+) 详细总结$/.exec(label); return daily ? t("{0} 日报",daily[1]!) : paper ? t("{0} 详细总结",paper[1]!) : t(label); }
