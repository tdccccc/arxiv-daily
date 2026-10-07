import { t, getUiLanguage, setUiLanguage } from "./i18n";
import { isCompletedDiscovery, DEFAULT_UI_APPEARANCE, normalizeUiAppearancePreferences, type UiAppearancePreferences } from "@arxiv-daily/core";
import { settingsForm, bindSettings, type SettingsSnapshot } from "./settings";
import type { DocumentEntry, WorkbenchDocuments } from "../documents";
import type { WorkbenchRun } from "../server";
import type { inspectProduct } from "../../inspect-cmd";
import { mountLibraryReview } from "./library-review";
import { mountLibrary } from "./library";
import { generationFooter } from "./generation-footer";
import { mountCalendar } from "./calendar";
import { mountSidebar } from "./sidebar";
import type { WorkbenchPaper, WorkbenchPaperList, PaperScope } from "../papers";
import { scopes, marks, paperRows, dayHeading, overview, scientificInline } from "./papers";
import { h, svg } from "./dom";
import { appendMarkdownNodes } from "./markdown-dom";

export interface WorkbenchClientOptions {
  fetch?: typeof fetch;
  searchDelayMs?: number;
  pollIntervalMs?: number;
  appearance?: UiAppearancePreferences;
}

type ProductStatus = Awaited<ReturnType<typeof inspectProduct>>;
type ReadingDocument = Awaited<ReturnType<WorkbenchDocuments["document"]>>;
interface DocumentList { documents: DocumentEntry[]; total: number; nextOffset: number | null; counts: { daily: number; papers: number } }

function searchIcon(): SVGElement {
  return svg("svg", { viewBox: "0 0 20 20", "aria-hidden": "true" }, svg("circle", { cx: "8.5", cy: "8.5", r: "5.5" }), svg("path", { d: "m13 13 4 4" }));
}
function arrowIcon(): SVGElement {
  return svg("svg", { viewBox: "0 0 20 20", "aria-hidden": "true" }, svg("path", { d: "M5 15 15 5M5 5h10v10" }));
}

export function mountWorkbench(root: HTMLElement, options: WorkbenchClientOptions = {}): () => void {
  const fetcher = options.fetch ?? window.fetch.bind(window);
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
    if (disposed || generation !== initialGeneration || !value.appearance || root.querySelector<HTMLFormElement>('.settings-form')?.dataset.edited === 'true') return;
    const saved = normalizeUiAppearancePreferences(value.appearance);
    if (saved.language !== appearance.language || saved.theme !== appearance.theme) render(saved);
  }).catch(() => { /* Sidebar and settings provide retry UI for unavailable preferences. */ });
  return () => { disposed = true; lifetime.abort(); restoreScroll?.disconnect(); disposeContent?.(); };
}

interface ReadingNavigation { session: string; index: number; end: number; scrolls: Map<number, number>; pending: boolean }

// `history.state` is typed `any` by the DOM lib (it holds whatever a page
// pushes), but this page only ever pushes this shape itself (see the
// `history.pushState`/`replaceState` calls below). Reading it through
// `unknown` keeps that assumption local instead of letting `any` flow out to
// every call site that reads navigation/scroll state back; every field here
// is optional, so a bare object already satisfies the return type.
interface WorkbenchHistoryState { listScroll?: number; readingNavigation?: { session: string; index: number } }
function workbenchHistoryState(): WorkbenchHistoryState {
  const state: unknown = history.state;
  return state && typeof state === "object" ? state : {};
}

function mountWorkbenchContent(root: HTMLElement, options: WorkbenchClientOptions, navigation: ReadingNavigation, onAppearanceSaved: (value: UiAppearancePreferences) => Promise<void>): () => void {
  const fetcher = options.fetch ?? window.fetch.bind(window);
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
  let selectedView = "";
  let reviewView: ReturnType<typeof mountLibraryReview> | undefined;
  let disposeLibrary: (() => void) | undefined;
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
  let searchTimer: number | undefined;
  let pollTimer: number | undefined;
  let returnFocus: HTMLElement | null = null;
  let dialog: HTMLDialogElement | null = null;
  let fontSize = Math.max(14, Math.min(22, Number(preference("font-size")) || 17));

  root.className = "workbench";
  root.dataset.view = "list";
  let appearance = { ...(options.appearance ?? DEFAULT_UI_APPEARANCE) };
  let settingsBinding: ReturnType<typeof bindSettings> | undefined;
  let settingsAppearanceChanged = false;
  const systemTheme = window.matchMedia?.('(prefers-color-scheme: dark)');
  const applyTheme = () => { root.dataset.theme = appearance.theme === 'system' ? (systemTheme?.matches ? 'dark' : 'light') : appearance.theme; };
  applyTheme(); systemTheme?.addEventListener?.('change', applyTheme);
  root.style.setProperty("--reading-size", `${fontSize}px`);
  root.replaceChildren(
    h("a", { class: "skip-link", href: "#reading-content" }, t("跳到正文")),
    h("header", { class: "app-header" },
      h("a", { class: "brand", href: "./", "aria-label": t("arxiv-daily 首页") }, "arxiv", h("span", { class: "brand-hyphen" }, "-"), "daily"),
      h("div", { class: "header-actions" },
        h("nav", { class: "reading-history", "aria-label": t("阅读历史") },
          h("button", { class: "quiet-button", "data-action": "history-back", "aria-label": t("后退"), title: t("后退"), disabled: true }, "← ", h("span", null, t("后退"))),
          h("button", { class: "quiet-button", "data-action": "history-forward", "aria-label": t("前进"), title: t("前进"), disabled: true }, h("span", null, t("前进")), " →"),
        ),
        h("button", { class: "quiet-button", "data-action": "settings" }, t("设置")),
        h("button", { class: "primary-button", "data-action": "generate" }, h("span", { "aria-hidden": "true" }, "＋"), ` ${t("生成")}`),
      ),
    ),
    h("div", { class: "connection-banner", role: "alert", hidden: true }),
    h("div", { class: "workspace" },
      h("aside", { class: "library-pane", "aria-label": t("日历与筛选") },
        h("div", { class: "library-heading" },
          h("span", null, t("我的阅读")),
          h("button", { class: "quiet-button show-filters", "data-action": "show-filters" }, t("返回列表")),
          h("button", { class: "icon-button", "data-action": "refresh", "aria-label": t("刷新文档") }, "↻"),
        ),
        h("section", { class: "calendar-panel", "aria-label": t("日报日历") }),
        h("label", { class: "search-box" }, searchIcon(), h("input", { type: "search", "aria-label": t("搜索标题、作者、arXiv ID 或日期"), placeholder: t("搜索标题、作者或关键词"), autocomplete: "off" })),
        h("nav", { class: "paper-scopes", "aria-label": t("阅读筛选") }, ...Object.entries(scopes).map(([key, label]) => h("button", { class: "scope-button", "data-scope": key }, h("span", null, t(label)), h("span", { "data-count": key }, "—")))),
        h("label", { class: "topic-filter" }, t("Topic"), h("select", { "data-filter": "topic", "aria-label": t("筛选主题") }, h("option", { value: "" }, t("全部主题")))),
        h("div", { class: "navigation-footer" },
          h("button", { class: "quiet-button", "data-action": "personal-library" }, t("个人文献库")),
          h("button", { class: "quiet-button", "data-action": "direction-review" }, t("方向审核")),
          h("button", { class: "quiet-button", "data-action": "clear-date" }, t("浏览全部日期")),
          h("button", { class: "quiet-button", "data-action": "browse-documents" }, t("浏览 Markdown 文件 ↗")),
        ),
      ),
      h("main", { class: "reading-pane", id: "reading-content", tabindex: -1 }),
      h("aside", { class: "toc-pane", "aria-label": t("文章目录") }),
    ),
    h("section", { class: "run-tray", "aria-label": t("生成任务"), hidden: true }),
    h("div", { class: "dialog-host" }),
  );

  const find = <T extends HTMLElement = HTMLElement>(selector: string) => root.querySelector<T>(selector)!;
  const reading = find(".reading-pane");
  const calendar = mountCalendar(find(".calendar-panel"), {
    request,
    selectDay: day => { selectedDate = day.date; documentsMode = false; offset = 0; void showList(true); },
    updateDay: day => {
      if (selectedDate === day.date && paperList && !documentsMode) {
        paperList.day = day;
        const header = reading.querySelector(".day-reading");
        if (root.dataset.view === "list" && header) { const next = dayHeading(day); if (next) header.replaceWith(next); }
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
    banner.replaceChildren(h("span", null, message(error)), h("button", { class: "quiet-button", "data-action": "reconnect" }, t("重新连接")));
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
    const values = { view: selectedView, q: query, scope: scope === "all" ? "" : scope, topic, sort, direction, offset: offset ? String(offset) : "", date: selectedDate, files: documentsMode ? "1" : "", kind: documentsMode ? kind : "", document: selectedPath, paper: selectedKey };
    for (const [key, value] of Object.entries(values)) { if (value) url.searchParams.set(key, value); else url.searchParams.delete(key); }
    url.hash = "";
    writeRoute(url, mode);
  }
  function readRoute(): void {
    const params = new URL(location.href).searchParams;
    selectedView = ["library", "review"].includes(params.get("view") || "") ? params.get("view")! : "";
    query = params.get("q") || ""; scope = Object.hasOwn(scopes, params.get("scope") || "") ? params.get("scope") as PaperScope : "all";
    topic = params.get("topic") || ""; sort = params.get("sort") || "published"; direction = params.get("direction") || "desc";
    offset = Math.max(0, Number(params.get("offset")) || 0); selectedDate = routeDate();
    documentsMode = params.get("files") === "1"; kind = params.get("kind") === "daily" ? "daily" : params.get("kind") === "papers" ? "papers" : "all";
    selectedPath = params.get("document") || ""; selectedKey = params.get("paper") || "";
    listScroll = scrollPositions.get(listIdentity()) ?? workbenchHistoryState().listScroll ?? 0;
  }
  function syncFilters(): void {
    find<HTMLInputElement>('input[type="search"]').value = query;
    for (const button of Array.from(root.querySelectorAll<HTMLElement>("[data-scope]"))) button.setAttribute("aria-pressed", String(button.dataset.scope === scope && !documentsMode));
    find<HTMLSelectElement>('[data-filter="topic"]').value = topic;
    find<HTMLSelectElement>('[data-filter="topic"]').disabled = documentsMode;
  }
  function listToolbar(): HTMLElement {
    return h("div", { class: "paper-list-heading" },
      h("div", null,
        h("span", { class: "day-eyebrow" }, documentsMode ? t("本地研究记录") : selectedDate || t("我的文献")),
        h("h1", null, documentsMode ? t("Markdown 文件") : t(scopes[scope])),
      ),
      h("button", { class: "quiet-button show-filters", "data-action": "show-filters" }, t("日历与筛选")),
    );
  }
  function renderList(): void {
    if (root.dataset.view !== "list" || !paperList) return;
    const unknown = paperList.total === 0 && paperList.day?.reportPath && paperList.day.papers !== 0 && !query && !topic && scope === "all";
    const rows = paperRows(paperList, pendingMarks);
    reading.replaceChildren(
      h("div", { class: "paper-workspace" },
        listToolbar(),
        dayHeading(paperList.day),
        h("div", { class: "paper-list-tools" },
          h("span", { class: "list-caption", "aria-live": "polite" }, unknown ? t("日报已保存，论文索引尚无可用条目") : t("{0} 篇论文", paperList.total)),
          h("label", null, `${t("排序")} `,
            h("select", { "data-filter": "sort", "aria-label": t("论文排序") },
              ...Object.entries({ published: t("发表日期"), title: t("标题"), priority: t("优先级"), relevance: t("相关度") }).map(([value, label]) => h("option", { value, selected: sort === value }, label)),
            ),
          ),
          h("button", { class: "quiet-button", "data-action": "direction", "aria-label": t("切换排序方向") }, direction === "asc" ? t("↑ 升序") : t("↓ 降序")),
        ),
        h("div", { class: "paper-list" }, rows.length ? rows : h("div", { class: "list-empty" }, h("h2", null, unknown ? t("可直接阅读完整日报") : t("没有匹配的论文")), h("p", null, unknown ? t("尚未找到对应的论文索引；原始 Markdown 仍可阅读。") : t("可调整日期或筛选条件，也可通过“生成”获取新论文。")))),
        pagination(paperList.total, paperList.nextOffset),
      ),
    );
    for (const [key, count] of Object.entries(paperList.counts)) find(`[data-count="${key}"]`).textContent = String(count);
    find<HTMLSelectElement>('[data-filter="topic"]').replaceChildren(
      h("option", { value: "" }, t("全部主题")),
      ...[...new Set([...paperList.topics, ...(topic ? [topic] : [])])].map(value => h("option", { value }, value)),
    );
    syncFilters(); syncMarkValues();
  }
  function pagination(total: number, next: number | null): HTMLElement {
    return h("div", { class: "paper-pagination" },
      h("button", { class: "quiet-button", "data-action": "previous-page", disabled: offset === 0 }, t("← 上一页")),
      h("span", null, `${t("第 {0} 页", Math.floor(offset / 20) + 1)}${total ? ` / ${Math.max(1, Math.ceil(total / 20))}` : ""}`),
      h("button", { class: "quiet-button", "data-action": "next-page", disabled: next === null }, t("下一页 →")),
    );
  }
  async function loadList(restore = false): Promise<void> {
    readingReady = false;
    const version = ++listVersion;
    const scroll = restore ? listScroll : 0;
    reading.replaceChildren(h("div", { class: "reading-loading", role: "status" }, t("正在读取列表…")));
    try {
      const params = new URLSearchParams({ q: query, offset: String(offset), limit: "20" });
      if (documentsMode) {
        params.set("kind", kind);
        const result = await request<DocumentList>(`api/documents?${params}`);
        if (disposed || version !== listVersion || root.dataset.view !== "list") return;
        entries = result.documents; nextOffset = result.nextOffset;
        const rows = entries.map(entry => h("button", { class: "document-row", "data-document": entry.path },
          h("span", { class: "document-row-meta" }, entry.date || t("已保存文档")),
          h("span", { class: "document-row-title" }, scientificInline(entry.title)),
          h("span", { class: "document-row-authors" }, entry.authors),
        ));
        reading.replaceChildren(
          h("div", { class: "paper-workspace" },
            listToolbar(),
            h("div", { class: "collection-tabs", role: "tablist", "aria-label": t("文档类型") },
              ...Object.entries({ all: t("全部文件"), daily: t("日报"), papers: t("论文总结") }).map(([value, label]) => h("button", { role: "tab", "data-kind": value, "aria-selected": String(kind === value) }, label)),
            ),
            h("div", { class: "document-list" }, rows.length ? rows : h("div", { class: "list-empty" }, t("暂无匹配文件"))),
            pagination(result.total, result.nextOffset),
          ),
        );
      } else {
        for (const [key, value] of Object.entries({ scope, topic, sort, direction, date: selectedDate })) if (value) params.set(key, value);
        const result = await request<WorkbenchPaperList>(`api/papers?${params}`);
        if (disposed || version !== listVersion || root.dataset.view !== "list") return;
        paperList = result; nextOffset = result.nextOffset; renderList();
      }
      reading.scrollTop = scroll; readingReady = true;
    } catch (error) {
      if (disposed || version !== listVersion || root.dataset.view !== "list") return;
      reading.replaceChildren(
        h("div", { class: "empty-reading" },
          h("h1", null, t("列表暂时不可用")),
          h("p", null, message(error)),
          h("button", { class: "quiet-button", "data-action": "refresh" }, t("重试")),
          h("button", { class: "quiet-button", "data-action": "browse-documents" }, t("浏览 Markdown 文件")),
        ),
      );
      readingReady = true; reportConnection(error);
    }
  }
  async function showLibrary(push = true): Promise<void> {
    if (setupRequired) { await showSettings(); return; }
    if (push) rememberScroll();
    leaveLibrary();
    window.clearTimeout(searchTimer); listVersion += 1; documentVersion += 1;
    selectedView = "library"; selectedKey = ""; selectedPath = ""; activePaper = null;
    root.dataset.view = "library"; root.classList.remove("is-reading", "show-filters");
    find(".toc-pane").replaceChildren();
    if (push) route();
    reading.scrollTop = 0; readingReady = true;
    disposeLibrary = mountLibrary(reading, { request, onSettings: () => { void showSettings(); }, onReview: () => { void showReview(); } });
  }
  async function showReview(push = true): Promise<void> {
    if (setupRequired) { await showSettings(); return; }
    if (push) rememberScroll();
    leaveLibrary(); window.clearTimeout(searchTimer); listVersion += 1; documentVersion += 1;
    selectedView = "review"; selectedKey = ""; selectedPath = ""; activePaper = null;
    root.dataset.view = "review"; root.classList.remove("is-reading", "show-filters");
    find(".toc-pane").replaceChildren();
    if (push) route();
    reading.scrollTop = 0; readingReady = true;
    reviewView = mountLibraryReview(reading, { request, onSettings: () => { void showSettings(); }, onLibrary: () => { void showLibrary(); }, onRun: acceptRun });
    if (currentRun) void reviewView.handleRun(currentRun);
  }
  function leaveLibrary(): void { disposeLibrary?.(); disposeLibrary = undefined; reviewView?.dispose(); reviewView = undefined; selectedView = ""; }
  async function showList(push = false, restore = false, keepFilters = false): Promise<void> {
    if (push) rememberScroll();
    leaveLibrary();
    documentVersion += 1; selectedPath = ""; selectedKey = ""; activePaper = null;
    root.dataset.view = "list"; root.classList.remove("is-reading");
    if (!keepFilters) root.classList.remove("show-filters");
    find(".toc-pane").replaceChildren();
    if (push) route();
    syncFilters();
    await loadList(restore);
  }
  function beginReading(captureScroll: boolean): void {
    if (captureScroll) rememberScroll(); leaveLibrary(); readingReady = false; listVersion += 1;
    root.dataset.view = "reading"; root.classList.add("is-reading"); root.classList.remove("show-filters");
    find(".toc-pane").replaceChildren();
  }
  async function openDocument(path: string, historyMode: "push" | "replace" | "none" = "push", hash = "", syncCalendar = true): Promise<void> {
    beginReading(historyMode !== "none");
    const version = ++documentVersion;
    selectedPath = path; selectedKey = ""; activePaper = null;
    if (historyMode !== "none") { route(historyMode); if (hash) { const url = new URL(location.href); url.hash = hash; history.replaceState(history.state, "", url); } }
    reading.replaceChildren(h("div", { class: "reading-loading", role: "status" }, t("正在打开文档…")));
    try {
      const result = await request<ReadingDocument>(`api/document?path=${encodeURIComponent(path)}`);
      if (disposed || version !== documentVersion) return;
      renderDocument(result);
      if (syncCalendar && result.kind === "daily" && /^\d{4}-\d{2}-\d{2}$/.test(result.date)) void calendar.selectDate(result.date);
      if (hash) scrollToHash(hash); else reading.scrollTop = 0;
      readingReady = true;
    } catch (error) {
      if (disposed || version !== documentVersion) return;
      reading.replaceChildren(
        h("div", { class: "empty-reading" },
          h("button", { class: "quiet-button", "data-action": "back" }, t("← 返回列表")),
          h("h1", null, t("暂时无法打开文档")),
          h("p", null, message(error)),
          h("button", { class: "primary-button", "data-action": "retry-document" }, t("重试读取")),
        ),
      );
      readingReady = true;
      if (message(error).includes("无法连接") || message(error).includes("Cannot connect")) reportConnection(error);
    }
  }
  async function openPaper(key: string, push = true, preserveScroll = false): Promise<void> {
    const scroll = reading.scrollTop;
    beginReading(push); const version = ++documentVersion;
    selectedKey = key; selectedPath = "";
    if (push) route();
    reading.replaceChildren(h("div", { class: "reading-loading", role: "status" }, t("正在打开论文…")));
    try {
      const result = await request<{ paper: WorkbenchPaper }>(`api/paper?key=${encodeURIComponent(key)}`);
      if (disposed || version !== documentVersion) return;
      activePaper = result.paper; reading.replaceChildren(...overview(result.paper, pendingMarks.has(key))); syncMarkValues(); reading.scrollTop = preserveScroll ? scroll : 0; readingReady = true;
    } catch (error) {
      if (disposed || version !== documentVersion) return;
      readingReady = true;
      reading.replaceChildren(
        h("div", { class: "empty-reading" },
          h("button", { class: "quiet-button", "data-action": "back" }, t("← 返回列表")),
          h("h1", null, t("论文暂时不可用")),
          h("p", null, message(error)),
          h("button", { class: "quiet-button", "data-action": "retry-paper" }, t("重试")),
        ),
      );
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
      for (const container of Array.from(root.querySelectorAll<HTMLElement>(".paper-marks"))) if (container.dataset.key === key) container.replaceWith(marks(current, pendingMarks.has(key)));
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
    const article = h("article", { class: "markdown-body", "aria-label": t("文档正文") });
    reading.replaceChildren(
      h("div", { class: "reading-toolbar" },
        h("button", { class: "quiet-button", "data-action": "back" }, t("← 返回列表")),
        h("span", { class: "reading-kind" }, entry.kind === "daily" ? t("研究日报") : t("论文总结")),
        h("div", { class: "reading-controls" },
          h("button", { class: "icon-button", "data-action": "font-down", "aria-label": t("缩小字号") }, "A−"),
          h("button", { class: "icon-button", "data-action": "font-up", "aria-label": t("放大字号") }, "A＋"),
          h("a", { class: "quiet-button", "data-source": "raw", href: `api/raw?path=${encodeURIComponent(entry.path)}`, target: "_blank", rel: "noopener noreferrer" }, "Markdown ", arrowIcon()),
        ),
      ),
      h("div", { class: "article-wrap" },
        h("header", { class: "document-header" },
          h("div", { class: "document-eyebrow" }, entry.date || t("已保存文档"), entry.arxivId ? h("span", null, ` · arXiv:${entry.arxivId}`) : null),
          h("h1", { class: "document-title" }, scientificInline(entry.title)),
          entry.authors ? h("p", { class: "document-authors" }, entry.authors) : null,
          h("div", { class: "document-links" },
            sourceLink(entry.originalUrl, t("arXiv 原文"), "original"),
            sourceLink(entry.pdfUrl, t("阅读 PDF"), "pdf"),
            ...entry.related.map(item => h("a", { href: `?document=${encodeURIComponent(item.path)}` }, `${t("来源日报 ·")} ${item.title}`)),
          ),
        ),
        article,
        h("footer", { class: "article-footer" }, h("span", null, t("Markdown 保存在本地")), h("span", null, entry.path)),
        generationFooter(entry.generationMetrics, entry.kind === "daily" ? "daily" : "paper"),
      ),
    );
    // Body nodes come from the safe reader's token stream; titles use its safe inline projection.
    appendMarkdownNodes(entry.nodes, article);
    const firstHeading = article.querySelector("h1");
    const firstHeadingMetadata = entry.headings.find(heading => heading.level === 1);
    const sameHeading = firstHeading && firstHeadingMetadata?.id === firstHeading.id
      && firstHeadingMetadata.title.trim() === entry.title.trim();
    if (firstHeading && (sameHeading || firstHeading.textContent === entry.title)) {
      find(".document-title").id = firstHeading.id;
      firstHeading.remove();
    }
    find(".toc-pane").replaceChildren(
      ...(entry.headings.length ? [
        h("div", { class: "toc-inner" },
          h("span", { class: "toc-label" }, t("本页目录")),
          h("nav", null, ...entry.headings.map(heading => h("a", { href: `#${encodeURIComponent(heading.id)}`, class: `toc-level-${heading.level}` }, scientificInline(heading.title)))),
          h("span", { class: "toc-note" }, t("阅读原文，核对结论。")),
        ),
      ] : []),
    );
  }

  function sourceLink(url: string | null, label: string, source: string): HTMLAnchorElement | null {
    if (!url || !(/^(?:https?:\/\/|api\/asset\?)/i.test(url))) return null;
    return h("a", { "data-source": source, href: url, target: "_blank", rel: "noopener noreferrer" }, `${label} `, arrowIcon());
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
    settingsBinding?.dispose(); settingsBinding = undefined;
    dialog?.close?.();
    dialog?.remove();
    dialog = null;
    returnFocus?.focus();
  }

  function requestCloseDialog(): void { if (settingsBinding) void settingsBinding.close(); else closeDialog(); }

  function showDialog(title: string, content: Node | Node[]): void {
    closeDialog();
    returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dialog = h("dialog", { class: "workbench-dialog", role: "dialog", "aria-modal": "true", "aria-labelledby": "dialog-title" });
    dialog.replaceChildren(
      h("div", { class: "dialog-heading" }, h("h2", { id: "dialog-title" }, title), h("button", { class: "icon-button", "data-action": "close-dialog", "aria-label": t("关闭弹窗") }, "×")),
      ...(Array.isArray(content) ? content : [content]),
    );
    find(".dialog-host").append(dialog);
    if (dialog.showModal) dialog.showModal(); else dialog.setAttribute("open", "");
    dialog.addEventListener("cancel", event => { event.preventDefault(); requestCloseDialog(); });
  }

  async function showSettings(): Promise<void> {
    showDialog(setupRequired ? t("首次使用 arXiv Daily") : t("设置"), h("p", { class: "dialog-description" }, t("正在读取设置…")));
    const activeDialog = dialog!;
    try {
      const snapshot = await request<SettingsSnapshot>("api/settings");
      if (disposed || dialog !== activeDialog) return;
      activeDialog.querySelector(".dialog-description")!.replaceWith(settingsForm(snapshot, status?.recentRuns?.some(isCompletedDiscovery), appearance));
      settingsAppearanceChanged = false;
      settingsBinding = bindSettings(activeDialog.querySelector("form")!, snapshot, request, async () => {
        if (disposed) return;
        closeDialog();
        if (settingsAppearanceChanged) { await onAppearanceSaved(appearance); if (disposed) return; }
        find(".connection-banner").hidden = true;
        if (!await loadStatus()) return;
        await initializeWorkspace(true);
      }, acceptRun, status?.recentRuns?.some(isCompletedDiscovery), { appearance, onAppearanceSaved: async next => {
        settingsAppearanceChanged = true; appearance = { ...next }; applyTheme();
        document.documentElement.lang = next.language === 'zh' ? 'zh-CN' : 'en'; document.title = t('arxiv-daily · 阅读工作台');
        activeDialog.querySelector('#dialog-title')!.textContent = t('设置');
        activeDialog.querySelector('[data-action="close-dialog"]')!.setAttribute('aria-label', t('关闭弹窗'));
      } });
    } catch (error) {
      if (disposed || dialog !== activeDialog) return;
      activeDialog.querySelector(".dialog-description")!.textContent = message(error);
    }
  }

  function showGeneration(date = ""): void {
    if (setupRequired) { void showSettings(); return; }
    const unavailable = !status || !status.llm.ready;
    showDialog(t("生成研究内容"), [
      h("p", { class: "dialog-description" }, t("筛选与总结由已配置的 arXiv Daily 流程完成，结果保存为 Markdown。")),
      h("form", { class: "generation-form" },
        h("fieldset", { class: "generation-kind" },
          h("legend", null, t("内容类型")),
          h("label", null, h("input", { type: "radio", name: "kind", value: "daily", checked: true }), ` ${t("日报")}`),
          h("label", null, h("input", { type: "radio", name: "kind", value: "paper" }), ` ${t("单篇详细总结")}`),
        ),
        h("div", { class: "daily-fields" },
          h("label", { class: "field-label", for: "run-date" }, t("日报日期"), " ", h("span", null, t("留空生成今天的日报"))),
          h("input", { id: "run-date", name: "date", type: "date" }),
        ),
        h("div", { class: "paper-fields", hidden: true },
          h("label", { class: "field-label", for: "run-paper" }, t("arXiv ID 或链接")),
          h("input", { id: "run-paper", name: "paper", type: "text", placeholder: t("例如 2609.12345"), autocomplete: "off" }),
        ),
        status?.emailEnabled ? h("p", { class: "generation-note" }, t("邮件已开启：日报成功后，会按现有配置发送邮件。")) : null,
        unavailable ? h("p", { class: "generation-note" }, t("模型 API 尚未就绪，请先打开“设置”完成配置。")) : null,
        h("p", { class: "form-error", role: "alert", hidden: true }),
        h("div", { class: "dialog-footer" },
          h("button", { type: "button", class: "quiet-button", "data-action": "close-dialog" }, t("取消")),
          h("button", { class: "primary-button", type: "submit", disabled: unavailable || currentRun?.status === "running" }, t("开始生成")),
        ),
      ),
    ]);
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
    const pre = h("pre");
    tray.replaceChildren(
      h("div", { class: "run-heading" },
        h("div", null, h("span", { class: `run-state ${currentRun.status}`, role: "status" }, label), h("strong", null, runLabel(currentRun.label))),
        currentRun.status === "running"
          ? h("button", { class: "quiet-button", "data-action": "cancel-run" }, t("取消任务"))
          : h("button", { class: "icon-button", "data-action": "dismiss-run", "aria-label": t("关闭任务状态") }, "×"),
      ),
      h("details", { open: keepOpen || currentRun.status === "failed" }, h("summary", null, t("查看运行详情")), pre),
    );
    pre.textContent = t(currentRun.output) || t("等待运行输出…");
  }

  function acceptRun(run: WorkbenchRun | null): void {
    if (disposed) return;
    runVersion += 1;
    if (run && currentRun?.id === run.id && currentRun.status !== "running" && run.status === "running") return;
    const previous = currentRun;
    currentRun = run;
    root.querySelector(".settings-form")?.dispatchEvent(new CustomEvent("workbench-run", { detail: run }));
    renderRun();
    if (run) void reviewView?.handleRun(run);
    if (run?.id !== previous?.id || run?.status !== previous?.status || run?.date !== previous?.date) void calendar.refresh();
    window.clearTimeout(pollTimer);
    if (run?.status === "running") pollTimer = window.setTimeout(() => { void pollRun(); }, options.pollIntervalMs ?? 1200);
    if (run && run.status !== "running" && previous?.id === run.id && previous.status === "running") {
      if (selectedView === "library") void showLibrary(false);
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
      reading.replaceChildren(
        h("div", { class: "empty-reading" },
          h("h1", null, t("开始积累你的研究记录")),
          h("p", null, t("先设置保存目录、模型 API 和关注主题。")),
          h("button", { class: "primary-button", "data-action": "settings" }, t("开始设置")),
        ),
      );
      await showSettings(); return;
    }
    void pollRun();
    if (refresh) void calendar.refresh();
    else if (selectedDate) void calendar.selectDate(selectedDate); else void calendar.load();
    if (selectedView === "review") void showReview(false);
    else if (selectedView === "library") void showLibrary(false);
    else if (selectedPath) void openDocument(selectedPath, "none", location.hash);
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
    else if (action === "personal-library") void showLibrary();
    else if (action === "direction-review") void showReview();
    else if (action === "close-dialog") requestCloseDialog();
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
      window.clearTimeout(searchTimer);
      offset = 0; documentVersion += 1;
      searchTimer = window.setTimeout(() => { void showList(true, false, root.classList.contains("show-filters")); }, options.searchDelayMs ?? 180);
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
    listVersion += 1; documentVersion += 1; window.clearTimeout(searchTimer);
    const entry = workbenchHistoryState().readingNavigation;
    if (entry?.session === navigation.session && Number.isInteger(entry.index) && entry.index >= 0 && entry.index <= navigation.end) navigation.index = entry.index;
    else {
      // Unknown browser entries are a new local boundary, never a license to leave the frame.
      navigation.index = 0; navigation.end = 0; navigation.scrolls.clear();
      history.replaceState({ ...history.state, readingNavigation: { session: navigation.session, index: 0 } }, "");
    }
    navigation.pending = false; syncNavigation();
    const scroll = navigation.scrolls.get(navigation.index);
    readRoute(); syncFilters();
    const loading = selectedView === "review" ? showReview(false) : selectedView === "library" ? showLibrary(false) : selectedPath ? openDocument(selectedPath, "none", location.hash)
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
    settingsBinding?.dispose();
    calendar.dispose(); disposeSidebar(); leaveLibrary();
    systemTheme?.removeEventListener?.("change", applyTheme);
    lifetime.abort();
    window.clearTimeout(searchTimer); window.clearTimeout(pollTimer);
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
function message(error: unknown): string { return error instanceof Error ? t(error.message) : t("操作未完成，请重试。"); }
function routeDate(): string { const date = new URL(location.href).searchParams.get("date") || ""; return /^\d{4}-\d{2}-\d{2}$/.test(date) ? date : ""; }

if (typeof document !== "undefined") {
  const app = document.getElementById("app");
  if (app) mountWorkbench(app);
}

function runLabel(label: string): string { const daily=/^(\d{4}-\d{2}-\d{2}) 日报$/.exec(label); const paper=/^(.+) 详细总结$/.exec(label); return daily ? t("{0} 日报",daily[1]!) : paper ? t("{0} 详细总结",paper[1]!) : t(label); }
