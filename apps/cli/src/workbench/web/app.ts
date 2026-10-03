import { settingsForm, bindSettings, type SettingsSnapshot } from "./settings";
import type { DocumentEntry, WorkbenchDocuments } from "../documents";
import type { WorkbenchRun } from "../server";
import type { inspectProduct } from "../../inspect-cmd";
import { mountCalendar } from "./calendar";
import { mountSidebar } from "./sidebar";
import type { WorkbenchPaper, WorkbenchPaperList, PaperScope } from "../papers";
import { scopes, marks, paperRows, dayHeading, overview } from "./papers";

export interface WorkbenchClientOptions {
  fetch?: typeof fetch;
  searchDelayMs?: number;
  pollIntervalMs?: number;
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
  root.dataset.theme = preference("theme") || "light";
  root.style.setProperty("--reading-size", `${fontSize}px`);
  root.innerHTML = `
    <a class="skip-link" href="#reading-content">跳到正文</a>
    <header class="app-header">
      <a class="brand" href="./" aria-label="arxiv-daily 首页">arxiv<span class="brand-hyphen">-</span>daily</a>
      <div class="header-actions"><button class="quiet-button" data-action="theme" aria-label="切换深浅色">◐ <span class="desktop-label">外观</span></button><button class="quiet-button" data-action="settings">设置</button><button class="primary-button" data-action="generate"><span aria-hidden="true">＋</span> 生成</button></div>
    </header>
    <div class="connection-banner" role="alert" hidden></div>
    <div class="workspace">
      <aside class="library-pane" aria-label="日历与筛选">
        <div class="library-heading"><span>我的阅读</span><button class="quiet-button show-filters" data-action="show-filters">返回列表</button><button class="icon-button" data-action="refresh" aria-label="刷新文档">↻</button></div>
        <section class="calendar-panel" aria-label="日报日历"></section>
        <label class="search-box">${symbols.search}<input type="search" aria-label="搜索标题、作者、arXiv ID 或日期" placeholder="搜索标题、作者或关键词" autocomplete="off"></label>
        <nav class="paper-scopes" aria-label="阅读筛选">${Object.entries(scopes).map(([key, label]) => `<button class="scope-button" data-scope="${key}"><span>${label}</span><span data-count="${key}">—</span></button>`).join("")}</nav>
        <label class="topic-filter">主题<select data-filter="topic" aria-label="筛选主题"><option value="">全部主题</option></select></label>
        <div class="navigation-footer"><button class="quiet-button" data-action="clear-date">浏览全部日期</button><button class="quiet-button" data-action="browse-documents">浏览 Markdown 文件 ↗</button></div>
      </aside>
      <main class="reading-pane" id="reading-content" tabindex="-1"></main>
      <aside class="toc-pane" aria-label="文章目录"></aside>
    </div>
    <section class="run-tray" aria-label="生成任务" hidden></section>
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
      if (!response.ok) throw new Error(value.error || `请求失败（${response.status}）`);
      return value;
    } catch (error) {
      if (error instanceof TypeError) throw new Error("无法连接本地工作台。请确认启动它的终端仍在运行，再重新连接。");
      throw error;
    }
  }

  function reportConnection(error: unknown): void {
    if (disposed) return;
    const banner = find(".connection-banner");
    banner.innerHTML = `<span>${escapeHtml(message(error))}</span><button class="quiet-button" data-action="reconnect">重新连接</button>`;
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
  function rememberScroll(): void {
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
    history[mode === "push" ? "pushState" : "replaceState"]({ listScroll }, "", url);
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
    return `<div class="paper-list-heading"><div><span class="day-eyebrow">${documentsMode ? "本地研究记录" : selectedDate || "我的文献"}</span><h1>${documentsMode ? "Markdown 文件" : scopes[scope]}</h1></div><button class="quiet-button show-filters" data-action="show-filters">日历与筛选</button></div>`;
  }
  function renderList(): void {
    if (root.dataset.view !== "list" || !paperList) return;
    const unknown = paperList.total === 0 && paperList.day?.reportPath && paperList.day.papers !== 0 && !query && !topic && scope === "all";
    reading.innerHTML = `<div class="paper-workspace">${listToolbar()}${dayHeading(paperList.day)}<div class="paper-list-tools"><span class="list-caption" aria-live="polite">${unknown ? "日报已保存，论文索引尚无可用条目" : `${paperList.total} 篇论文`}</span><label>排序 <select data-filter="sort" aria-label="论文排序">${Object.entries({ published: "发表日期", title: "标题", priority: "优先级", relevance: "相关度" }).map(([value, label]) => `<option value="${value}" ${sort === value ? "selected" : ""}>${label}</option>`).join("")}</select></label><button class="quiet-button" data-action="direction" aria-label="切换排序方向">${direction === "asc" ? "↑ 升序" : "↓ 降序"}</button></div><div class="paper-list">${paperRows(paperList, pendingMarks) || `<div class="list-empty"><h2>${unknown ? "可直接阅读完整日报" : "没有匹配的论文"}</h2><p>${unknown ? "尚未找到对应的论文索引；原始 Markdown 仍可阅读。" : "可调整日期或筛选条件，也可通过“生成”获取新论文。"}</p></div>`}</div>${pagination(paperList.total, paperList.nextOffset)}</div>`;
    for (const [key, count] of Object.entries(paperList.counts)) find(`[data-count="${key}"]`).textContent = String(count);
    find<HTMLSelectElement>('[data-filter="topic"]').innerHTML = `<option value="">全部主题</option>${[...new Set([...paperList.topics, ...(topic ? [topic] : [])])].map(value => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
    syncFilters(); syncMarkValues();
  }
  function pagination(total: number, next: number | null): string {
    return `<div class="paper-pagination"><button class="quiet-button" data-action="previous-page" ${offset === 0 ? "disabled" : ""}>← 上一页</button><span>第 ${Math.floor(offset / 20) + 1} 页${total ? ` / ${Math.max(1, Math.ceil(total / 20))}` : ""}</span><button class="quiet-button" data-action="next-page" ${next === null ? "disabled" : ""}>下一页 →</button></div>`;
  }
  async function loadList(restore = false): Promise<void> {
    const version = ++listVersion;
    const scroll = restore ? listScroll : 0;
    reading.innerHTML = '<div class="reading-loading" role="status">正在读取列表…</div>';
    try {
      const params = new URLSearchParams({ q: query, offset: String(offset), limit: "20" });
      if (documentsMode) {
        params.set("kind", kind);
        const result = await request<DocumentList>(`api/documents?${params}`);
        if (disposed || version !== listVersion || root.dataset.view !== "list") return;
        entries = result.documents; nextOffset = result.nextOffset;
        reading.innerHTML = `<div class="paper-workspace">${listToolbar()}<div class="collection-tabs" role="tablist" aria-label="文档类型">${Object.entries({ all: "全部文件", daily: "日报", papers: "论文总结" }).map(([value, label]) => `<button role="tab" data-kind="${value}" aria-selected="${kind === value}">${label}</button>`).join("")}</div><div class="document-list">${entries.map(entry => `<button class="document-row" data-document="${escapeHtml(entry.path)}"><span class="document-row-meta">${escapeHtml(entry.date || "已保存文档")}</span><span class="document-row-title">${escapeHtml(entry.title)}</span><span class="document-row-authors">${escapeHtml(entry.authors)}</span></button>`).join("") || '<div class="list-empty">暂无匹配文件</div>'}</div>${pagination(result.total, result.nextOffset)}</div>`;
      } else {
        for (const [key, value] of Object.entries({ scope, topic, sort, direction, date: selectedDate })) if (value) params.set(key, value);
        const result = await request<WorkbenchPaperList>(`api/papers?${params}`);
        if (disposed || version !== listVersion || root.dataset.view !== "list") return;
        paperList = result; nextOffset = result.nextOffset; renderList();
      }
      reading.scrollTop = scroll;
    } catch (error) {
      if (disposed || version !== listVersion || root.dataset.view !== "list") return;
      reading.innerHTML = `<div class="empty-reading"><h1>列表暂时不可用</h1><p>${escapeHtml(message(error))}</p><button class="quiet-button" data-action="refresh">重试</button><button class="quiet-button" data-action="browse-documents">浏览 Markdown 文件</button></div>`;
      reportConnection(error);
    }
  }
  async function showList(push = false, restore = false, keepFilters = false): Promise<void> {
    documentVersion += 1; selectedPath = ""; selectedKey = ""; activePaper = null;
    root.dataset.view = "list"; root.classList.remove("is-reading");
    if (!keepFilters) root.classList.remove("show-filters");
    find(".toc-pane").innerHTML = "";
    if (push) route();
    syncFilters();
    await loadList(restore);
  }
  function beginReading(captureScroll: boolean): void {
    if (captureScroll) rememberScroll(); listVersion += 1;
    root.dataset.view = "reading"; root.classList.add("is-reading"); root.classList.remove("show-filters");
    find(".toc-pane").innerHTML = "";
  }
  async function openDocument(path: string, historyMode: "push" | "replace" | "none" = "push", hash = "", syncCalendar = true): Promise<void> {
    beginReading(historyMode !== "none");
    const version = ++documentVersion;
    selectedPath = path; selectedKey = ""; activePaper = null;
    if (historyMode !== "none") { route(historyMode); if (hash) { const url = new URL(location.href); url.hash = hash; history.replaceState(history.state, "", url); } }
    reading.innerHTML = '<div class="reading-loading" role="status">正在打开文档…</div>';
    try {
      const result = await request<ReadingDocument>(`api/document?path=${encodeURIComponent(path)}`);
      if (disposed || version !== documentVersion) return;
      renderDocument(result);
      if (syncCalendar && result.kind === "daily" && /^\d{4}-\d{2}-\d{2}$/.test(result.date)) void calendar.selectDate(result.date);
      if (hash) scrollToHash(hash); else reading.scrollTop = 0;
    } catch (error) {
      if (disposed || version !== documentVersion) return;
      reading.innerHTML = `<div class="empty-reading"><button class="quiet-button" data-action="back">← 返回列表</button><h1>暂时无法打开文档</h1><p>${escapeHtml(message(error))}</p><button class="primary-button" data-action="retry-document">重试读取</button></div>`;
      if (message(error).includes("无法连接")) reportConnection(error);
    }
  }
  async function openPaper(key: string, push = true, preserveScroll = false): Promise<void> {
    const scroll = reading.scrollTop;
    beginReading(push); const version = ++documentVersion;
    selectedKey = key; selectedPath = "";
    if (push) route();
    reading.innerHTML = '<div class="reading-loading" role="status">正在打开论文…</div>';
    try {
      const result = await request<{ paper: WorkbenchPaper }>(`api/paper?key=${encodeURIComponent(key)}`);
      if (disposed || version !== documentVersion) return;
      activePaper = result.paper; reading.innerHTML = overview(result.paper, pendingMarks.has(key)); syncMarkValues(); reading.scrollTop = preserveScroll ? scroll : 0;
    } catch (error) {
      if (disposed || version !== documentVersion) return;
      reading.innerHTML = `<div class="empty-reading"><button class="quiet-button" data-action="back">← 返回列表</button><h1>论文暂时不可用</h1><p>${escapeHtml(message(error))}</p><button class="quiet-button" data-action="retry-paper">重试</button></div>`;
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
    reading.innerHTML = `<div class="reading-toolbar"><button class="quiet-button" data-action="back">← 返回列表</button><span class="reading-kind">${entry.kind === "daily" ? "研究日报" : "论文总结"}</span><div class="reading-controls"><button class="icon-button" data-action="font-down" aria-label="缩小字号">A−</button><button class="icon-button" data-action="font-up" aria-label="放大字号">A＋</button><a class="quiet-button" data-source="raw" href="api/raw?path=${encodeURIComponent(entry.path)}" target="_blank" rel="noopener noreferrer">Markdown ${symbols.arrow}</a></div></div>
      <div class="article-wrap"><header class="document-header"><div class="document-eyebrow">${escapeHtml(entry.date || "已保存文档")}${entry.arxivId ? ` <span>· arXiv:${escapeHtml(entry.arxivId)}</span>` : ""}</div><h1 class="document-title">${escapeHtml(entry.title)}</h1>${entry.authors ? `<p class="document-authors">${escapeHtml(entry.authors)}</p>` : ""}<div class="document-links">${sourceLink(entry.originalUrl, "arXiv 原文", "original")}${sourceLink(entry.pdfUrl, "阅读 PDF", "pdf")}${entry.related.map(item => `<a href="?document=${encodeURIComponent(item.path)}">来源日报 · ${escapeHtml(item.title)}</a>`).join("")}</div></header><article class="markdown-body" aria-label="文档正文"></article><footer class="article-footer"><span>Markdown 保存在本地</span><span>${escapeHtml(entry.path)}</span></footer></div>`;
    // The local service owns Markdown sanitization. All other API text is escaped above.
    find("article").innerHTML = entry.html;
    const firstHeading = find("article").querySelector("h1");
    if (firstHeading?.textContent === entry.title) {
      find(".document-title").id = firstHeading.id;
      firstHeading.remove();
    }
    find(".toc-pane").innerHTML = entry.headings.length ? `<div class="toc-inner"><span class="toc-label">本页目录</span><nav>${entry.headings.map(heading => `<a href="#${encodeURIComponent(heading.id)}" class="toc-level-${heading.level}">${escapeHtml(heading.title)}</a>`).join("")}</nav><span class="toc-note">阅读原文，核对结论。</span></div>` : "";
  }

  function sourceLink(url: string | null, label: string, source: string): string {
    if (!url || !(/^(?:https?:\/\/|api\/asset\?)/i.test(url))) return "";
    return `<a data-source="${source}" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${label} ${symbols.arrow}</a>`;
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
    dialog.innerHTML = `<div class="dialog-heading"><h2 id="dialog-title">${title}</h2><button class="icon-button" data-action="close-dialog" aria-label="关闭弹窗">×</button></div>${content}`;
    find(".dialog-host").append(dialog);
    if (dialog.showModal) dialog.showModal(); else dialog.setAttribute("open", "");
    dialog.addEventListener("cancel", event => { event.preventDefault(); closeDialog(); });
  }

  async function showSettings(): Promise<void> {
    showDialog(setupRequired ? "首次使用 arXiv Daily" : "设置", '<p class="dialog-description">正在读取设置…</p>');
    const activeDialog = dialog!;
    try {
      const snapshot = await request<SettingsSnapshot>("api/settings");
      if (disposed || dialog !== activeDialog) return;
      activeDialog.querySelector(".dialog-description")!.outerHTML = settingsForm(snapshot, status?.recentRuns?.some(run => run.status === "completed"));
      bindSettings(activeDialog.querySelector("form")!, snapshot, request, async () => {
        closeDialog();
        find(".connection-banner").hidden = true;
        if (!await loadStatus()) return;
        await initializeWorkspace(true);
      }, acceptRun, status?.recentRuns?.some(run => run.status === "completed"));
    } catch (error) {
      if (disposed || dialog !== activeDialog) return;
      activeDialog.querySelector(".dialog-description")!.textContent = message(error);
    }
  }

  function showGeneration(date = ""): void {
    if (setupRequired) { void showSettings(); return; }
    const unavailable = !status || !status.llm.ready;
    showDialog("生成研究内容", `<p class="dialog-description">筛选与总结由已配置的 arXiv Daily 流程完成，结果保存为 Markdown。</p><form class="generation-form"><fieldset class="generation-kind"><legend>内容类型</legend><label><input type="radio" name="kind" value="daily" checked> 日报</label><label><input type="radio" name="kind" value="paper"> 单篇详细总结</label></fieldset><div class="daily-fields"><label class="field-label" for="run-date">日报日期 <span>留空生成今天的日报</span></label><input id="run-date" name="date" type="date"></div><div class="paper-fields" hidden><label class="field-label" for="run-paper">arXiv ID 或链接</label><input id="run-paper" name="paper" type="text" placeholder="例如 2609.12345" autocomplete="off"></div>${status?.emailEnabled ? '<p class="generation-note">邮件已开启：日报成功后，会按现有配置发送邮件。</p>' : ""}${unavailable ? '<p class="generation-note">模型 API 尚未就绪，请先打开“设置”完成配置。</p>' : ""}<p class="form-error" role="alert" hidden></p><div class="dialog-footer"><button type="button" class="quiet-button" data-action="close-dialog">取消</button><button class="primary-button" type="submit" ${unavailable || currentRun?.status === "running" ? "disabled" : ""}>开始生成</button></div></form>`);
    find<HTMLInputElement>("#run-date").value = date;
  }

  function renderRun(): void {
    const tray = find(".run-tray");
    if (!currentRun || dismissedRun === currentRun.id) { tray.hidden = true; return; }
    const labels = { running: "正在运行", completed: "已完成", failed: "生成失败", cancelled: "已取消", skipped: "已跳过" };
    const keepOpen = tray.querySelector("details")?.open;
    tray.hidden = false;
    tray.innerHTML = `<div class="run-heading"><div><span class="run-state ${currentRun.status}" role="status">${labels[currentRun.status]}</span><strong>${escapeHtml(currentRun.label)}</strong></div>${currentRun.status === "running" ? '<button class="quiet-button" data-action="cancel-run">取消任务</button>' : '<button class="icon-button" data-action="dismiss-run" aria-label="关闭任务状态">×</button>'}</div><details ${keepOpen || currentRun.status === "failed" ? "open" : ""}><summary>查看运行详情</summary><pre></pre></details>`;
    tray.querySelector("pre")!.textContent = currentRun.output || "等待运行输出…";
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
    if (selected === "paper" && !id) { errorBox.hidden = false; errorBox.textContent = "请输入 arXiv ID 或链接。"; return; }
    submit.disabled = true;
    submit.textContent = "正在启动…";
    try {
      const result = await request<{ run: WorkbenchRun }>("api/runs", body);
      if (disposed) return;
      acceptRun(result.run);
      closeDialog();
    } catch (error) {
      if (disposed) return;
      submit.disabled = false;
      submit.textContent = "开始生成";
      errorBox.hidden = false;
      errorBox.textContent = message(error);
    }
  }

  async function cancelRun(): Promise<void> {
    if (!currentRun || currentRun.status !== "running") return;
    const button = find<HTMLButtonElement>('[data-action="cancel-run"]');
    button.disabled = true;
    button.textContent = "正在取消…";
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
      reading.innerHTML = '<div class="empty-reading"><h1>开始积累你的研究记录</h1><p>先设置保存目录、模型 API 和关注主题。</p><button class="primary-button" data-action="settings">开始设置</button></div>';
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
        if (path === selectedPath && url.hash) { history.pushState({}, "", url); scrollToHash(url.hash); }
        else void openDocument(path, "push", url.hash);
        return;
      }
      if (link.getAttribute("href")?.startsWith("#")) { event.preventDefault(); history.pushState({}, "", url); scrollToHash(url.hash); return; }
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
    if (action === "generate") showGeneration();
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
    else if (action === "theme") { root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark"; preference("theme", root.dataset.theme); }
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
    readRoute(); syncFilters();
    if (selectedPath) void openDocument(selectedPath, "none", location.hash);
    else if (selectedKey) void openPaper(selectedKey, false);
    else void showList(false, true);
    if (selectedDate) void calendar.selectDate(selectedDate); else calendar.clearSelection();
  }
  root.addEventListener("click", click);
  root.addEventListener("input", input);
  root.addEventListener("change", input);
  root.addEventListener("submit", submit);
  root.addEventListener("keydown", keydown);
  window.addEventListener("popstate", popstate);
  readRoute(); syncFilters();
  void loadStatus().then(ok => { if (ok) void initializeWorkspace(); });
  return () => {
    disposed = true;
    calendar.dispose(); disposeSidebar();
    lifetime.abort();
    clearTimeout(searchTimer); clearTimeout(pollTimer);
    closeDialog();
    root.removeEventListener("click", click); root.removeEventListener("input", input); root.removeEventListener("change", input); root.removeEventListener("submit", submit);
    root.removeEventListener("keydown", keydown);
    window.removeEventListener("popstate", popstate);
  };
}

function preference(key: string, value?: string): string | null {
  try { if (value !== undefined) localStorage.setItem(`arxiv-daily-reader:${key}`, value); return localStorage.getItem(`arxiv-daily-reader:${key}`); } catch { return null; }
}
function escapeHtml(value: string): string { return value.replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!); }
function message(error: unknown): string { return error instanceof Error ? error.message : "操作未完成，请重试。"; }
function routeDate(): string { const date = new URL(location.href).searchParams.get("date") || ""; return /^\d{4}-\d{2}-\d{2}$/.test(date) ? date : ""; }

if (typeof document !== "undefined") {
  const app = document.getElementById("app");
  if (app) mountWorkbench(app);
}
