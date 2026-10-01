import type { DocumentEntry, WorkbenchDocuments } from "../documents";
import type { WorkbenchRun } from "../server";
import type { inspectProduct } from "../../inspect-cmd";

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
  let kind: "daily" | "papers" = "daily";
  let query = "";
  let entries: DocumentEntry[] = [];
  let nextOffset: number | null = null;
  let selectedPath = new URL(location.href).searchParams.get("document") || "";
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
  root.dataset.theme = preference("theme") || "light";
  root.style.setProperty("--reading-size", `${fontSize}px`);
  root.innerHTML = `
    <a class="skip-link" href="#reading-content">跳到正文</a>
    <header class="app-header">
      <a class="brand" href="./" aria-label="arXiv Daily 首页"><span class="brand-mark" aria-hidden="true">a<span>↗</span></span><span>arXiv Daily<small>阅读工作台</small></span></a>
      <div class="header-actions"><button class="quiet-button" data-action="theme" aria-label="切换深浅色">◐ <span class="desktop-label">外观</span></button><button class="quiet-button" data-action="settings">设置</button><button class="primary-button" data-action="generate"><span aria-hidden="true">＋</span> 生成</button></div>
    </header>
    <div class="connection-banner" role="alert" hidden></div>
    <div class="workspace">
      <aside class="library-pane" aria-label="文档列表">
        <div class="library-heading"><span>我的阅读</span><button class="icon-button" data-action="refresh" aria-label="刷新文档">↻</button></div>
        <div class="collection-tabs" role="tablist" aria-label="文档类型"><button role="tab" aria-selected="true" data-kind="daily">日报 <span data-count="daily">0</span></button><button role="tab" aria-selected="false" data-kind="papers">论文总结 <span data-count="papers">0</span></button></div>
        <label class="search-box">${symbols.search}<input type="search" aria-label="搜索标题、作者、arXiv ID 或日期" placeholder="搜索标题、作者或 ID" autocomplete="off"></label>
        <div class="list-caption" aria-live="polite">正在读取文档…</div>
        <div class="document-list"></div>
        <div class="list-footer"><button class="quiet-button" data-action="more" hidden>加载更多</button><span>本地 Markdown</span></div>
      </aside>
      <main class="reading-pane" id="reading-content" tabindex="-1"><div class="empty-reading"><span class="empty-symbol" aria-hidden="true">≡</span><h1>从这里开始阅读</h1><p>选择一份日报或论文总结。</p></div></main>
      <aside class="toc-pane" aria-label="文章目录"></aside>
    </div>
    <section class="run-tray" aria-label="生成任务" hidden></section>
    <div class="dialog-host"></div>`;

  const find = <T extends HTMLElement = HTMLElement>(selector: string) => root.querySelector<T>(selector)!;
  const reading = find(".reading-pane");

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

  async function loadStatus(): Promise<void> {
    try {
      const result = await request<ProductStatus>("api/status");
      if (disposed) return;
      status = result;
    } catch (error) { reportConnection(error); }
  }

  function renderList(total: number): void {
    find(".list-caption").textContent = query ? `${total} 项匹配 · 标题、作者、ID、日期` : `${total} 份${kind === "daily" ? "日报" : "论文总结"}`;
    find(".document-list").innerHTML = entries.length ? entries.map(entry => `
      <button class="document-row${entry.path === selectedPath ? " is-selected" : ""}" data-document="${escapeHtml(entry.path)}" ${entry.path === selectedPath ? 'aria-current="page"' : ""}>
        <span class="document-row-meta">${escapeHtml(entry.date || (entry.kind === "daily" ? "日报" : "论文总结"))}${entry.arxivId ? ` <span>· ${escapeHtml(entry.arxivId)}</span>` : ""}</span>
        <span class="document-row-title">${escapeHtml(entry.title)}</span>
        ${entry.authors ? `<span class="document-row-authors">${escapeHtml(entry.authors)}</span>` : ""}
      </button>`).join("") : `<div class="list-empty"><strong>${query ? "没有匹配的文档" : kind === "daily" ? "还没有日报" : "还没有论文总结"}</strong><p>${query ? "试试其他标题、作者、ID 或日期。" : "通过右上角“生成”保存第一份文档。"}</p></div>`;
    find<HTMLButtonElement>('[data-action="more"]').hidden = nextOffset === null;
  }

  async function loadList(append = false, autoOpen = false): Promise<void> {
    const version = ++listVersion;
    const offset = append ? nextOffset ?? 0 : 0;
    find(".list-caption").textContent = "正在读取文档…";
    try {
      const params = new URLSearchParams({ kind, q: query, offset: String(offset), limit: "60" });
      const result = await request<DocumentList>(`api/documents?${params}`);
      if (disposed || version !== listVersion) return;
      entries = append ? [...entries, ...result.documents] : result.documents;
      nextOffset = result.nextOffset;
      find('[data-count="daily"]').textContent = String(result.counts.daily);
      find('[data-count="papers"]').textContent = String(result.counts.papers);
      renderList(result.total);
      if (autoOpen && !selectedPath && !window.matchMedia("(max-width: 760px)").matches && entries[0]) void openDocument(entries[0].path, "replace");
    } catch (error) {
      if (disposed || version !== listVersion) return;
      find(".list-caption").textContent = "读取未完成";
      find(".document-list").innerHTML = `<div class="list-empty"><strong>文档列表暂时不可用</strong><p>${escapeHtml(message(error))}</p><button class="quiet-button" data-action="refresh">重试</button></div>`;
      reportConnection(error);
    }
  }

  function markSelection(): void {
    for (const row of Array.from(root.querySelectorAll<HTMLElement>("[data-document]"))) {
      const selected = row.dataset.document === selectedPath;
      row.classList.toggle("is-selected", selected);
      if (selected) row.setAttribute("aria-current", "page"); else row.removeAttribute("aria-current");
    }
  }

  async function openDocument(path: string, historyMode: "push" | "replace" | "none" = "push", hash = ""): Promise<void> {
    const version = ++documentVersion;
    selectedPath = path;
    markSelection();
    root.classList.add("is-reading");
    if (historyMode !== "none") {
      const url = new URL(location.href);
      url.searchParams.set("document", path);
      url.hash = hash;
      history[historyMode === "push" ? "pushState" : "replaceState"]({}, "", url);
    }
    reading.innerHTML = `<div class="reading-loading" role="status">正在打开文档…</div>`;
    find(".toc-pane").innerHTML = "";
    try {
      const result = await request<ReadingDocument>(`api/document?path=${encodeURIComponent(path)}`);
      if (disposed || version !== documentVersion) return;
      renderDocument(result);
      if (hash) scrollToHash(hash); else reading.scrollTop = 0;
    } catch (error) {
      if (disposed || version !== documentVersion) return;
      reading.innerHTML = `<div class="empty-reading"><button class="quiet-button mobile-back" data-action="back">← 返回列表</button><h1>暂时无法打开文档</h1><p>${escapeHtml(message(error))}</p><button class="primary-button" data-action="retry-document">重试读取</button></div>`;
      if (message(error).includes("无法连接")) reportConnection(error);
    }
  }

  function renderDocument(entry: ReadingDocument): void {
    reading.innerHTML = `<div class="reading-toolbar"><button class="quiet-button mobile-back" data-action="back">← 返回列表</button><span class="reading-kind">${entry.kind === "daily" ? "研究日报" : "论文总结"}</span><div class="reading-controls"><button class="icon-button" data-action="font-down" aria-label="缩小字号">A−</button><button class="icon-button" data-action="font-up" aria-label="放大字号">A＋</button><a class="quiet-button" data-source="raw" href="api/raw?path=${encodeURIComponent(entry.path)}" target="_blank" rel="noopener noreferrer">Markdown ${symbols.arrow}</a></div></div>
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

  function showSettings(): void {
    if (!status) {
      showDialog("当前设置", '<p class="dialog-description">尚未读取到设置，请重新连接工作台。</p><button class="primary-button" data-action="reconnect">重新连接</button>');
      return;
    }
    const fields = [["输出目录", status.vaultRoot], ["模型", `${status.llm.provider} · ${status.llm.model}`], ["模型 API", status.llm.keyConfigured ? "已配置" : "未配置"], ["arXiv 分类", status.categories.join("、")], ["总结语言", status.output.summaryLanguage === "zh" ? "中文" : "English"], ["邮件", status.emailEnabled ? "已开启，日报成功后按配置发送" : "未开启"]];
    showDialog("当前设置", `<p class="dialog-description">工作台使用启动时读取的 CLI 设置。</p><dl class="settings-list">${fields.map(([name, value]) => `<div><dt>${name}</dt><dd>${escapeHtml(value || "未设置")}</dd></div>`).join("")}</dl><h3 class="settings-subtitle">关注主题</h3><div class="settings-topics">${status.topics.length ? status.topics.map(topic => `<div><strong>${escapeHtml(topic.name)}</strong><p>${escapeHtml(topic.description)}</p></div>`).join("") : "尚未配置主题"}</div><div class="settings-help"><strong>修改设置</strong><p>在终端运行 <code>arxiv-daily init</code>，或编辑下方 TOML 文件。保存后重启工作台，使新设置生效。</p><code class="config-path">${escapeHtml(status.configPath)}</code><p>论文生成使用这里配置的模型 API，与 Claude Code 的对话模型独立。</p></div>`);
  }

  function showGeneration(): void {
    const unavailable = !status || !status.llm.ready;
    showDialog("生成研究内容", `<p class="dialog-description">筛选与总结由已配置的 arXiv Daily 流程完成，结果保存为 Markdown。</p><form class="generation-form"><fieldset class="generation-kind"><legend>内容类型</legend><label><input type="radio" name="kind" value="daily" checked> 日报</label><label><input type="radio" name="kind" value="paper"> 单篇详细总结</label></fieldset><div class="daily-fields"><label class="field-label" for="run-date">日报日期 <span>留空生成今天的日报</span></label><input id="run-date" name="date" type="date"></div><div class="paper-fields" hidden><label class="field-label" for="run-paper">arXiv ID 或链接</label><input id="run-paper" name="paper" type="text" placeholder="例如 2609.12345" autocomplete="off"></div>${status?.emailEnabled ? '<p class="generation-note">邮件已开启：日报成功后，会按现有配置发送邮件。</p>' : ""}${unavailable ? '<p class="generation-note">模型 API 尚未就绪，请先在终端完成配置并重启工作台。</p>' : ""}<p class="form-error" role="alert" hidden></p><div class="dialog-footer"><button type="button" class="quiet-button" data-action="close-dialog">取消</button><button class="primary-button" type="submit" ${unavailable || currentRun?.status === "running" ? "disabled" : ""}>开始生成</button></div></form>`);
  }

  function renderRun(): void {
    const tray = find(".run-tray");
    if (!currentRun || dismissedRun === currentRun.id) { tray.hidden = true; return; }
    const labels = { running: "正在运行", completed: "已完成", failed: "生成失败", cancelled: "已取消" };
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
    renderRun();
    clearTimeout(pollTimer);
    if (run?.status === "running") pollTimer = setTimeout(() => { void pollRun(); }, options.pollIntervalMs ?? 1200);
    if (run && run.status !== "running" && previous?.id === run.id && previous.status === "running") {
      void loadList(false, true);
      if (selectedPath) void openDocument(selectedPath, "none", location.hash);
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

  function switchCollection(value: "daily" | "papers"): void {
    kind = value;
    for (const tab of Array.from(root.querySelectorAll<HTMLElement>("[data-kind]"))) tab.setAttribute("aria-selected", String(tab.dataset.kind === kind));
    root.classList.remove("is-reading");
    void loadList();
  }

  function reconnect(): void {
    find(".connection-banner").hidden = true;
    void loadStatus(); void loadList(false, true); void pollRun();
    if (selectedPath) void openDocument(selectedPath, "none", location.hash);
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
    const documentButton = target.closest<HTMLElement>("[data-document]");
    if (documentButton) { void openDocument(documentButton.dataset.document!); return; }
    const tab = target.closest<HTMLElement>("[data-kind]");
    if (tab) { switchCollection(tab.dataset.kind as typeof kind); return; }
    const action = target.closest<HTMLElement>("[data-action]")?.dataset.action;
    if (action === "generate") showGeneration();
    else if (action === "settings") showSettings();
    else if (action === "close-dialog") closeDialog();
    else if (action === "reconnect" || action === "refresh") reconnect();
    else if (action === "more") void loadList(true);
    else if (action === "retry-document" && selectedPath) void openDocument(selectedPath, "none", location.hash);
    else if (action === "back") { root.classList.remove("is-reading"); find<HTMLInputElement>('input[type="search"]').focus(); }
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
      searchTimer = setTimeout(() => { void loadList(); }, options.searchDelayMs ?? 180);
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
    switchCollection(event.key === "Home" ? "daily" : event.key === "End" ? "papers" : kind === "daily" ? "papers" : "daily");
    find<HTMLButtonElement>(`[data-kind="${kind}"]`).focus();
  }
  function popstate(): void {
    const path = new URL(location.href).searchParams.get("document");
    if (path) void openDocument(path, "none", location.hash);
    else { documentVersion += 1; selectedPath = ""; root.classList.remove("is-reading"); markSelection(); reading.innerHTML = '<div class="empty-reading"><h1>选择文档继续阅读</h1><p>日报和论文总结保存在左侧列表中。</p></div>'; find(".toc-pane").innerHTML = ""; }
  }
  root.addEventListener("click", click);
  root.addEventListener("input", input);
  root.addEventListener("change", input);
  root.addEventListener("submit", submit);
  root.addEventListener("keydown", keydown);
  window.addEventListener("popstate", popstate);
  void loadStatus(); void loadList(false, true); void pollRun();
  if (selectedPath) void openDocument(selectedPath, "none", location.hash);
  return () => {
    disposed = true;
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

if (typeof document !== "undefined") {
  const app = document.getElementById("app");
  if (app) mountWorkbench(app);
}
