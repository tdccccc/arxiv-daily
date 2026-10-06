import { t } from "./i18n";
export interface SidebarOptions {
  request: <T>(url: string, body?: unknown) => Promise<T>;
}

interface Preferences { sidebarWidth: number | null; sidebarCollapsed: boolean }

export function mountSidebar(root: HTMLElement, options: SidebarOptions): () => void {
  const pane = root.querySelector<HTMLElement>(".library-pane")!;
  const workspace = root.querySelector<HTMLElement>(".workspace")!;
  let preferences: Preferences = { sidebarWidth: null, sidebarCollapsed: false };
  let revision = 0, pointer: number | null = null, disposed = false, saving = false, dirty = false;
  pane.id ||= "sidebar-navigation";
  const divider = document.createElement("div"); divider.className = "sidebar-resize";
  divider.tabIndex = 0; divider.setAttribute("role", "separator"); divider.setAttribute("aria-orientation", "vertical");
  divider.setAttribute("aria-label", t("调整侧栏宽度")); divider.setAttribute("aria-controls", pane.id);
  divider.title = t("拖动或使用左右方向键调整宽度；双击恢复默认");
  pane.after(divider);
  const toggle = document.createElement("button"); toggle.className = "quiet-button sidebar-toggle";
  toggle.setAttribute("aria-controls", pane.id); root.querySelector(".header-actions")!.prepend(toggle);
  const notice = document.createElement("span"); notice.className = "sidebar-save-status";
  notice.setAttribute("role", "status"); notice.hidden = true; toggle.after(notice);
  const bounds = () => {
    const toc = root.dataset.view === "reading" && window.innerWidth > 1360 ? window.innerWidth >= 1600 ? 220 : 190 : 0;
    return { min: 280, max: Math.max(280, Math.min(900, window.innerWidth - toc - 428)) };
  };
  function width(): number {
    const { min, max } = bounds();
    return Math.round(Math.max(min, Math.min(max, preferences.sidebarWidth ?? Math.min(640, window.innerWidth * .35))));
  }
  function render(): void {
    const mobile = window.innerWidth <= 760;
    root.style.setProperty("--sidebar-width", `${width()}px`);
    root.classList.toggle("sidebar-collapsed", preferences.sidebarCollapsed);
    divider.hidden = mobile || preferences.sidebarCollapsed;
    divider.setAttribute("aria-valuenow", String(width()));
    divider.setAttribute("aria-valuemin", String(bounds().min)); divider.setAttribute("aria-valuemax", String(bounds().max));
    toggle.textContent = preferences.sidebarCollapsed ? t("展开侧栏") : t("收起侧栏");
    toggle.setAttribute("aria-expanded", String(!preferences.sidebarCollapsed));
  }
  function error(text: string): void { if (!disposed) { notice.hidden = false; notice.textContent = text; } }
  async function save(): Promise<void> {
    dirty = true;
    if (saving) return;
    saving = true;
    while (dirty && !disposed) {
      dirty = false;
      try { await options.request("api/preferences", { ...preferences }); if (!disposed) notice.hidden = true; }
      catch { error(t("布局偏好未保存，当前页面仍可使用")); }
    }
    saving = false;
  }
  function choose(value: number | null, persist = true): void {
    revision += 1;
    preferences.sidebarWidth = value === null ? null : Math.max(bounds().min, Math.min(bounds().max, Math.round(value)));
    render(); if (persist) void save();
  }
  const down = (event: PointerEvent) => {
    if (event.button !== 0 || window.innerWidth <= 760) return;
    event.preventDefault(); pointer = event.pointerId; revision += 1;
    divider.setPointerCapture?.(event.pointerId); root.classList.add("is-resizing");
  };
  const move = (event: PointerEvent) => { if (pointer === event.pointerId) choose(event.clientX - workspace.getBoundingClientRect().left, false); };
  const up = (event: PointerEvent) => {
    if (pointer !== event.pointerId) return;
    pointer = null; root.classList.remove("is-resizing"); void save();
  };
  const key = (event: KeyboardEvent) => {
    const step = event.shiftKey ? 32 : 16;
    if (event.key === "ArrowLeft") choose(width() - step);
    else if (event.key === "ArrowRight") choose(width() + step);
    else if (event.key === "Home") choose(bounds().min);
    else if (event.key === "End") choose(bounds().max);
    else return;
    event.preventDefault();
  };
  const collapse = () => { revision += 1; preferences.sidebarCollapsed = !preferences.sidebarCollapsed; render(); void save(); };
  const reset = () => choose(null);
  divider.addEventListener("pointerdown", down); divider.addEventListener("keydown", key); divider.addEventListener("dblclick", reset);
  toggle.addEventListener("click", collapse);
  window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up); window.addEventListener("resize", render);
  const observer = new MutationObserver(render); observer.observe(root, { attributes: true, attributeFilter: ["data-view"] });
  render();
  const initialRevision = revision;
  void options.request<Preferences>("api/preferences").then(value => {
    if (disposed || revision !== initialRevision) return;
    preferences = { sidebarWidth: typeof value.sidebarWidth === "number" && Number.isFinite(value.sidebarWidth) && value.sidebarWidth >= 280 && value.sidebarWidth <= 900 ? value.sidebarWidth : null, sidebarCollapsed: value.sidebarCollapsed === true };
    render();
  }).catch(() => error(t("无法读取布局偏好，已使用默认布局")));
  return () => {
    disposed = true; observer.disconnect();
    divider.removeEventListener("pointerdown", down); divider.removeEventListener("keydown", key); divider.removeEventListener("dblclick", reset);
    toggle.removeEventListener("click", collapse);
    window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up); window.removeEventListener("resize", render);
    root.classList.remove("is-resizing"); divider.remove(); toggle.remove(); notice.remove();
  };
}
