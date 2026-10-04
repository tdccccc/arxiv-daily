import { t } from "./i18n";
import type { WorkbenchCalendar, WorkbenchCalendarDay } from "../calendar";

export const calendarStateLabels: Record<WorkbenchCalendarDay["state"], string> = {
  "has-report": "已有日报", "not-generated": "尚未生成", running: "正在生成", failed: "生成失败",
  "no-matches": "无匹配论文", skipped: "已跳过", "report-missing": "日报文件缺失", future: "未来日期",
};

interface CalendarOptions {
  request: <T>(url: string) => Promise<T>;
  selectDay: (day: WorkbenchCalendarDay) => void;
  updateDay: (day: WorkbenchCalendarDay) => void;
  onError: (error: unknown) => void;
}

/** Calendar navigation is read-only. The app owns document routes and explicit generation. */
export function mountCalendar(host: HTMLElement, options: CalendarOptions) {
  let data: WorkbenchCalendar | null = null;
  let selectedDate = "";
  let focusedDate = "";
  let version = 0;
  let requestedMonth: string | undefined;
  let disposed = false;
  const mobile = window.matchMedia("(max-width: 760px)");
  let expanded = !mobile.matches;

  host.innerHTML = `<div class="calendar-mobile-heading"><span>${t("按日期阅读")}</span><button class="quiet-button" data-calendar-action="toggle" aria-controls="daily-calendar-body"></button></div>
    <div class="calendar-body" id="daily-calendar-body"><div class="calendar-navigation"><button class="icon-button" data-calendar-action="previous" aria-label="${t("上个月")}">‹</button><strong class="calendar-month" aria-live="polite">${t("日报日历")}</strong><button class="icon-button" data-calendar-action="next" aria-label="${t("下个月")}">›</button><button class="quiet-button calendar-today" data-calendar-action="today">${t("回到今天")}</button></div>
    <div class="calendar-error" role="alert" hidden></div><div class="calendar-weekdays" aria-hidden="true"><span>${t("一")}</span><span>${t("二")}</span><span>${t("三")}</span><span>${t("四")}</span><span>${t("五")}</span><span>${t("六")}</span><span>${t("日")}</span></div>
    <div class="calendar-grid" role="group" aria-label="${t("日报日期")}"><div class="calendar-loading" role="status">${t("正在读取日历…")}</div></div>
    <div class="calendar-legend" aria-label="${t("日历状态图例")}"><span><i class="calendar-dot has-report"></i>${t("有日报")}</span><span><i class="calendar-dot not-generated"></i>${t("可生成")}</span><span><i class="calendar-dot running"></i>${t("生成中")}</span><span><i class="calendar-dot failed"></i>${t("需处理")}</span><span><i class="calendar-dot no-matches"></i>${t("无匹配")}</span></div>
    <div class="calendar-summary" aria-live="polite"></div></div>`;
  const find = <T extends HTMLElement = HTMLElement>(selector: string) => host.querySelector<T>(selector)!;

  function renderExpansion(): void {
    find(".calendar-body").hidden = !expanded;
    const toggle = find<HTMLButtonElement>('[data-calendar-action="toggle"]');
    toggle.textContent = expanded ? t("收起日历") : t("展开日历");
    toggle.setAttribute("aria-expanded", String(expanded));
  }

  function render(): void {
    if (!data) return;
    const active = document.activeElement instanceof HTMLElement && host.contains(document.activeElement) ? document.activeElement : null;
    const restoreDate = active?.dataset.calendarDate;
    const days = data.cells.filter((day): day is WorkbenchCalendarDay => day !== null);
    const target = [focusedDate, selectedDate, data.today, days[0]?.date].find(date => days.some(day => day.date === date));
    find(".calendar-month").textContent = t("{0} 年 {1} 月", Number(data.month.slice(0, 4)), Number(data.month.slice(5)));
    find(".calendar-grid").setAttribute("aria-label", t("{0} 日报日期", data.month));
    find<HTMLButtonElement>('[data-calendar-action="previous"]').disabled = data.previousMonth === data.month;
    find<HTMLButtonElement>('[data-calendar-action="next"]').disabled = data.nextMonth === data.month;
    find(".calendar-grid").innerHTML = data.cells.map(day => {
      if (!day) return '<span class="calendar-blank" aria-hidden="true"></span>';
      const showCount = day.state === "has-report" || day.state === "no-matches";
      const countLabel = showCount ? day.papers === null ? t("，论文数未知") : t("，{0} 篇论文", day.papers) : "";
      const caption = showCount
        ? `<span class="calendar-day-count">${day.papers === null ? "—" : t("{0}篇", day.papers)}</span>`
        : `<span class="calendar-day-mark" aria-hidden="true">${day.state === "not-generated" ? "+" : day.state === "running" ? "…" : day.state === "failed" || day.state === "report-missing" ? "!" : day.state === "skipped" ? "–" : ""}</span>`;
      return `<button class="calendar-day ${day.state}${day.date === selectedDate ? " is-selected" : ""}${day.date === data!.today ? " is-today" : ""}" data-calendar-date="${day.date}" tabindex="${day.date === target ? 0 : -1}" aria-label="${day.date} · ${t(calendarStateLabels[day.state])}${countLabel}" title="${day.date} · ${t(calendarStateLabels[day.state])}${countLabel}" aria-pressed="${day.date === selectedDate}" ${day.date === data!.today ? 'aria-current="date"' : ""}><span class="calendar-day-number">${Number(day.date.slice(8))}</span>${caption}</button>`;
    }).join("");
    const selected = days.find(day => day.date === selectedDate);
    find(".calendar-summary").innerHTML = selected ? `<div class="calendar-selected-heading"><time datetime="${selected.date}">${selected.date}</time><span>${t(calendarStateLabels[selected.state])}${selected.papers !== null ? ` · ${t("{0} 篇", selected.papers)}` : ""}</span></div>${selected.canGenerate ? `<button class="quiet-button" data-action="generate-date" data-date="${selected.date}">${escapeHtml(t(selected.actionLabel || "生成日报"))} <span aria-hidden="true">↗</span></button>` : ""}` : `<span class="calendar-hint">${t("选择日期查看日报或运行状态")}</span>`;
    if (restoreDate) focusDate(restoreDate);
  }

  function selectedDay(): WorkbenchCalendarDay | undefined { return data?.cells.find((day): day is WorkbenchCalendarDay => day?.date === selectedDate); }

  async function load(month?: string): Promise<boolean> {
    const current = ++version;
    requestedMonth = month ?? data?.today.slice(0, 7);
    host.setAttribute("aria-busy", "true");
    find(".calendar-error").hidden = true;
    try {
      const result = await options.request<WorkbenchCalendar>(`api/calendar${month ? `?month=${encodeURIComponent(month)}` : ""}`);
      if (disposed || current !== version) return false;
      data = result;
      requestedMonth = result.month;
      if (!selectedDate) selectedDate = result.today;
      render();
      const day = selectedDay();
      if (day) options.updateDay(day);
      return true;
    } catch (error) {
      if (!disposed && current === version) {
        find(".calendar-error").innerHTML = `<span>${t("日历暂时不可用")}</span><button class="quiet-button" data-calendar-action="retry">${t("重试")}</button>`;
        find(".calendar-error").hidden = false;
        if (!data) find(".calendar-grid").innerHTML = "";
        options.onError(error);
      }
      return false;
    } finally { if (!disposed && current === version) host.setAttribute("aria-busy", "false"); }
  }

  async function selectDate(date: string): Promise<WorkbenchCalendarDay | undefined> {
    selectedDate = date;
    const month = date.slice(0, 7);
    if (data?.month !== month) {
      if (!await load(month)) return;
    } else {
      supersedeNavigation();
      render();
    }
    return selectedDate === date ? selectedDay() : undefined;
  }

  function chooseDay(day: WorkbenchCalendarDay): void {
    selectedDate = day.date;
    focusedDate = day.date;
    supersedeNavigation();
    render();
    options.selectDay(day);
  }

  function supersedeNavigation(): void {
    // Keep a same-month refresh alive so its current state follows the latest selection.
    if (requestedMonth === data?.month) return;
    version += 1;
    requestedMonth = data?.month;
    host.setAttribute("aria-busy", "false");
  }

  function focusDate(date: string): void {
    const button = host.querySelector<HTMLButtonElement>(`[data-calendar-date="${date}"]`);
    if (!button) return;
    focusedDate = date;
    for (const day of Array.from(host.querySelectorAll<HTMLButtonElement>("[data-calendar-date]"))) day.tabIndex = day === button ? 0 : -1;
    button.focus();
  }

  function click(event: MouseEvent): void {
    const target = event.target instanceof Element ? event.target : null;
    const date = target?.closest<HTMLElement>("[data-calendar-date]")?.dataset.calendarDate;
    const day = date ? data?.cells.find((value): value is WorkbenchCalendarDay => value?.date === date) : undefined;
    if (day) { chooseDay(day); return; }
    const action = target?.closest<HTMLElement>("[data-calendar-action]")?.dataset.calendarAction;
    if (action === "toggle") { expanded = !expanded; renderExpansion(); }
    else if (action === "previous" && data) void load(data.previousMonth);
    else if (action === "next" && data) void load(data.nextMonth);
    else if (action === "retry") void load(requestedMonth);
    else if (action === "today") void (async () => {
      // Fetch the configured timezone's current day again, including across midnight.
      if (!await load() || !data) return;
      const today = data.cells.find((value): value is WorkbenchCalendarDay => value?.date === data!.today);
      if (today) chooseDay(today);
    })();
  }

  async function keydown(event: KeyboardEvent): Promise<void> {
    const date = event.target instanceof HTMLElement ? event.target.closest<HTMLElement>("[data-calendar-date]")?.dataset.calendarDate : undefined;
    if (!date || !data || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const days = data.cells.filter((day): day is WorkbenchCalendarDay => day !== null);
    const index = days.findIndex(day => day.date === date);
    if (event.key === "Home" || event.key === "End") {
      const cell = data.cells.findIndex(day => day?.date === date);
      const row = data.cells.slice(Math.floor(cell / 7) * 7, Math.floor(cell / 7) * 7 + 7).filter((day): day is WorkbenchCalendarDay => day !== null);
      const target = event.key === "Home" ? row[0] : row[row.length - 1];
      if (target) focusDate(target.date);
      return;
    }
    const offset = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -7, ArrowDown: 7 }[event.key] ?? 0;
    const target = index + offset;
    if (days[target]) { focusDate(days[target].date); return; }
    const month = target < 0 ? data.previousMonth : data.nextMonth;
    if (month === data.month || !await load(month) || !data) return;
    const adjacent = data.cells.filter((day): day is WorkbenchCalendarDay => day !== null);
    const adjacentDay = adjacent[target < 0 ? adjacent.length + target : target - days.length];
    if (adjacentDay) focusDate(adjacentDay.date);
  }

  const resize = () => { expanded = !mobile.matches; renderExpansion(); };
  renderExpansion();
  host.addEventListener("click", click);
  host.addEventListener("keydown", keydown);
  mobile.addEventListener?.("change", resize);
  return {
    load, selectDate,
    refresh: () => load(requestedMonth ?? data?.month),
    clearSelection: () => { selectedDate = ""; render(); },
    dispose: () => { disposed = true; version += 1; host.removeEventListener("click", click); host.removeEventListener("keydown", keydown); mobile.removeEventListener?.("change", resize); },
  };
}

function escapeHtml(value: string): string { return value.replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]!); }
