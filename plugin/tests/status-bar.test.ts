import { afterEach, describe, it, expect, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { StatusBarController, withBusyGuard } from "../src/services/status-bar";
import { StateStore } from "@arxiv-daily/core";
import type { RunState } from "@arxiv-daily/core";

const pluginMainSource = readFileSync(resolve(process.cwd(), "main.ts"), "utf-8");
const statusBarSource = readFileSync(
  resolve(process.cwd(), "src/services/status-bar.ts"),
  "utf-8",
);

function installCreateEl(parent: HTMLElement): void {
  const create = function (
    this: HTMLElement,
    tag: string,
    options?: { cls?: string; attr?: Record<string, string> },
  ): HTMLElement {
    const child = this.ownerDocument.createElement(tag);
    if (options?.cls) child.className = options.cls;
    for (const [name, value] of Object.entries(options?.attr ?? {})) {
      child.setAttribute(name, value);
    }
    installCreateEl(child);
    this.append(child);
    return child;
  };
  (parent as any).createEl = create;
  (parent as any).createDiv = function (
    this: HTMLElement,
    options?: { cls?: string; attr?: Record<string, string> },
  ): HTMLElement {
    return create.call(this, "div", options);
  };
}

installCreateEl(document.body);

function makeEl(): HTMLElement {
  return document.createElement("span");
}

function makeStore(initial: RunState = {}): StateStore {
  const data = { runState: { ...initial } };
  return new StateStore(
    async () => ({ runState: { ...data.runState } }),
    async (d) => {
      data.runState = { ...d.runState };
    },
  );
}

describe("StatusBarController", () => {
  afterEach(() => {
    vi.useRealTimers();
    document.body.innerHTML = "";
  });

  it("uses Obsidian scoped element creation in production code", () => {
    expect(statusBarSource).not.toContain(".createElement(");
    expect(statusBarSource).toContain("this.el.ownerDocument.body.createDiv");
  });

  it("renders 'arXiv: disabled' when constructed with disabled state", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: false });
    expect(el.textContent).toBe("arXiv: disabled");
  });

  it("renders 'arXiv: idle' with no history", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    new StatusBarController(el, store, { initiallyEnabled: true });
    expect(el.textContent).toBe("arXiv: idle");
  });

  it("renders 'arXiv: idle · last YYYY-MM-DD' with completed history", async () => {
    const store = makeStore();
    await store.load();
    await store.setRunning("2026-05-10");
    await store.setCompleted("2026-05-10", 5);
    await store.setRunning("2026-05-11");
    await store.setCompleted("2026-05-11", 2);
    const el = makeEl();
    new StatusBarController(el, store, { initiallyEnabled: true });
    expect(el.textContent).toBe("arXiv: idle · last 2026-05-11");
  });

  it("setIdle with weekend reason shows '· weekend'", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });
    ctrl.setIdle(undefined, "weekend");
    expect(el.textContent).toBe("arXiv: idle · weekend");
  });

  it("renders single-date run as 'arXiv: DATE · stage'", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });
    ctrl.setBatch(1, 1, "2026-05-11");
    ctrl.setStage("summarize-daily");
    expect(el.textContent).toBe("arXiv: 2026-05-11 · summarize");
  });

  it("renders batch run as 'arXiv: DATE [n/N] · stage i/n'", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });
    ctrl.setBatch(2, 5, "2026-05-10");
    ctrl.setStage("fetch-content", 3, 8);
    expect(el.textContent).toBe("arXiv: 2026-05-10 [2/5] · fetch 3/8");
  });

  it("shows a floating progress panel while a task runs", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });
    ctrl.setTask("arXiv Daily detail", "2606.12938");
    ctrl.setStage("summarize-detail");

    expect(el.textContent).toBe("arXiv: arXiv Daily detail · detail summary");
    const panel = document.body.querySelector(".arxiv-daily-progress");
    expect(panel?.classList.contains("is-hidden")).toBe(false);
    expect(panel?.textContent).toContain("arXiv Daily detail");
    expect(panel?.textContent).toContain("2606.12938");
    expect(panel?.textContent).toContain("detail summary");
  });

  it("uses the status element document and window for the panel and timer", async () => {
    vi.useFakeTimers();
    const store = makeStore();
    await store.load();
    const frame = document.createElement("iframe");
    document.body.append(frame);
    const secondaryDocument = frame.contentDocument!;
    installCreateEl(secondaryDocument.body);
    const secondaryWindow = frame.contentWindow!;
    const setTimeoutSpy = vi.spyOn(secondaryWindow, "setTimeout");
    const clearTimeoutSpy = vi.spyOn(secondaryWindow, "clearTimeout");
    const el = secondaryDocument.createElement("span");
    secondaryDocument.body.append(el);
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });

    ctrl.setComplete();

    expect(secondaryDocument.body.querySelector(".arxiv-daily-progress")).not.toBeNull();
    expect(document.body.querySelector(":scope > .arxiv-daily-progress")).toBeNull();
    expect(setTimeoutSpy).toHaveBeenCalledWith(expect.any(Function), 4_000);

    ctrl.dispose();

    expect(clearTimeoutSpy).toHaveBeenCalled();
    expect(secondaryDocument.body.querySelector(".arxiv-daily-progress")).toBeNull();
  });

  it("sets progressbar aria value attributes as progress changes", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });
    ctrl.setBatch(1, 1, "2026-05-11");
    ctrl.setStage("fetch-content", 2, 4);

    const progress = document.body.querySelector(
      ".arxiv-daily-progress__track",
    );
    expect(progress?.getAttribute("role")).toBe("progressbar");
    expect(progress?.getAttribute("aria-valuemin")).toBe("0");
    expect(progress?.getAttribute("aria-valuemax")).toBe("100");
    // fetch-content is stage 5 of 8 ((4 + 2/4) / 8 * 100 rounded).
    expect(progress?.getAttribute("aria-valuenow")).toBe("56");
  });

  it("keeps completion panel visible when setIdle follows setComplete", async () => {
    vi.useFakeTimers();
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });

    ctrl.setTask("arXiv Daily report", "2026-06-16");
    ctrl.setComplete("Daily report complete: 2026-06-16");
    ctrl.setIdle("2026-06-16");

    const panel = document.body.querySelector(".arxiv-daily-progress");
    expect(panel?.classList.contains("is-complete")).toBe(true);
    expect(panel?.classList.contains("is-hidden")).toBe(false);

    vi.advanceTimersByTime(1_500);
    expect(panel?.classList.contains("is-hidden")).toBe(false);

    vi.advanceTimersByTime(2_500);
    expect(panel?.classList.contains("is-hidden")).toBe(true);
  });

  it("returns to the idle text after a task ends in complete or error", async () => {
    vi.useFakeTimers();
    const store = makeStore({
      "2026-05-10": { status: "completed", lastAttempt: 1, attempts: 1, papersWritten: 3 },
    });
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });

    ctrl.setTask("Scanning personal library", "Identifying library papers");
    ctrl.setComplete("Personal library scan complete");
    expect(el.textContent).toBe("Complete");
    vi.advanceTimersByTime(4_000);
    expect(el.textContent).toBe("arXiv: idle · last 2026-05-10");

    ctrl.setTask("Indexing library titles and abstracts");
    ctrl.setError("Personal library preparation failed");
    expect(el.textContent).toBe("Failed");
    vi.advanceTimersByTime(10_000);
    expect(el.textContent).toBe("arXiv: idle · last 2026-05-10");

    // The finished task no longer titles a later stage-only update.
    ctrl.setStage("summarize-detail");
    expect(document.body.querySelector(".arxiv-daily-progress__title")?.textContent).toBe("arXiv Daily");
  });

  it("returns to the disabled text when the scheduler is off", async () => {
    vi.useFakeTimers();
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: false });

    ctrl.setTask("Scanning personal library");
    ctrl.setComplete();
    vi.advanceTimersByTime(4_000);

    expect(el.textContent).toBe("arXiv: disabled");
  });

  it("does not let a stale hide timer clobber a task started meanwhile", async () => {
    vi.useFakeTimers();
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });

    ctrl.setComplete();
    ctrl.setTask("Indexing library titles and abstracts", "embedding");
    vi.advanceTimersByTime(10_000);

    expect(el.textContent).toBe("arXiv: Indexing library titles and abstracts · embedding");
  });

  it("withBusyGuard drops idle and disabled updates while another task owns the bar", () => {
    const inner = {
      setTask: vi.fn(), setBatch: vi.fn(), setStage: vi.fn(), setComplete: vi.fn(),
      setError: vi.fn(), setIdle: vi.fn(), setDisabled: vi.fn(),
    };
    let busy = true;
    const guarded = withBusyGuard(inner, () => busy);

    guarded.setIdle("2026-05-10", "weekend");
    guarded.setDisabled();
    guarded.setBatch(1, 1, "2026-05-10");
    expect(inner.setIdle).not.toHaveBeenCalled();
    expect(inner.setDisabled).not.toHaveBeenCalled();
    expect(inner.setBatch).toHaveBeenCalledWith(1, 1, "2026-05-10");

    busy = false;
    guarded.setIdle("2026-05-10", "weekend");
    guarded.setDisabled();
    expect(inner.setIdle).toHaveBeenCalledWith("2026-05-10", "weekend");
    expect(inner.setDisabled).toHaveBeenCalledTimes(1);
  });

  it("setDisabled overrides any prior state", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: true });
    ctrl.setBatch(1, 1, "2026-05-11");
    ctrl.setStage("filter");
    ctrl.setDisabled();
    expect(el.textContent).toBe("arXiv: disabled");
  });

  it("setIdle after disabled re-enables and uses idle text", async () => {
    const store = makeStore();
    await store.load();
    const el = makeEl();
    const ctrl = new StatusBarController(el, store, { initiallyEnabled: false });
    ctrl.setIdle("2026-05-11");
    expect(el.textContent).toBe("arXiv: idle · last 2026-05-11");
  });

  it("dispose clears pending timers and removes the floating panel", async () => {
    vi.useFakeTimers();
    const store = makeStore();
    await store.load();
    const ctrl = new StatusBarController(makeEl(), store, { initiallyEnabled: true });
    ctrl.setComplete();

    expect(document.body.querySelector(".arxiv-daily-progress")).not.toBeNull();
    expect(vi.getTimerCount()).toBe(1);

    ctrl.dispose();

    expect(vi.getTimerCount()).toBe(0);
    expect(document.body.querySelector(".arxiv-daily-progress")).toBeNull();
  });

  it("disposes the status controller when the plugin unloads", () => {
    const unloadBody = pluginMainSource.match(
      /onunload\(\)[\s\S]*?\n  async saveSettings/,
    )?.[0];

    expect(unloadBody).toContain("this.progress instanceof StatusBarController");
    expect(unloadBody).toContain("this.progress.dispose()");
  });
});
