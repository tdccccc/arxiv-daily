import { afterEach, describe, it, expect, vi } from "vitest";
import { StatusBarController } from "../src/services/status-bar";
import { StateStore } from "@arxiv-daily/core";
import type { RunState } from "@arxiv-daily/core";

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
});
