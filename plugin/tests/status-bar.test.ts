import { describe, it, expect } from "vitest";
import { StatusBarController } from "../src/services/status-bar";
import { StateStore } from "../src/services/state-store";
import type { RunState } from "../src/settings/types";

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
