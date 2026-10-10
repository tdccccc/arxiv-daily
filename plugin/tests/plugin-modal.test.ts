import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { App, Modal } from "obsidian";
import { ArxivDailyModal, ARXIV_DAILY_MODAL_CLASS, markArxivDailyModal } from "../src/plugin-modal";
import { openDatePickerModal } from "../src/date-picker-modal";
import { DiagnosticsModal, FullTextSearchResultsModal } from "../src/commands";
import { HubModal } from "../src/dashboard/hub-modal";
import { SimilarPapersModal } from "../src/dashboard/similar-papers-modal";
import { PersonalLibraryInterestProfileModal } from "../src/library/interest-profile-modal";

// Obsidian extends HTMLElement with `addClass`; the global mock
// (plugin/tests/__mocks__/obsidian.ts) patches HTMLElement.prototype.addClass
// once, so no per-file patch is required here. `createEl`/`createDiv` are
// only needed below because opening DatePickerModal (there is no exported
// class to construct without calling onOpen) renders real content; same
// pattern as date-picker-modal.test.ts.
Object.assign(HTMLElement.prototype, {
  createEl(this: HTMLElement, tag: string, options: { text?: string; cls?: string; attr?: Record<string, string> } = {}) {
    const element = document.createElement(tag);
    if (options.text) element.textContent = options.text;
    if (options.cls) element.className = options.cls;
    for (const [name, value] of Object.entries(options.attr ?? {})) element.setAttribute(name, value);
    this.appendChild(element);
    return element;
  },
  createDiv(this: HTMLElement, options = {}) { return (this as any).createEl("div", options); },
  empty(this: HTMLElement) { this.replaceChildren(); },
});

describe("ArxivDailyModal / markArxivDailyModal (plugin/src/plugin-modal.ts)", () => {
  it("ArxivDailyModal tags modalEl with the shared marker in its constructor, before onOpen runs", () => {
    class Probe extends ArxivDailyModal {
      onOpen() {
        // The marker must already be present by the time onOpen runs.
        expect(this.modalEl.classList.contains(ARXIV_DAILY_MODAL_CLASS)).toBe(true);
      }
    }
    const modal = new Probe({} as App);
    expect(modal.modalEl.classList.contains(ARXIV_DAILY_MODAL_CLASS)).toBe(true);
    modal.open();
  });

  it("markArxivDailyModal() tags a bare `new Modal(app)` instance", () => {
    const modal = new Modal({} as App);
    expect(modal.modalEl.classList.contains(ARXIV_DAILY_MODAL_CLASS)).toBe(false);
    markArxivDailyModal(modal);
    expect(modal.modalEl.classList.contains(ARXIV_DAILY_MODAL_CLASS)).toBe(true);
  });

  it("keeps an existing per-modal class alongside the shared marker", () => {
    class Probe extends ArxivDailyModal {}
    const modal = new Probe({} as App);
    modal.modalEl.addClass("some-other-modal-class");
    expect(modal.modalEl.classList.contains(ARXIV_DAILY_MODAL_CLASS)).toBe(true);
    expect(modal.modalEl.classList.contains("some-other-modal-class")).toBe(true);
  });
});

describe("real plugin modals carry the shared marker", () => {
  it("DatePickerModal (opened via openDatePickerModal)", () => {
    Modal.opened.length = 0;
    openDatePickerModal({} as App, () => {}, {}, () => {});
    const modal = Modal.opened.at(-1)!;
    expect(modal.modalEl.classList.contains(ARXIV_DAILY_MODAL_CLASS)).toBe(true);
  });

  it("FullTextSearchResultsModal", () => {
    const modal = new FullTextSearchResultsModal({} as App, [], {} as any);
    expect(modal.modalEl.classList.contains(ARXIV_DAILY_MODAL_CLASS)).toBe(true);
  });

  it("DiagnosticsModal", () => {
    const plugin = { app: {}, logger: { warn: () => {} } } as any;
    const modal = new DiagnosticsModal({} as App, plugin);
    expect(modal.modalEl.classList.contains(ARXIV_DAILY_MODAL_CLASS)).toBe(true);
  });

  it("HubModal (the marker is already present before onOpen adds its own .arxiv-daily-hub-modal class)", () => {
    const plugin = {
      stateStore: { snapshot: () => ({}) },
      runHistoryStore: { readLatest: async () => [] },
      logger: { warn: () => {}, info: () => {} },
    } as any;
    const modal = new HubModal({} as App, plugin);
    expect(modal.modalEl.classList.contains(ARXIV_DAILY_MODAL_CLASS)).toBe(true);
  });

  it("SimilarPapersModal", () => {
    const modal = new SimilarPapersModal({} as App, {
      source: { arxivId: "1", title: "t" } as any,
      results: [],
      openDetail: () => {},
      openDaily: () => {},
      openArxiv: () => {},
      openPdf: () => {},
    });
    expect(modal.modalEl.classList.contains(ARXIV_DAILY_MODAL_CLASS)).toBe(true);
  });

  it("PersonalLibraryInterestProfileModal", () => {
    const controller = { snapshot: () => ({}) } as any;
    const modal = new PersonalLibraryInterestProfileModal({} as App, controller);
    expect(modal.modalEl.classList.contains(ARXIV_DAILY_MODAL_CLASS)).toBe(true);
  });
});

/**
 * Backstop for every current and future `Modal` use in plugin/src, not just
 * the ones exercised directly above: every `extends Modal` declaration must
 * actually read `extends ArxivDailyModal` (plugin-modal.ts's own class
 * declaration is the one permitted exception), and every bare `new Modal(`
 * construction must be paired with a `markArxivDailyModal(...)` call nearby.
 * This is what the chunk-2 contract calls "test via the shared helper/base
 * class plus a grep-style guard test" for the plugin modals that are
 * one-off Promise-wrapped confirmation dialogs (library/modal.ts,
 * services/modal.ts, settings/tab.ts) and would otherwise need heavy
 * App/plugin fixtures just to prove a one-line marker call fired.
 */
describe("every Modal use in plugin/src goes through the shared marker (static guard)", () => {
  const root = join(__dirname, "..", "src");
  const files: string[] = [];
  (function walk(dir: string) {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      if (statSync(full).isDirectory()) walk(full);
      else if (entry.endsWith(".ts")) files.push(full);
    }
  })(root);

  it("no file other than plugin-modal.ts declares `extends Modal` directly", () => {
    const offenders: string[] = [];
    for (const file of files) {
      if (relative(root, file) === "plugin-modal.ts") continue;
      const source = readFileSync(file, "utf8");
      if (source.includes("extends Modal")) offenders.push(relative(root, file));
    }
    expect(offenders).toEqual([]);
  });

  it("every bare `new Modal(` construction is immediately marked", () => {
    const unmarked: string[] = [];
    for (const file of files) {
      if (relative(root, file) === "plugin-modal.ts") continue;
      const source = readFileSync(file, "utf8");
      let index = source.indexOf("new Modal(");
      while (index !== -1) {
        const after = source.slice(index, index + 200);
        if (!after.includes("markArxivDailyModal(")) {
          unmarked.push(`${relative(root, file)}@${index}`);
        }
        index = source.indexOf("new Modal(", index + 1);
      }
    }
    expect(unmarked).toEqual([]);
  });
});
