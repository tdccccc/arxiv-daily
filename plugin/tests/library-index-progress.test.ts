/**
 * The Library row while an index run is in flight, and what it says after one.
 *
 * These are the statements happy-dom can settle: which buttons the row asks
 * for, what they read, what the sentence beside them claims. Whether the row is
 * still legible once those buttons are on it is settled in the real renderer by
 * `scripts/desktop-acceptance/library-settings.mjs`, which has a layout engine.
 */
import { describe, expect, it, vi } from "vitest";
import {
  libraryRowPresentation,
  lastIndexedSentence,
  type LibraryConnectionStatus,
} from "../src/library/connection";
import { LibraryIndexStatusStore } from "../src/library/index-status";

const CONNECTED: LibraryConnectionStatus = {
  kind: "authorization-required",
  rootLabel: "papers",
};

const AUTHORIZED: LibraryConnectionStatus = {
  kind: "authorized",
  rootLabel: "papers",
  grantedAt: "2026-08-02T12:00:00.000Z",
};

function buttons(row: ReturnType<typeof libraryRowPresentation>) {
  return [row.chooseFolder, row.primary, row.cancel, row.revoke]
    .filter((button) => button !== undefined)
    .map((button) => button.label);
}

describe("the Library row while indexing", () => {
  it("puts the count on the main button and stops offering to start again", () => {
    const row = libraryRowPresentation({
      status: CONNECTED,
      embeddingMode: "local",
      activity: { phase: "extracting and embedding PDF text", completed: 5, total: 120, cancelling: false },
    });
    expect(row.primary).toEqual({ label: "Indexing… (5/120)", disabled: true });
    expect(row.cancel).toEqual({ label: "Cancel", disabled: false });
    expect(row.description).toContain("extracting and embedding PDF text");
  });

  /**
   * A run spends its first moments in phases that count nothing. A fraction
   * invented for those would be the one number on the row that is not true.
   */
  it("drops the fraction while the phase counts nothing", () => {
    const row = libraryRowPresentation({
      status: CONNECTED,
      embeddingMode: "local",
      activity: { phase: "reading the library catalog", cancelling: false },
    });
    expect(row.primary?.label).toBe("Indexing…");
  });

  /**
   * Cancelling can take as long as the step it interrupts. A button that still
   * reads "Cancel" invites a second press at a request already sent.
   */
  it("says it is stopping once cancellation has been asked for", () => {
    const row = libraryRowPresentation({
      status: CONNECTED,
      embeddingMode: "local",
      activity: { phase: "extracting and embedding PDF text", completed: 5, total: 120, cancelling: true },
    });
    expect(row.cancel).toEqual({ label: "Cancelling…", disabled: true });
    expect(row.description).toContain("Stopping the index run");
  });

  /**
   * Changing the folder mid-run aborts it at the next identity check, which on
   * a large library can be hours in. The row does not offer that until the run
   * is over.
   */
  it("does not offer the folder picker while a run is in flight", () => {
    const row = libraryRowPresentation({
      status: CONNECTED,
      embeddingMode: "local",
      activity: { phase: "extracting and embedding PDF text", cancelling: false },
    });
    expect(row.chooseFolder.disabled).toBe(true);
  });

  /** Nothing is committed until the run ends, and the row has to say so. */
  it("warns that a cancelled run leaves nothing behind", () => {
    const row = libraryRowPresentation({
      status: CONNECTED,
      embeddingMode: "local",
      activity: { phase: "extracting and embedding PDF text", cancelling: false },
    });
    expect(row.description).toContain("Nothing is saved until the run finishes");
  });

  /**
   * The row's standing limit is three buttons. Cancel is a fourth unless
   * something steps aside, and Revoke is the one with no meaning mid-run.
   */
  it("keeps at most three buttons in every state", () => {
    const states: Array<Parameters<typeof libraryRowPresentation>[0]> = [];
    for (const status of [{ kind: "disconnected" } as const, CONNECTED, AUTHORIZED]) {
      for (const embeddingMode of ["local", "remote"] as const) {
        for (const activity of [
          undefined,
          { phase: "reading the library catalog", cancelling: false },
          { phase: "extracting and embedding PDF text", completed: 2, total: 3, cancelling: true },
        ]) {
          for (const lastRun of [undefined, { updatedAt: "2026-08-30T21:15:00.000Z", papers: 12 }]) {
            states.push({
              status,
              embeddingMode,
              ...(activity ? { activity } : {}),
              ...(lastRun ? { lastRun } : {}),
            });
          }
        }
      }
    }
    for (const state of states) {
      const row = libraryRowPresentation(state);
      expect(buttons(row).length).toBeLessThanOrEqual(3);
      expect(buttons(row).some((label) => /authorize/i.test(label))).toBe(false);
    }
  });

  it("hides Revoke on a granted library while its index is building", () => {
    const idle = libraryRowPresentation({ status: AUTHORIZED, embeddingMode: "remote" });
    expect(buttons(idle)).toEqual(["Change folder", "Build index", "Revoke"]);
    const running = libraryRowPresentation({
      status: AUTHORIZED,
      embeddingMode: "remote",
      activity: { phase: "extracting and embedding PDF text", completed: 1, total: 4, cancelling: false },
    });
    expect(buttons(running)).toEqual(["Change folder", "Indexing… (1/4)", "Cancel"]);
  });
});

describe("what an index run leaves on the row", () => {
  /**
   * The status bar hides its completion after four seconds and the notice after
   * ten, so this sentence is the only thing an overnight run leaves behind.
   */
  it("names when the index was last built and how much it holds", () => {
    const row = libraryRowPresentation({
      status: CONNECTED,
      embeddingMode: "local",
      lastRun: { updatedAt: "2026-08-30T21:15:00.000Z", papers: 128 },
    });
    expect(row.description).toContain("128 papers searchable");
    expect(row.description).toContain("Last indexed");
    // The next step it replaced is still there; the trace is added to it.
    expect(row.description).toContain("Build the search index");
  });

  it("says nothing about past runs when there have been none", () => {
    const row = libraryRowPresentation({ status: CONNECTED, embeddingMode: "local" });
    expect(row.description).not.toContain("Last indexed");
  });

  it("counts one paper as one paper", () => {
    expect(lastIndexedSentence({ updatedAt: "2026-08-30T21:15:00.000Z", papers: 1 }))
      .toContain("1 paper searchable");
  });

  /** Minute precision, local clock: the question is "did last night's run finish". */
  it("reads the timestamp in local time, to the minute", () => {
    const sentence = lastIndexedSentence({ updatedAt: "2026-08-30T21:15:00.000Z", papers: 3 });
    const local = new Date("2026-08-30T21:15:00.000Z");
    const pad = (value: number) => String(value).padStart(2, "0");
    expect(sentence).toContain(
      `${local.getFullYear()}-${pad(local.getMonth() + 1)}-${pad(local.getDate())} `
      + `${pad(local.getHours())}:${pad(local.getMinutes())}`,
    );
  });

  /** A clock the manifest cannot parse must not put "Invalid Date" on the row. */
  it("drops the timestamp rather than printing a broken one", () => {
    const sentence = lastIndexedSentence({ updatedAt: "not a date", papers: 4 });
    expect(sentence).toBe("4 papers searchable.");
  });
});

describe("the store the row subscribes to", () => {
  it("replays the current status to a new subscriber", () => {
    const store = new LibraryIndexStatusStore();
    store.beginRun("op:1", "reading the library catalog");
    const seen: unknown[] = [];
    store.subscribe((status) => seen.push(status.activity?.phase));
    expect(seen).toEqual(["reading the library catalog"]);
  });

  /**
   * A run reports once per paper. Re-rendering for a report that changed nothing
   * would be the settings page redrawing itself for no reason several times a
   * second.
   */
  it("stays quiet when a report repeats what the row already shows", () => {
    const store = new LibraryIndexStatusStore();
    store.beginRun("op:1", "indexing");
    const listener = vi.fn();
    store.subscribe(listener);
    listener.mockClear();
    store.report({ phase: "indexing", completed: 1, total: 3 });
    store.report({ phase: "indexing", completed: 1, total: 3 });
    expect(listener).toHaveBeenCalledTimes(1);
  });

  /** A phase that counts nothing must not inherit the previous phase's count. */
  it("drops the counts when a phase stops counting", () => {
    const store = new LibraryIndexStatusStore();
    store.beginRun("op:1", "indexing");
    store.report({ phase: "indexing", completed: 7, total: 9 });
    store.report({ phase: "validating full-text generation" });
    expect(store.snapshot().activity).toEqual({
      operationId: "op:1",
      phase: "validating full-text generation",
      cancelling: false,
    });
  });

  it("ignores reports that arrive after the run ended", () => {
    const store = new LibraryIndexStatusStore();
    store.beginRun("op:1", "indexing");
    store.endRun();
    store.report({ phase: "indexing", completed: 1, total: 2 });
    expect(store.snapshot().activity).toBeUndefined();
  });

  /** The trace outlives the run that produced it; that is its whole purpose. */
  it("keeps the last run's trace after the run ends", () => {
    const store = new LibraryIndexStatusStore();
    store.beginRun("op:1", "indexing");
    store.setLastRun({ updatedAt: "2026-08-30T21:15:00.000Z", papers: 12 });
    store.endRun();
    expect(store.snapshot()).toEqual({
      lastRun: { updatedAt: "2026-08-30T21:15:00.000Z", papers: 12 },
    });
  });

  /** A row that throws while drawing progress must not stop the run reporting. */
  it("survives a listener that throws", () => {
    const store = new LibraryIndexStatusStore();
    const second = vi.fn();
    store.subscribe(() => { throw new Error("render failed"); });
    store.subscribe(second);
    expect(() => store.beginRun("op:1", "indexing")).not.toThrow();
    expect(second).toHaveBeenCalled();
  });
});
