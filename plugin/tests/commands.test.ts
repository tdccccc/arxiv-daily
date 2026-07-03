import { describe, expect, it, vi } from "vitest";
import { bindEnterToButton, registerCommands } from "../src/commands";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";

function makePlugin() {
  const commands: Array<{ id: string; name: string; callback?: () => unknown }> = [];
  return {
    settings: DEFAULT_SETTINGS,
    app: {
      workspace: {
        openLinkText: vi.fn(),
      },
      vault: {
        adapter: {
          exists: vi.fn(async () => false),
        },
      },
    },
    manifest: { id: "arxiv-daily", version: "0.1.0" },
    scheduler: {
      runForDateNow: vi.fn(),
      runAllPending: vi.fn(),
      retryFailedInLookback: vi.fn(),
      forceRunForDate: vi.fn(),
      activeRuns: vi.fn(() => []),
      cancelCurrentRun: vi.fn(() => []),
    },
    stateStore: { clearAll: vi.fn(), snapshot: vi.fn(() => ({})) },
    runHistoryStore: { readLatest: vi.fn(async () => []) },
    progress: { setIdle: vi.fn() },
    logger: { warn: vi.fn() },
    manualFetch: { fetchAndSummarize: vi.fn() },
    buildPaperIndex: vi.fn(() => ({
      paths: {
        papersJsonPath: "arxiv-daily/.index/papers.json",
        legacyPapersJsonPath: "arxiv-daily/papers.json",
      },
      get: vi.fn(),
      setStatus: vi.fn(),
      setPriority: vi.fn(),
    })),
    addCommand: vi.fn((command) => {
      commands.push(command);
    }),
    addRibbonIcon: vi.fn(() => ({ addClass: vi.fn() })),
  };
}

describe("registerCommands", () => {
  it("registers the run history viewer command", () => {
    const plugin = makePlugin();

    registerCommands(plugin as any);

    expect(plugin.addCommand).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "arxiv-daily-show-run-history",
        name: "Show run history",
      }),
    );
  });

  it("binds Enter in a single-field modal input to the submit button", () => {
    const input = document.createElement("input");
    const button = document.createElement("button");
    const click = vi.spyOn(button, "click");

    bindEnterToButton(input, button);
    input.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );

    expect(click).toHaveBeenCalledTimes(1);
  });
});
