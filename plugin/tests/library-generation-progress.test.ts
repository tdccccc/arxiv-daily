import { describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SETTINGS,
  OperationRegistry,
  createEmptyPersonalLibraryCatalog,
  createPersonalLibraryIdentificationFingerprint,
  createPersonalLibraryScopeFingerprint,
} from "@arxiv-daily/core";
import ArxivDailyPlugin from "../main.ts";
import { authorizeLibraryConnection, createLibraryConnection } from "../src/library/connection";

/**
 * Direction generation writes a task into the status bar as it runs. Whoever
 * writes one has to take it back out: this reported progress but never an
 * ending, so a finished or failed run left the bar reading "naming directions
 * (5/9)" until some later task happened to replace it — visible long after the
 * review modal had been closed.
 */
function makePlugin() {
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.embedding.mode = "local";
  const connection = authorizeLibraryConnection(
    createLibraryConnection("/private/library", "1:2"),
    { llmBaseUrl: settings.llm.baseUrl },
  );
  const catalog = createEmptyPersonalLibraryCatalog(
    createPersonalLibraryScopeFingerprint({ rootIdentity: "1:2", eligibleExtensions: [".pdf"] }),
    createPersonalLibraryIdentificationFingerprint([".pdf"]),
  );
  const progress = {
    setTask: vi.fn(), setComplete: vi.fn(), setError: vi.fn(),
    setIdle: vi.fn(), setDisabled: vi.fn(),
  };
  const plugin = Object.create(ArxivDailyPlugin.prototype) as ArxivDailyPlugin;
  Object.assign(plugin, {
    settings,
    logger: { warn: vi.fn(), error: vi.fn(), setSensitiveValues: vi.fn() },
    host: { storage: {}, http: {}, markupParser: {} },
    progress,
    operations: new OperationRegistry(),
    libraryConnection: connection,
    libraryCatalog: catalog,
    libraryProposal: null,
    libraryConnectionRevision: 0,
    libraryOutputRevision: 0,
    libraryMutationQueue: Promise.resolve(),
    libraryIndexedPapers: [{ paperKey: `file:sha256:${"1".repeat(64)}`, title: "A Local Paper" }],
    buildPersonalLibraryProfileStores: () => ({ proposal: {}, suggestions: {} }),
    // Fails inside the run, after the status bar already belongs to it.
    buildFullTextKnowledgeBaseStore: () => { throw new Error("knowledge base unavailable"); },
  });
  return { plugin, progress };
}

describe("direction generation status bar", () => {
  it("stays out of the status bar when the caller reports progress itself", () => {
    // The review modal shows its own counter on the button. Writing the bar as
    // well put two differently-worded counters on screen for one run.
    const { plugin, progress } = makePlugin();
    const caller = vi.fn();
    plugin.proposalProgressReporter(caller)({ phase: "organization", completed: 0, total: 1 });
    expect(caller).toHaveBeenCalled();
    expect(progress.setTask).not.toHaveBeenCalled();
  });

  it("writes the status bar for the command-palette path, naming the slow phases", () => {
    const { plugin, progress } = makePlugin();
    const report = plugin.proposalProgressReporter();
    report({ phase: "reading", completed: 0, total: 0 });
    report({ phase: "grouping", completed: 0, total: 0 });
    report({ phase: "organization", completed: 0, total: 1 });
    expect(progress.setTask.mock.calls.map(([, detail]: [string, string]) => detail)).toEqual([
      "reading the index",
      "grouping papers",
      "organizing topics and directions",
    ]);
  });

  it("hands the status bar back when generation fails", async () => {
    const { plugin, progress } = makePlugin();
    await expect(plugin.generatePersonalLibraryDirections()).rejects.toThrow("knowledge base unavailable");
    expect(progress.setError).toHaveBeenCalledWith("Personal library direction generation failed");
    expect(progress.setComplete).not.toHaveBeenCalled();
  });
});
