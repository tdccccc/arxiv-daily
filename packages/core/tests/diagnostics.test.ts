import { describe, expect, it } from "vitest";
import { buildDiagnosticsReport } from "../src/services/diagnostics";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { PluginSettings, RunState } from "../src/settings/types";

function makeSettings(overrides: Partial<PluginSettings> = {}): PluginSettings {
  return {
    ...DEFAULT_SETTINGS,
    ...overrides,
    llm: { ...DEFAULT_SETTINGS.llm, ...(overrides.llm ?? {}) },
    arxiv: { ...DEFAULT_SETTINGS.arxiv, ...(overrides.arxiv ?? {}) },
    output: { ...DEFAULT_SETTINGS.output, ...(overrides.output ?? {}) },
    schedule: { ...DEFAULT_SETTINGS.schedule, ...(overrides.schedule ?? {}) },
    advanced: { ...DEFAULT_SETTINGS.advanced, ...(overrides.advanced ?? {}) },
  };
}

describe("buildDiagnosticsReport", () => {
  it("reports local config and state without exposing the API key", () => {
    const settings = makeSettings({
      llm: {
        ...DEFAULT_SETTINGS.llm,
        apiKey: "sk-secret-value",
        baseUrl: "https://llm.example/v1",
        model: "test-model",
      },
      arxiv: {
        ...DEFAULT_SETTINGS.arxiv,
        timezone: "Asia/Shanghai",
        topics: [
          {
            id: "topic-1",
            name: "Photo-z",
            tag: "photo-z",
            description: "private matching criteria",
            detail: true,
          },
        ],
      },
      schedule: { ...DEFAULT_SETTINGS.schedule, enabled: true },
    });
    const runState: RunState = {
      "2026-06-11": {
        status: "failed_transient",
        attempts: 2,
        lastAttempt: Date.UTC(2026, 5, 11, 0, 30),
        error: "date 2026-06-11 not in /recent",
      },
      "2026-06-10": {
        status: "completed",
        attempts: 1,
        lastAttempt: Date.UTC(2026, 5, 10, 1, 0),
        papersWritten: 3,
      },
    };

    const report = buildDiagnosticsReport({
      settings,
      runState,
      version: "0.1.3",
      now: new Date("2026-06-11T01:00:00.000Z"),
    });

    expect(report).toContain("pluginVersion: 0.1.3");
    expect(report).toContain("baseUrl: https://llm.example/v1");
    expect(report).toContain("model: test-model");
    expect(report).toContain("apiKeySet: yes");
    expect(report).toContain("linkStyle: wikilink");
    expect(report).not.toContain("sk-secret-value");
    expect(report).not.toContain("private matching criteria");
    expect(report).toContain("localDate: 2026-06-11");
    expect(report).toContain("localWeekday: Thursday");
    expect(report).toContain("2026-06-11: state=failed_transient, weekend=no, attempts=2");
    expect(report).toContain("2026-06-10: state=completed, weekend=no, attempts=1");
    expect(report).toContain("failedDates:\n  - 2026-06-11");
  });

  it("redacts secrets embedded in URLs and diagnostic errors", () => {
    const secret = "sk-complete-secret-value";
    const report = buildDiagnosticsReport({
      settings: makeSettings({
        llm: { ...DEFAULT_SETTINGS.llm, apiKey: secret, baseUrl: `https://example.test/v1?api_key=${secret}` },
      }),
      runState: {
        "2026-07-16": {
          status: "failed_transient", attempts: 1, lastAttempt: 1,
          error: `provider echoed Bearer ${secret}`,
        },
      },
      paperIndex: { path: `index?token=${secret}`, exists: false, error: secret },
    });
    expect(report).not.toContain(secret);
    expect(report).not.toContain("sk-complete");
  });

  it("includes validation reasons for incomplete topic settings", () => {
    const settings = makeSettings({
      llm: { ...DEFAULT_SETTINGS.llm, apiKey: "" },
      arxiv: {
        ...DEFAULT_SETTINGS.arxiv,
        topics: [
          { id: "empty", name: "", tag: "", description: "", detail: false },
        ],
      },
    });

    const report = buildDiagnosticsReport({
      settings,
      runState: {},
      now: new Date("2026-06-11T01:00:00.000Z"),
    });

    expect(report).toContain("llm: invalid");
    expect(report).toContain("filter: invalid");
    expect(report).toContain("LLM API Key is empty");
    expect(report).toContain("Topic 1 name is empty");
    expect(report).toContain("Topic 1 tag is empty");
    expect(report).toContain("Topic 1 description is empty");
    expect(report).toContain("apiKeySet: no");
    expect(report).toContain('name="(empty)", tag="(empty)"');
  });

  it("includes paper index diagnostics when provided", () => {
    const report = buildDiagnosticsReport({
      settings: makeSettings(),
      runState: {},
      now: new Date("2026-06-11T01:00:00.000Z"),
      paperIndex: {
        path: "arxiv-daily/.index/papers.json",
        exists: true,
        schemaVersion: 2,
        total: 3,
        statusCounts: { inbox: 2, saved: 1 },
        unsupportedSchemaVersion: "9",
        invalidStatuses: ["2606.22222: stale"],
        invalidPriorities: ["2606.33333: urgent"],
        invalidSeenDates: ["2606.44444: 20260611"],
        missingPaperPaths: ["2606.12345: arxiv-daily/papers/2606.12345.md"],
        noteArxivIdMismatches: [
          "2606.55555: arxiv-daily/papers/2606.55555.md has arxiv_id 2606.99999",
        ],
      },
    });

    expect(report).toContain("paperIndex:");
    expect(report).toContain("path: arxiv-daily/.index/papers.json");
    expect(report).toContain("schemaVersion: 2");
    expect(report).toContain("unsupportedSchemaVersion: 9");
    expect(report).toContain("total: 3");
    expect(report).toContain("statusCounts: inbox=2, saved=1");
    expect(report).toContain("2606.22222: stale");
    expect(report).toContain("2606.33333: urgent");
    expect(report).toContain("2606.44444: 20260611");
    expect(report).toContain("2606.12345: arxiv-daily/papers/2606.12345.md");
    expect(report).toContain("has arxiv_id 2606.99999");
  });
});
