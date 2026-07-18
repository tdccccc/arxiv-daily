import { describe, expect, it } from "vitest";
import { getSetupStatus, shouldRenderSetupGuide } from "../src/onboarding";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import type { PluginSettings } from "@arxiv-daily/core";

type SettingsOverrides = Omit<
  Partial<PluginSettings>,
  "llm" | "arxiv" | "output" | "schedule" | "advanced"
> & {
  llm?: Partial<PluginSettings["llm"]>;
  arxiv?: Partial<PluginSettings["arxiv"]>;
  output?: Partial<PluginSettings["output"]>;
  schedule?: Partial<PluginSettings["schedule"]>;
  advanced?: Partial<PluginSettings["advanced"]>;
};

function makeSettings(overrides: SettingsOverrides = {}): PluginSettings {
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

describe("getSetupStatus", () => {
  it("reports missing LLM and topic setup from defaults", () => {
    const status = getSetupStatus(makeSettings());

    expect(status.llmReady).toBe(false);
    expect(status.categoriesReady).toBe(true);
    expect(status.topicsReady).toBe(false);
    expect(status.readyToRun).toBe(false);
    expect(status.reasons.join("; ")).toMatch(/api key/i);
    expect(status.reasons.join("; ")).toMatch(/research topics/i);
  });

  it("passes when the minimal run configuration is complete", () => {
    const status = getSetupStatus(
      makeSettings({
        llm: { apiKey: "sk-test" },
        arxiv: {
          topics: [
            {
              id: "topic",
              name: "Compact objects",
              tag: "compact-objects",
              description: "Neutron stars and black holes",
              detail: false,
            },
          ],
        },
      }),
    );

    expect(status.llmReady).toBe(true);
    expect(status.categoriesReady).toBe(true);
    expect(status.topicsReady).toBe(true);
    expect(status.readyToRun).toBe(true);
    expect(status.reasons).toEqual([]);
  });

  it("keeps the guide until a first report completes", () => {
    const settings = makeSettings({
      llm: { apiKey: "sk-test" },
      arxiv: {
        topics: [
          {
            id: "topic",
            name: "Compact objects",
            tag: "compact-objects",
            description: "Neutron stars and black holes",
            detail: false,
          },
        ],
      },
    });
    const beforeFirstReport = getSetupStatus(settings);
    const afterFirstReport = getSetupStatus(settings, {
      "2026-07-15": { status: "completed", lastAttempt: 1, attempts: 1 },
      "2026-07-16": { status: "failed_transient", lastAttempt: 2, attempts: 1 },
      "2026-07-14": { status: "completed", lastAttempt: 3, attempts: 1 },
    });

    expect(beforeFirstReport.firstReportComplete).toBe(false);
    expect(shouldRenderSetupGuide(beforeFirstReport)).toBe(true);
    expect(afterFirstReport.firstReportComplete).toBe(true);
    expect(afterFirstReport.latestCompletedReportDate).toBe("2026-07-15");
    expect(shouldRenderSetupGuide(afterFirstReport)).toBe(false);
  });

  it("returns the guide when configuration becomes invalid after a report", () => {
    const status = getSetupStatus(makeSettings(), {
      "2026-07-15": { status: "completed", lastAttempt: 1, attempts: 1 },
    });

    expect(status.firstReportComplete).toBe(true);
    expect(status.readyToRun).toBe(false);
    expect(shouldRenderSetupGuide(status)).toBe(true);
  });

  it("keeps incomplete topics actionable", () => {
    const status = getSetupStatus(
      makeSettings({
        llm: { apiKey: "sk-test" },
        arxiv: {
          topics: [
            {
              id: "topic",
              name: "Compact objects",
              tag: "",
              description: "",
              detail: false,
            },
          ],
        },
      }),
    );

    expect(status.llmReady).toBe(true);
    expect(status.topicsReady).toBe(false);
    expect(status.readyToRun).toBe(false);
    expect(status.reasons.join("; ")).toMatch(/tag is empty/i);
    expect(status.reasons.join("; ")).toMatch(/description is empty/i);
  });
});
