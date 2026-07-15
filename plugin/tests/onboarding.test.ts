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

  it("renders the setup guide only while setup is incomplete", () => {
    const incomplete = getSetupStatus(makeSettings());
    const complete = getSetupStatus(
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

    expect(shouldRenderSetupGuide(incomplete)).toBe(true);
    expect(shouldRenderSetupGuide(complete)).toBe(false);
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
