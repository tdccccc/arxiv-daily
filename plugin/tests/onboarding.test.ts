import { describe, expect, it } from "vitest";
import {
  getSetupStatus,
  markSetupGuideCompleteIfDone,
  shouldRenderSetupGuide,
} from "../src/onboarding";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";
import type { PluginSettings } from "@arxiv-daily/core";

type SettingsOverrides = Omit<
  Partial<PluginSettings>,
  "llm" | "arxiv" | "output" | "schedule" | "advanced" | "onboarding"
> & {
  llm?: Partial<PluginSettings["llm"]>;
  arxiv?: Partial<PluginSettings["arxiv"]>;
  output?: Partial<PluginSettings["output"]>;
  schedule?: Partial<PluginSettings["schedule"]>;
  advanced?: Partial<PluginSettings["advanced"]>;
  onboarding?: Partial<PluginSettings["onboarding"]>;
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
    // Cloned (not just spread) so tests that flip the marker in place
    // cannot leak the mutation back into DEFAULT_SETTINGS.
    onboarding: { ...DEFAULT_SETTINGS.onboarding, ...(overrides.onboarding ?? {}) },
  };
}

describe("getSetupStatus", () => {
  it("judges topic readiness by authored directions rather than the rollback description", () => {
    const settings = makeSettings({ arxiv: { topics: [{
      id: "topic", name: "Agents", tag: "agents", detail: false,
      description: "Old shadow", directions: [],
    }] } });
    expect(getSetupStatus(settings).topicsReady).toBe(false);
    settings.arxiv.topics[0]!.description = "";
    settings.arxiv.topics[0]!.directions = [{ id: "d", text: "Reliable agents", origin: "manual" }];
    expect(getSetupStatus(settings).topicsReady).toBe(true);
  });

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
              directions: [
                { id: "d1", text: "Neutron stars and black holes", origin: "migrated" },
              ],
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
            directions: [
              { id: "d1", text: "Neutron stars and black holes", origin: "migrated" },
            ],
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
    expect(shouldRenderSetupGuide(beforeFirstReport, false)).toBe(true);
    expect(afterFirstReport.firstReportComplete).toBe(true);
    expect(afterFirstReport.latestCompletedReportDate).toBe("2026-07-15");
    expect(shouldRenderSetupGuide(afterFirstReport, false)).toBe(true);
  });

  it("keeps the guide until daily runs are turned on", () => {
    const topics = [{ directions: [{ id: "fixture-direction", text: "Neutron stars and black holes", origin: "manual" as const }],
      id: "topic",
      name: "Compact objects",
      tag: "compact-objects",
      description: "Neutron stars and black holes",
      detail: false,
    }];
    const runState = {
      "2026-07-15": { status: "completed" as const, lastAttempt: 1, attempts: 1 },
    };
    const paused = getSetupStatus(
      makeSettings({ llm: { apiKey: "sk-test" }, arxiv: { topics } }),
      runState,
    );
    const running = getSetupStatus(
      makeSettings({
        llm: { apiKey: "sk-test" },
        arxiv: { topics },
        schedule: { enabled: true },
      }),
      runState,
    );

    expect(paused.scheduleEnabled).toBe(false);
    expect(shouldRenderSetupGuide(paused, false)).toBe(true);
    expect(running.scheduleEnabled).toBe(true);
    expect(shouldRenderSetupGuide(running, false)).toBe(false);
  });

  it("keeps the guide hidden when configuration becomes invalid after completion", () => {
    // Once the guide has completed (marker persisted), a later report that
    // reveals a broken configuration must not resurrect the full guide.
    const status = getSetupStatus(makeSettings(), {
      "2026-07-15": { status: "completed", lastAttempt: 1, attempts: 1 },
    });

    expect(status.firstReportComplete).toBe(true);
    expect(status.readyToRun).toBe(false);
    expect(shouldRenderSetupGuide(status, true)).toBe(false);
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
              directions: [],
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
    expect(status.reasons.join("; ")).toMatch(/has no directions/i);
  });

  it("does not call topics ready when two topics share a tag", () => {
    const topic = (id: string, name: string) => ({
      id,
      name,
      tag: "shared",
      description: `${name} papers`,
      directions: [{ id: "direction", text: `${name} papers`, origin: "manual" as const }],
      detail: false,
    });
    const status = getSetupStatus(
      makeSettings({
        llm: { apiKey: "sk-test" },
        arxiv: { topics: [topic("a", "First"), topic("b", "Second")] },
      }),
    );

    expect(status.topicsReady).toBe(false);
    expect(status.reasons.join("; ")).toMatch(/duplicate topic tag/i);
  });
});

describe("setup guide completion marker", () => {
  const topics = [{ directions: [{ id: "fixture-direction", text: "Neutron stars and black holes", origin: "manual" as const }],
    id: "topic",
    name: "Compact objects",
    tag: "compact-objects",
    description: "Neutron stars and black holes",
    detail: false,
  }];
  const runState = {
    "2026-07-15": { status: "completed" as const, lastAttempt: 1, attempts: 1 },
  };
  function completeSettings(): PluginSettings {
    return makeSettings({
      llm: { apiKey: "sk-test" },
      arxiv: { topics },
      schedule: { enabled: true },
    });
  }

  it("stays hidden once complete even after the schedule is turned off", () => {
    const status = getSetupStatus(completeSettings(), runState);
    expect(status.readyToRun && status.firstReportComplete && status.scheduleEnabled).toBe(true);

    const afterScheduleOff = getSetupStatus(
      makeSettings({ llm: { apiKey: "sk-test" }, arxiv: { topics } }),
      runState,
    );

    expect(afterScheduleOff.scheduleEnabled).toBe(false);
    expect(shouldRenderSetupGuide(afterScheduleOff, true)).toBe(false);
  });

  it("sets the marker the first time every milestone is true at once", () => {
    const settings = completeSettings();
    const status = getSetupStatus(settings, runState);

    expect(settings.onboarding.guideCompleted).toBe(false);
    expect(markSetupGuideCompleteIfDone(settings, status)).toBe(true);
    expect(settings.onboarding.guideCompleted).toBe(true);
  });

  it("does not re-flip or report a change once the marker is already set", () => {
    const settings = completeSettings();
    settings.onboarding.guideCompleted = true;
    const status = getSetupStatus(settings, runState);

    expect(markSetupGuideCompleteIfDone(settings, status)).toBe(false);
    expect(settings.onboarding.guideCompleted).toBe(true);
  });

  it("leaves the marker unset while any milestone is still incomplete", () => {
    const settings = makeSettings({ llm: { apiKey: "sk-test" }, arxiv: { topics } });
    const status = getSetupStatus(settings, runState);

    expect(status.scheduleEnabled).toBe(false);
    expect(markSetupGuideCompleteIfDone(settings, status)).toBe(false);
    expect(settings.onboarding.guideCompleted).toBe(false);
  });
});

it("does not count a no-update day as first report completion", () => {
  expect(getSetupStatus(DEFAULT_SETTINGS, {
    "2026-10-03": { status: "completed", attempts: 1, lastAttempt: 1, papersWritten: 0, outcome: "no_updates" },
  }).firstReportComplete).toBe(false);
});
