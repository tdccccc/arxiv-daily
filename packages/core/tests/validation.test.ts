import { describe, it, expect } from "vitest";
import {
  validateLlmConfig,
  validateFilterConfig,
  validateScheduleConfig,
  validateVaultRelativeDirectory,
} from "../src/settings/validation";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { PluginSettings } from "../src/settings/types";

function makeSettings(overrides: Partial<PluginSettings> = {}): PluginSettings {
  return {
    ...DEFAULT_SETTINGS,
    ...overrides,
    llm: { ...DEFAULT_SETTINGS.llm, ...(overrides.llm ?? {}) },
    arxiv: { ...DEFAULT_SETTINGS.arxiv, ...(overrides.arxiv ?? {}) },
    output: { ...DEFAULT_SETTINGS.output, ...(overrides.output ?? {}) },
    email: { ...DEFAULT_SETTINGS.email, ...(overrides.email ?? {}) },
  };
}

describe("validateVaultRelativeDirectory", () => {
  it("trims and canonicalizes safe Unicode nested paths", () => {
    expect(validateVaultRelativeDirectory("  研究 资料\\每日论文  ")).toEqual({
      ok: true,
      value: "研究 资料/每日论文",
    });
  });

  it("normalizes canonically equivalent Unicode paths to NFC", () => {
    expect(validateVaultRelativeDirectory("Cafe\u0301/papers")).toEqual({
      ok: true,
      value: "Café/papers",
    });
  });

  it.each([
    "",
    "../papers",
    "daily/./papers",
    "/tmp/papers",
    "C:\\papers",
    "\\\\server\\share",
    "file:papers",
    ".obsidian/plugins",
    "papers?bad",
    "papers\u0000bad",
    "CON",
    "aux.txt/papers",
    "papers/COM1",
    "papers/lpt9.md",
  ])("rejects unsafe path %j", (value) => {
    expect(validateVaultRelativeDirectory(value).ok).toBe(false);
  });
});

describe("validateLlmConfig", () => {
  it("flags empty API key", () => {
    const r = validateLlmConfig(makeSettings({ llm: { ...DEFAULT_SETTINGS.llm, apiKey: "" } }));
    expect(r.ok).toBe(false);
    expect(r.reasons.join("; ")).toMatch(/api key/i);
  });

  it("flags whitespace-only API key", () => {
    const r = validateLlmConfig(makeSettings({ llm: { ...DEFAULT_SETTINGS.llm, apiKey: "   " } }));
    expect(r.ok).toBe(false);
  });

  it("flags empty base URL", () => {
    const r = validateLlmConfig(makeSettings({ llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x", baseUrl: "" } }));
    expect(r.ok).toBe(false);
    expect(r.reasons.join("; ")).toMatch(/base url/i);
  });

  it("flags empty model", () => {
    const r = validateLlmConfig(makeSettings({ llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x", model: "" } }));
    expect(r.ok).toBe(false);
    expect(r.reasons.join("; ")).toMatch(/model/i);
  });

  it("passes with all LLM fields populated", () => {
    const r = validateLlmConfig(
      makeSettings({ llm: { ...DEFAULT_SETTINGS.llm, apiKey: "sk-x", baseUrl: "https://x", model: "gpt-x" } }),
    );
    expect(r.ok).toBe(true);
    expect(r.reasons).toEqual([]);
  });

  it("does not require topics", () => {
    const r = validateLlmConfig(
      makeSettings({
        llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x" },
        arxiv: { ...DEFAULT_SETTINGS.arxiv, topics: [] },
      }),
    );
    expect(r.ok).toBe(true);
  });
});

describe("validateFilterConfig", () => {
  it("flags empty topics", () => {
    const r = validateFilterConfig(
      makeSettings({
        llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x" },
        arxiv: { ...DEFAULT_SETTINGS.arxiv, topics: [] },
      }),
    );
    expect(r.ok).toBe(false);
    expect(r.reasons.join("; ")).toMatch(/topic/i);
  });

  it("combines LLM and topics reasons", () => {
    const r = validateFilterConfig(
      makeSettings({
        llm: { ...DEFAULT_SETTINGS.llm, apiKey: "" },
        arxiv: { ...DEFAULT_SETTINGS.arxiv, topics: [] },
      }),
    );
    expect(r.ok).toBe(false);
    expect(r.reasons.length).toBeGreaterThanOrEqual(2);
  });

  it("passes with both LLM and topics populated", () => {
    const r = validateFilterConfig(
      makeSettings({
        llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x" },
        arxiv: {
          ...DEFAULT_SETTINGS.arxiv,
          topics: [{ id: "t", name: "T", tag: "t", description: "x", detail: false }],
        },
      }),
    );
    expect(r.ok).toBe(true);
  });

  it("flags empty and duplicate arXiv categories", () => {
    const r = validateFilterConfig(
      makeSettings({
        llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x" },
        arxiv: {
          ...DEFAULT_SETTINGS.arxiv,
          categories: ["astro-ph", " ", "astro-ph"],
          topics: [{ id: "t", name: "T", tag: "t", description: "x", detail: false }],
        },
      }),
    );
    expect(r.ok).toBe(false);
    expect(r.reasons.join("; ")).toMatch(/arXiv category is empty/);
    expect(r.reasons.join("; ")).toMatch(/Duplicate arXiv category: astro-ph/);
  });

  it("flags output directories that collide portably", () => {
    const r = validateFilterConfig(
      makeSettings({
        llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x" },
        output: {
          ...DEFAULT_SETTINGS.output,
          dailyDir: "Café/Notes",
          papersDir: "CAFE\u0301/notes",
        },
      }),
    );
    expect(r.reasons.join("; ")).toMatch(/must be different/i);
  });

  it("flags invalid link style", () => {
    const r = validateFilterConfig(
      makeSettings({
        llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x" },
        arxiv: {
          ...DEFAULT_SETTINGS.arxiv,
          topics: [{ id: "t", name: "T", tag: "t", description: "x", detail: false }],
        },
        output: {
          ...DEFAULT_SETTINGS.output,
          linkStyle: "absolute" as any,
        },
      }),
    );
    expect(r.ok).toBe(false);
    expect(r.reasons.join("; ")).toMatch(/Invalid link style: absolute/);
  });

  it("flags invalid summary language", () => {
    const r = validateFilterConfig(
      makeSettings({
        llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x" },
        arxiv: {
          ...DEFAULT_SETTINGS.arxiv,
          topics: [{ id: "t", name: "T", tag: "t", description: "x", detail: false }],
        },
        output: {
          ...DEFAULT_SETTINGS.output,
          summaryLanguage: "fr" as any,
        },
      }),
    );
    expect(r.ok).toBe(false);
    expect(r.reasons.join("; ")).toMatch(/Invalid summary language: fr/);
  });

  it("flags empty topic fields", () => {
    const r = validateFilterConfig(
      makeSettings({
        llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x" },
        arxiv: {
          ...DEFAULT_SETTINGS.arxiv,
          topics: [{ id: "t", name: " ", tag: "", description: " ", detail: false }],
        },
      }),
    );
    expect(r.ok).toBe(false);
    expect(r.reasons.join("; ")).toMatch(/name is empty/i);
    expect(r.reasons.join("; ")).toMatch(/tag is empty/i);
    expect(r.reasons.join("; ")).toMatch(/description is empty/i);
  });

  it("flags duplicate topic tags", () => {
    const r = validateFilterConfig(
      makeSettings({
        llm: { ...DEFAULT_SETTINGS.llm, apiKey: "x" },
        arxiv: {
          ...DEFAULT_SETTINGS.arxiv,
          topics: [
            { id: "t1", name: "One", tag: "same", description: "x", detail: false },
            { id: "t2", name: "Two", tag: "same", description: "y", detail: false },
          ],
        },
      }),
    );
    expect(r.ok).toBe(false);
    expect(r.reasons.join("; ")).toMatch(/duplicate topic tag: same/i);
  });
});

describe("advanced default char limits", () => {
  it("uses the relaxed extraction budgets", () => {
    expect(DEFAULT_SETTINGS.advanced.paperCharLimit).toBe(100_000);
    expect(DEFAULT_SETTINGS.advanced.sectionCharLimit).toBe(16_000);
  });
});

describe("schedule defaults", () => {
  it("uses a default run window from 09:00 to 18:00", () => {
    expect(DEFAULT_SETTINGS.schedule.runAtLocal).toBe("09:00");
    expect(DEFAULT_SETTINGS.schedule.runUntilLocal).toBe("18:00");
  });

  it("rejects overnight scheduler windows", () => {
    const r = validateScheduleConfig(
      makeSettings({
        schedule: {
          ...DEFAULT_SETTINGS.schedule,
          runAtLocal: "22:00",
          runUntilLocal: "06:00",
        },
      }),
    );

    expect(r.ok).toBe(false);
    expect(r.reasons.join("; ")).toMatch(/run window/i);
  });
});
