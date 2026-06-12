import { describe, it, expect } from "vitest";
import { validateLlmConfig, validateFilterConfig } from "../src/settings/validation";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { PluginSettings } from "../src/settings/types";

function makeSettings(overrides: Partial<PluginSettings> = {}): PluginSettings {
  return {
    ...DEFAULT_SETTINGS,
    ...overrides,
    llm: { ...DEFAULT_SETTINGS.llm, ...(overrides.llm ?? {}) },
    arxiv: { ...DEFAULT_SETTINGS.arxiv, ...(overrides.arxiv ?? {}) },
  };
}

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
