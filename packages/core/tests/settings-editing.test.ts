import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import { normalizeSettingsEdits } from "../src/settings/editing";
const fresh = () => structuredClone(DEFAULT_SETTINGS);
describe("shared settings editing", () => {
  it("normalizes an isolated clone while preserving unrelated legacy domains", () => {
    const input = fresh(); input.llm.model = "  model-name  "; input.email.mode = "legacy" as "self";
    const result = normalizeSettingsEdits(input, ["llm.model"]);
    expect(result.llm.model).toBe("model-name"); expect(input.llm.model).toBe("  model-name  ");
    expect(result.email.mode).toBe("legacy"); expect(result).not.toBe(input); expect(result.arxiv.topics).not.toBe(input.arxiv.topics);
  });
  it("accepts incomplete model, topic and email drafts", () => {
    const input = fresh(); input.llm.apiKey = ""; input.llm.model = ""; input.llm.baseUrl = "";
    input.arxiv.topics = [{ id: "topic-1", name: "", tag: "", description: "line one\nline two", detail: false }];
    input.email.to = "user@"; input.email.fromEmail = "partial"; input.email.fromName = "  Sender Name  ";
    expect(normalizeSettingsEdits(input, ["llm", "arxiv.topics", "email"]).email.to).toBe("user@");
    expect(normalizeSettingsEdits(input, ["email"]).email.fromName).toBe("  Sender Name  ");
    expect(normalizeSettingsEdits(input, ["arxiv.topics"]).arxiv.topics[0]?.description).toContain("\n");
  });
  it("normalizes categories, output paths and a named detail policy", () => {
    const input = fresh(); input.arxiv.categories = [" cs.LG ", "cs.AI"]; input.output.dailyDir = " notes\\daily "; input.detailSelection.profile = "broad";
    const result = normalizeSettingsEdits(input, ["arxiv.categories", "output.dailyDir", "detailSelection.profile"]);
    expect(result.arxiv.categories).toEqual(["cs.LG", "cs.AI"]); expect(result.arxiv.category).toBe("cs.LG");
    expect(result.output.dailyDir).toBe("notes/daily"); expect(result.detailSelection.softLimit).toBe(5);
  });
  it.each([
    ["llm.baseUrl", "file:///tmp/model"], ["llm.baseUrl", "https://user:secret@example.com/v1"], ["llm.baseUrl", "https://example.com/v1?secret=x"],
    ["llm.model", "bad\nmodel"], ["llm.thinkingMode", "true"], ["email.mode", "smtp"], ["email.to", "a\nb"],
    ["embedding.mode", "invalid"], ["embedding.dimension", 0], ["embedding.dimension", 1.5],
    ["output.linkStyle", "invalid"], ["output.summaryLanguage", "fr"], ["output.dailyDir", "../escape"],
    ["schedule.tickIntervalMin", 1.5], ["schedule.tickIntervalMin", 1441], ["schedule.runAtLocal", "25:00"],
    ["arxiv.timezone", "invalid/timezone"], ["advanced.logLevel", "invalid"], ["detailSelection.profile", "invalid"],
  ])("rejects invalid %s", (key, value) => {
    const input = fresh(); const [domain, field] = key.split(".");
    (input as unknown as Record<string, Record<string, unknown>>)[domain!]![field!] = value;
    expect(() => normalizeSettingsEdits(input, [key])).toThrow();
  });
  it("rejects duplicate categories and duplicate or missing topic identities", () => {
    const input = fresh(); input.arxiv.categories = ["cs.AI", " cs.AI "];
    expect(() => normalizeSettingsEdits(input, ["arxiv.categories"])).toThrow(/categor/i);
    input.arxiv.topics = [{ id: "one", name: "", tag: "", description: "", detail: false }, { id: "one", name: "", tag: "", description: "", detail: false }];
    expect(() => normalizeSettingsEdits(input, ["arxiv.topics"])).toThrow(/topic/i);
    input.arxiv.topics = [{ id: "", name: "", tag: "", description: "", detail: false }];
    expect(() => normalizeSettingsEdits(input, ["arxiv.topics"])).toThrow(/topic/i);
  });
  it("enforces run-window ordering and readiness only when scheduling is enabled", () => {
    const input = fresh(); input.schedule.enabled = false; input.schedule.runAtLocal = "18:00"; input.schedule.runUntilLocal = "09:00";
    expect(() => normalizeSettingsEdits(input, ["schedule"])).toThrow();
    input.schedule.runAtLocal = "09:00"; input.schedule.runUntilLocal = "18:00";
    expect(() => normalizeSettingsEdits(input, ["schedule"])).not.toThrow();
    input.schedule.enabled = true;
    expect(() => normalizeSettingsEdits(input, ["schedule.enabled"])).toThrow(/API Key/i);
  });
  it("checks enabled sidecars against the shared loopback same-origin rule", () => {
    const input = fresh(); input.pdfParserSidecar.enabled = true; input.pdfParserSidecar.parseUrl = "https://external.example/parse";
    expect(() => normalizeSettingsEdits(input, ["pdfParserSidecar.parseUrl"])).toThrow();
    input.pdfParserSidecar.enabled = false;
    expect(() => normalizeSettingsEdits(input, ["pdfParserSidecar.enabled"])).not.toThrow();
  });
});
