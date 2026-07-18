import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  isValidLocalTime,
  llmHttpWarning,
  modelFetchNoticeMessage,
  runWindowTimeOptions,
  validateOutputDirectoryDraft,
} from "../src/settings/tab";

const settingsTabSource = readFileSync(
  resolve(process.cwd(), "src/settings/tab.ts"),
  "utf-8",
);

describe("modelFetchNoticeMessage", () => {
  it("reports a successful model fetch in English", () => {
    expect(modelFetchNoticeMessage({ kind: "success", count: 3 })).toBe(
      "API connection successful. Found 3 models.",
    );
  });

  it("reports an empty model list in English", () => {
    expect(modelFetchNoticeMessage({ kind: "empty" })).toBe(
      "API connection successful, but no available models were found.",
    );
  });

  it("reports a failed model fetch in English", () => {
    expect(
      modelFetchNoticeMessage({ kind: "error", message: "Unauthorized" }),
    ).toBe("API connection failed: Unauthorized");
  });
});

describe("llmHttpWarning", () => {
  it("warns without blocking for non-loopback HTTP endpoints", () => {
    expect(llmHttpWarning("http://59.64.32.247:5001/v1")).toEqual({
      kind: "plaintext",
      message: "Your LLM endpoint uses HTTP; API keys will be sent in plaintext.",
    });
  });

  it("uses a softer warning for local HTTP endpoints", () => {
    expect(llmHttpWarning("http://localhost:5001/v1")).toEqual({
      kind: "local",
      message: "Using a local HTTP LLM endpoint; ensure this is intentional.",
    });
    expect(llmHttpWarning("http://127.12.0.1:5001/v1")?.kind).toBe("local");
    expect(llmHttpWarning("http://[::1]:5001/v1")?.kind).toBe("local");
  });

  it("does not warn for HTTPS or invalid partial input", () => {
    expect(llmHttpWarning("https://api.deepseek.com/v1")).toBeNull();
    expect(llmHttpWarning("59.64.32.247:5001/v1")).toBeNull();
  });
});

describe("settings tab regressions", () => {
  it("never renders the saved API key and requires explicit replace/save/cancel/clear actions", () => {
    const apiKeyBody = settingsTabSource.match(
      /private renderApiKeySetting\([\s\S]*?\n  private renderSetupGuide/,
    )?.[0];
    expect(apiKeyBody).toBeDefined();
    expect(apiKeyBody).not.toContain("input.value = this.plugin.settings.llm.apiKey");
    expect(apiKeyBody).not.toContain("setValue(s.llm.apiKey)");
    expect(apiKeyBody).toContain("API_KEY_CONFIGURED_SENTINEL");
    expect(apiKeyBody).toContain('text: configured ? "Replace" : "Save"');
    expect(apiKeyBody).toContain('text: "Cancel"');
    expect(apiKeyBody).toContain('text: "Clear"');
    expect(apiKeyBody).toContain("this.plugin.logger.setSensitiveValues([next])");
    expect(apiKeyBody).toContain("this.plugin.logger.setSensitiveValues([])");
  });

  it("does not register a second change listener when models are fetched", () => {
    const showModelDropdownBody = settingsTabSource.match(
      /private showModelDropdown\([\s\S]*?\n  private textareaSetting/,
    )?.[0];

    expect(showModelDropdownBody).toBeDefined();
    expect(showModelDropdownBody).not.toContain('select.addEventListener("change"');
  });

  it("warns that quick-start templates replace categories", () => {
    expect(settingsTabSource).toContain("and arXiv categories");
  });

  it("uses accessible topic disclosure controls and associated field labels", () => {
    expect(settingsTabSource).toContain('card.createEl("button"');
    expect(settingsTabSource).toContain('"aria-expanded": String(isExpanded)');
    expect(settingsTabSource).toContain('"aria-controls": formId');
    expect(settingsTabSource).toContain("form.hidden = !isExpanded");
    expect(settingsTabSource).toContain('attr: { for: nameId }');
    expect(settingsTabSource).toContain('attr: { for: tagId }');
    expect(settingsTabSource).toContain('attr: { for: descId }');
    expect(settingsTabSource).toContain('"aria-describedby": nameHintId');
  });

  it("confirms topic deletion by name before persistence", () => {
    expect(settingsTabSource).toContain('Delete the research topic "${topicName}"?');
    expect(settingsTabSource).toContain("if (!confirmed) return");
    expect(settingsTabSource.indexOf("if (!confirmed) return")).toBeLessThan(
      settingsTabSource.indexOf("topics.splice(index, 1)"),
    );
  });

  it("renders automatic detail policy near topics without changing manual summarize", () => {
    const headingIndex = settingsTabSource.indexOf('"Research Topics"');
    const policyIndex = settingsTabSource.indexOf('"Automatic deep-dive selection"');
    const timezoneIndex = settingsTabSource.indexOf('.setName("Timezone")');
    expect(policyIndex).toBeGreaterThan(headingIndex);
    expect(policyIndex).toBeLessThan(timezoneIndex);
    expect(settingsTabSource).toContain("Topic Detail report checkboxes enable eligibility");
    expect(settingsTabSource).toContain("automatic selection only; manual summarize is unaffected");
    expect(settingsTabSource).toContain('.addOption("conservative", "Conservative")');
    expect(settingsTabSource).toContain('.addOption("balanced", "Balanced")');
    expect(settingsTabSource).toContain('.addOption("broad", "Broad")');
    expect(settingsTabSource).toContain('profile: "custom"');
    expect(settingsTabSource).toContain("sanitizeDetailSelection");
    expect(settingsTabSource).toContain("await this.plugin.saveSettings()");
  });

  it("uses numeric limits for all editable automatic detail controls", () => {
    expect(settingsTabSource).toContain('t.inputEl.type = "number"');
    expect(settingsTabSource).toContain('t.inputEl.min = "0"');
    expect(settingsTabSource).toContain('"normalThreshold"');
    expect(settingsTabSource).toContain('"exceptionalThreshold"');
    expect(settingsTabSource).toContain('"softLimit"');
  });

  it("uses explicit Start and End labels with non-cyclic select controls", () => {
    expect(settingsTabSource).toContain('"Start"');
    expect(settingsTabSource).toContain('"End"');
    expect(settingsTabSource).toContain('field.createEl("select"');
    expect(settingsTabSource).not.toContain('inputEl.type = "time"');
  });

  it("does not normalize or persist categories merely while displaying them", () => {
    expect(settingsTabSource).toContain("const categories = arxivCategories(s.arxiv);");
    expect(settingsTabSource).toContain(
      "this.plugin.settings.arxiv.categories = normalized;",
    );
    expect(settingsTabSource).toMatch(
      /const apply = async \(\) => \{[\s\S]*?s\.arxiv\.categories = \[tpl\.category\];/,
    );
  });
});

describe("output path drafts", () => {
  it("normalizes safe vault-relative directories", () => {
    expect(validateOutputDirectoryDraft(" arxiv\\papers/details ")).toEqual({
      ok: true,
      value: "arxiv/papers/details",
    });
  });

  it("rejects a sibling directory collision portably", () => {
    expect(validateOutputDirectoryDraft("cafe\u0301/NOTES", "Café/notes")).toEqual({
      ok: false,
      reason: "Daily and papers directories must be different",
    });
  });

  it("rejects empty, absolute, traversal, and configuration paths", () => {
    expect(validateOutputDirectoryDraft("").ok).toBe(false);
    expect(validateOutputDirectoryDraft("/tmp/papers").ok).toBe(false);
    expect(validateOutputDirectoryDraft("C:/papers").ok).toBe(false);
    expect(validateOutputDirectoryDraft("arxiv/../notes").ok).toBe(false);
    expect(validateOutputDirectoryDraft(".obsidian/plugins").ok).toBe(false);
  });

  it("reloads before persistence and restores the prior value on failure", () => {
    const body = settingsTabSource.match(
      /private async applyOutputDirectoryDraft[\s\S]*?\n  display\(\): void/,
    )?.[0];
    expect(body).toBeDefined();
    expect(body!.indexOf("reloadStateStoreForOutputPaths()")).toBeLessThan(
      body!.indexOf("saveSettings()"),
    );
    expect(body).toContain("this.plugin.settings.output[key] = previous");
    expect(body).toContain("await this.plugin.reloadStateStoreForOutputPaths()");
  });
});

describe("run window time options", () => {
  it("renders standard 24-hour quarter-hour values without 24:00", () => {
    const options = runWindowTimeOptions("09:00");
    expect(options).toHaveLength(96);
    expect(options[0]).toMatchObject({ value: "00:00", label: "00:00" });
    expect(options.at(-1)).toMatchObject({ value: "23:45", label: "23:45" });
    expect(options.some((option) => option.value === "24:00")).toBe(false);
  });

  it("preserves arbitrary valid minutes as a selectable value", () => {
    const options = runWindowTimeOptions("09:07");
    expect(options).toContainEqual({ value: "09:07", label: "09:07", valid: true });
    expect(isValidLocalTime("09:07")).toBe(true);
  });

  it("displays invalid and legacy values without treating them as persistable", () => {
    expect(runWindowTimeOptions("24:00")).toContainEqual({
      value: "24:00",
      label: "24:00 — invalid",
      valid: false,
    });
    expect(isValidLocalTime("24:00")).toBe(false);
    expect(isValidLocalTime("9:00 AM")).toBe(false);
  });
});
