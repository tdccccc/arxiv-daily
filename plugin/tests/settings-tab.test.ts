import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  isValidLocalTime,
  llmHttpWarning,
  modelFetchNoticeMessage,
  persistApiKeyChange,
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

describe("API key persistence", () => {
  it("keeps a saved replacement and updates redaction", async () => {
    const settings = { llm: { apiKey: "old-secret" } };
    const logger = { setSensitiveValues: vi.fn() };
    const saveSettings = vi.fn().mockResolvedValue(undefined);

    await persistApiKeyChange(settings, logger, saveSettings, "new-secret");

    expect(settings.llm.apiKey).toBe("new-secret");
    expect(logger.setSensitiveValues).toHaveBeenCalledTimes(1);
    expect(logger.setSensitiveValues).toHaveBeenCalledWith(["new-secret"]);
  });

  it("restores the previous replacement and redaction when saving fails", async () => {
    const settings = { llm: { apiKey: "old-secret" } };
    const logger = { setSensitiveValues: vi.fn() };
    const failure = new Error("disk full");

    await expect(
      persistApiKeyChange(settings, logger, async () => Promise.reject(failure), "new-secret"),
    ).rejects.toBe(failure);

    expect(settings.llm.apiKey).toBe("old-secret");
    expect(logger.setSensitiveValues.mock.calls).toEqual([
      [["new-secret"]],
      [["old-secret"]],
    ]);
  });

  it("restores a cleared key and redaction when saving fails", async () => {
    const settings = { llm: { apiKey: "old-secret" } };
    const logger = { setSensitiveValues: vi.fn() };

    await expect(
      persistApiKeyChange(settings, logger, async () => Promise.reject(new Error("read only")), ""),
    ).rejects.toThrow("read only");

    expect(settings.llm.apiKey).toBe("old-secret");
    expect(logger.setSensitiveValues.mock.calls).toEqual([
      [[]],
      [["old-secret"]],
    ]);
  });
});

describe("settings tab regressions", () => {
  it("uses Obsidian 1.4-compatible title and ARIA help text", () => {
    const attachHelpBody = settingsTabSource.match(
      /private attachHelp[\s\S]*?\n  private reportActionError/,
    )?.[0];
    expect(attachHelpBody).toContain('title: text, "aria-label": text');
    expect(settingsTabSource).not.toContain("setTooltip");
  });

  it("uses scoped element creation in production settings code", () => {
    expect(settingsTabSource).not.toContain("document.createElement");
  });

  it("reports focused fire-and-forget failures instead of swallowing them", () => {
    expect(settingsTabSource).toContain("this.plugin.logger.error(`settings: ${action} failed`");
    expect(settingsTabSource).toContain("new Notice(`arXiv Daily: ${action} failed:");
    expect(settingsTabSource).not.toContain(".catch(() => {})");
    expect(settingsTabSource).toContain('this.runAction("update daily path"');
    expect(settingsTabSource).toContain('this.runAction("generate first report"');
    expect(settingsTabSource).toContain('this.runAction("open dashboard"');
    expect(settingsTabSource).toContain('this.runAction("save selected model"');
    expect(settingsTabSource).toContain('this.reportActionError("save run window"');
  });

  it("renders an accessible four-step first-report guide without duplicate inputs", () => {
    const guideBody = settingsTabSource.match(
      /private createSetupGuide\(\)[\s\S]*?\n  private renderSetupItem/,
    )?.[0];
    expect(guideBody).toBeDefined();
    expect(guideBody).toContain('createEl("ol"');
    expect(settingsTabSource).toContain('parent.createEl("li"');
    expect(guideBody).toContain('text: `${completedCount} of 4 complete`');
    expect(guideBody).toContain('"Connect AI"');
    expect(guideBody).toContain('"Choose paper sources"');
    expect(guideBody).toContain('"Describe your research interests"');
    expect(guideBody).toContain('"Generate your first report"');
    expect(settingsTabSource).toContain('text: done ? "Complete" : "Next"');
    expect(settingsTabSource).not.toContain('text: done ? "Done"');
    expect(guideBody).not.toContain("new Setting(");
    expect(guideBody).not.toContain("PROVIDER_PRESETS");
  });

  it("uses run-state completion, awaits the first report, and renders compact completion", () => {
    const guideBody = settingsTabSource.match(
      /private createSetupGuide\(\)[\s\S]*?\n  private renderSetupItem/,
    )?.[0];
    const firstReportBody = settingsTabSource.match(
      /private async generateFirstReport\(\)[\s\S]*?\n  private renderTopicCard/,
    )?.[0];
    expect(guideBody).toContain("this.plugin.stateStore.snapshot()");
    expect(guideBody).toContain("status.firstReportComplete");
    expect(guideBody).toContain("status.readyToRun ? \"Generate first report\" : undefined");
    expect(guideBody).toContain('this.runAction("generate first report"');
    expect(firstReportBody).toContain("await this.plugin.scheduler.runForDateNow(date)");
    expect(firstReportBody).toContain("this.refreshSetupGuide()");
    expect(settingsTabSource).not.toContain('this.executeCommand("run-now")');
    expect(guideBody).toContain('guide.addClass("arxiv-daily-setup--complete")');
    expect(guideBody).toContain('text: "Setup complete"');
    expect(guideBody).toContain("status.latestCompletedReportDate");
    expect(settingsTabSource).toContain('text: "Open dashboard"');
  });

  it("keeps validation reasons in guide details and removes the duplicate banner", () => {
    expect(settingsTabSource).toContain('details.createEl("summary", { text: "Configuration details" })');
    expect(settingsTabSource).toContain("status.schedulerReasons");
    expect(settingsTabSource).toContain("for (const reason of reasons)");
    expect(settingsTabSource).not.toContain('text: "Configuration incomplete"');
    expect(settingsTabSource).not.toContain("arxiv-daily-settings__invalid-banner");
  });

  it("focuses setup targets and respects reduced motion through ownerDocument", () => {
    const scrollBody = settingsTabSource.match(
      /private scrollToSection\([\s\S]*?\n  private async generateFirstReport/,
    )?.[0];
    expect(scrollBody).toContain("targetEl.ownerDocument.defaultView");
    expect(scrollBody).toContain('matchMedia?.("(prefers-reduced-motion: reduce)")');
    expect(scrollBody).toContain('targetEl.setAttribute("tabindex", "-1")');
    expect(scrollBody).toContain('behavior: reduceMotion ? "auto" : "smooth"');
    expect(scrollBody).toContain("targetEl.focus({ preventScroll: true })");
  });

  it("uses clear sentence-case labels", () => {
    expect(settingsTabSource).toContain('"arXiv categories"');
    expect(settingsTabSource).toContain('"Research topics"');
    expect(settingsTabSource).toContain('"Output & schedule"');
    expect(settingsTabSource).toContain('"API key"');
    expect(settingsTabSource).not.toContain('"+ Add Category"');
  });

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
    expect(apiKeyBody).toContain("await persistApiKeyChange(");
    expect(apiKeyBody).toContain("next,");
    expect(apiKeyBody).toContain('          "",');
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

  it("renders one understandable automatic detail-note setting near topics", () => {
    const headingIndex = settingsTabSource.indexOf('"Research topics"');
    const policyIndex = settingsTabSource.indexOf('"Automatic detail notes"');
    const timezoneIndex = settingsTabSource.indexOf('.setName("Timezone")');
    expect(policyIndex).toBeGreaterThan(headingIndex);
    expect(policyIndex).toBeLessThan(timezoneIndex);
    expect(settingsTabSource).toContain("Only topics with Detail report enabled are eligible");
    expect(settingsTabSource).toContain("Manual summaries are unaffected");
    expect(settingsTabSource).toContain('.addOption("conservative", "Fewer")');
    expect(settingsTabSource).toContain('.addOption("balanced", "Recommended")');
    expect(settingsTabSource).toContain('.addOption("broad", "More")');
    expect(settingsTabSource).toContain('d.addOption("custom", "Custom (current values)")');
    expect(settingsTabSource).toContain('s.detailSelection.profile === "custom"');
    expect(settingsTabSource).toContain("detailSelectionPreset(profile)");
    expect(settingsTabSource).toContain("await this.plugin.saveSettings()");
  });

  it("does not expose automatic detail thresholds or numeric controls", () => {
    expect(settingsTabSource).not.toContain('"Normal threshold"');
    expect(settingsTabSource).not.toContain('"Exceptional threshold"');
    expect(settingsTabSource).not.toContain('"Soft limit"');
    expect(settingsTabSource).not.toContain("renderDetailSelectionNumber");
    expect(settingsTabSource).not.toContain("detail-selection-number");
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
