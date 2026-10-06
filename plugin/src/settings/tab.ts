import { isValidLocalTime, runWindowTimeOptions, getTopicSettingField } from "@arxiv-daily/core";
import { getBusinessSetting, TIMEZONE_OPTIONS, REASONING_EFFORT_OPTIONS, type BusinessSettingId, type BusinessSettingsContext } from "@arxiv-daily/core";
export { TIMEZONE_OPTIONS } from "@arxiv-daily/core";
import {
  App,
  type ButtonComponent,
  Modal,
  Notice,
  PluginSettingTab,
  requireApiVersion,
  Setting,
  type SettingDefinitionItem,
} from "obsidian";
import type ArxivDailyPlugin from "../../main";
import {
  buildSettingDefinitions,
  dailyAutoSendDesc,
  libraryDirectionsRowDesc,
  readSettingValue,
  SETTING_KEYS,
  settingsSectionClass,
  type SettingsSection,
} from "./definitions";
import {
  SettingsChangeError,
  isValidTimezone,
  type SettingsValueChange,
} from "./change-service";
import * as declarativeRows from "./declarative-rows";
import {
  describeResult,
  detailSelectionPreset,
  formatDate,
  isCancellationError,
  todayInTz,
  type LogLevel,
  type PluginSettings,
} from "@arxiv-daily/core";
import { ARXIV_CATEGORIES } from "@arxiv-daily/core";
import {
  TOPIC_TEMPLATES,
  deriveTopicDescription,
  normalizeTopic,
  topicFromSeed,
} from "@arxiv-daily/core";
import type { Topic } from "@arxiv-daily/core";
import { slugify } from "@arxiv-daily/core";
import {
  validateVaultRelativeDirectory,
  vaultRelativeDirectoriesCollide,
} from "@arxiv-daily/core";
import { arxivCategories } from "@arxiv-daily/core";
import {
  getSetupStatus,
  isSetupComplete,
  markSetupGuideCompleteIfDone,
  type SetupStatus,
} from "../onboarding";
import { openDashboardView, refreshOpenDashboardViews } from "../dashboard/view";
import { redactText } from "@arxiv-daily/core";
import {
  ARXIV_DAILY_DOCS_URL,
  ARXIV_DAILY_REPO_URL,
  buildBugReportUrl,
  buildFeatureRequestUrl,
} from "../feedback";
import { ObsidianResourceOpener } from "../hosts/obsidian/resource-opener";
import {
  libraryRowPresentation,
} from "../library/connection";
import type { LibraryIndexStatus } from "../library/index-status";
import { describeFullTextIndexCompletion } from "../library/index-completion";
import {
  confirmLibraryAuthorization,
  confirmLibraryRevocation,
} from "../library/modal";
import { renderSensitiveInput } from "./sensitive-input";


function addBusinessOptions<T extends { addOption(value: string, label: string): unknown }>(dropdown: T, id: BusinessSettingId, context: Partial<BusinessSettingsContext> = {}): T {
  for (const [value,label] of Object.entries(getBusinessSetting(id,context).options ?? {})) dropdown.addOption(value,label);
  return dropdown;
}

export function validateOutputDirectoryDraft(
  draft: string,
  siblingDirectory?: string,
) {
  const validation = validateVaultRelativeDirectory(draft);
  if (
    validation.ok &&
    validation.value &&
    siblingDirectory &&
    vaultRelativeDirectoriesCollide(validation.value, siblingDirectory)
  ) {
    return {
      ok: false as const,
      reason: "Daily and papers directories must be different",
    };
  }
  return validation;
}

export type ModelFetchNotice =
  | { kind: "success"; count: number }
  | { kind: "empty" }
  | { kind: "error"; message: string };

export function modelFetchNoticeMessage(
  result: ModelFetchNotice,
  secrets: readonly string[] = [],
): string {
  switch (result.kind) {
    case "success":
      return `API connection successful. Found ${result.count} models.`;
    case "empty":
      return "API connection successful, but no available models were found.";
    case "error":
      return `API connection failed: ${redactText(result.message, { secrets })}`;
  }
}

export type LlmHttpWarning =
  | { kind: "plaintext"; message: string }
  | { kind: "local"; message: string };

export function llmHttpWarning(baseUrl: string): LlmHttpWarning | null {
  let url: URL;
  try {
    url = new URL(baseUrl.trim());
  } catch {
    return null;
  }
  if (url.protocol !== "http:") return null;
  if (isLoopbackHost(url.hostname)) {
    return {
      kind: "local",
      message: "This address uses plain HTTP on this computer. Only continue if you meant to use a local AI service.",
    };
  }
  return {
    kind: "plaintext",
    message: "This address uses plain HTTP. Your API key would be sent without encryption—prefer HTTPS.",
  };
}

function isLoopbackHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return host === "localhost" || host === "::1" || /^127(?:\.\d{1,3}){3}$/.test(host);
}

function isLogLevel(value: string): value is LogLevel {
  return value === "debug" || value === "info" || value === "warn" || value === "error";
}

/**
 * Style a button as destructive. `setDestructive` replaced `setWarning` in
 * Obsidian 1.13.0 (above this plugin's 1.4.0 `minAppVersion`), so this calls
 * it only when the running app supports it and otherwise falls back to the
 * older styling via a structural type (not the deprecated declaration) so
 * the fallback itself does not re-trigger the deprecation warning.
 */
interface LegacyWarningButton {
  setWarning(): unknown;
}

function applyDestructiveButtonStyle(button: ButtonComponent): void {
  if (requireApiVersion("1.13.0")) {
    button.setDestructive();
  } else {
    (button as unknown as LegacyWarningButton).setWarning();
  }
}

/**
 * How often the Library row may be rewritten while a run reports.
 *
 * Indexing reports once per paper, which on a large library is several times a
 * second and, on a small fast one, faster than a person can read. The interval
 * is what turns that stream into a readable count; it is not a re-render budget,
 * because the updates it paces do not re-render anything.
 */
export const LIBRARY_INDEX_PROGRESS_INTERVAL_MS = 250;

/**
 * The parts of a rendered Library row that a reporting run may rewrite.
 *
 * Buttons are held as components rather than elements so a progress update goes
 * through the same `setButtonText` / `setDisabled` the render used; writing to
 * the element directly would drift from whatever else those set.
 */
interface LibraryRowElements {
  descEl: HTMLElement;
  primary?: ButtonComponent;
  cancel?: ButtonComponent;
}

interface TopicFocusSnapshot {
  topicId: string;
  field: "name" | "direction" | "detail";
  directionId?: string;
  selectionStart: number | null;
  selectionEnd: number | null;
  selectionDirection: "forward" | "backward" | "none" | null;
  scrollTop: number;
  scrollLeft: number;
}

export class ArxivDailySettingTab extends PluginSettingTab {
  private expandedTopics = new Set<string>();
  private libraryRowElements: LibraryRowElements | undefined;
  /** Which structure the row was last rendered with: with a run, or without. */
  private libraryRowShowsRun = false;
  private libraryRowHasIndex = false;
  private libraryRowPreparationError: string | undefined;
  private libraryStatusUnsubscribe: (() => void) | undefined;
  private libraryStatusFlushTimer: number | undefined;
  private pendingLibraryIndexStatus: LibraryIndexStatus | undefined;
  private readonly controlRevisions = new WeakMap<object, number>();
  private readonly declarativeKeyRevisions = new Map<string, number>();
  private readonly pendingTopicEdits = new Set<Promise<void>>();
  private declarativeSetupGuideRow: Setting | undefined;
  private pendingTopicFocusId: string | undefined;
  /** Kept on the tab so a guide re-render during the run still shows it. */
  private firstReportRunning = false;
  private pendingTopicDeletionAnchor:
    | { topicId: string; scroller: HTMLElement; top: number }
    | undefined;

  constructor(app: App, public plugin: ArxivDailyPlugin) {
    super(app, plugin);
  }

  /**
   * Declarative settings for Obsidian 1.13+ (searchable in Settings
   * search). Host callbacks bind the shared row renderers and the tab's
   * mutation methods; display() stays as the <1.13 fallback. Only called
   * by the framework on 1.13+, so no version guard is needed here.
   */
  override getSettingDefinitions(): SettingDefinitionItem[] {
    return buildSettingDefinitions({
      plugin: this.plugin,
      showSetupGuide: this.shouldShowSetupGuide(),
      automaticEmailSupported: this.plugin.automaticEmailSupported(),
      renderSetupGuideRow: (setting) =>
        declarativeRows.renderSetupGuideRow(this, setting),
      renderScheduleEnabledRow: (setting) =>
        declarativeRows.renderScheduleEnabledRow(this, setting),
      renderLlmBaseUrlRow: (setting) =>
        declarativeRows.renderLlmBaseUrlRow(this, setting),
      renderApiKeyRow: (setting) =>
        declarativeRows.renderApiKeyRow(this, setting),
      renderModelRow: (setting) =>
        declarativeRows.renderModelRow(this, setting),
      renderReasoningEffortRow: (setting) =>
        declarativeRows.renderReasoningEffortRow(this, setting),
      renderEmbeddingModeRow: (setting) =>
        declarativeRows.renderEmbeddingModeRow(this, setting),
      renderEmbeddingBaseUrlRow: (setting) =>
        declarativeRows.renderEmbeddingBaseUrlRow(this, setting),
      renderEmbeddingApiKeyRow: (setting) =>
        declarativeRows.renderEmbeddingApiKeyRow(this, setting),
      renderEmbeddingModelRow: (setting) =>
        declarativeRows.renderEmbeddingModelRow(this, setting),
      renderEmbeddingDimensionRow: (setting) =>
        declarativeRows.renderEmbeddingDimensionRow(this, setting),
      renderPdfParserSidecarEnabledRow: (setting) =>
        declarativeRows.renderPdfParserSidecarEnabledRow(this, setting),
      renderPdfParserSidecarCapabilitiesUrlRow: (setting) =>
        declarativeRows.renderPdfParserSidecarCapabilitiesUrlRow(this, setting),
      renderPdfParserSidecarParseUrlRow: (setting) =>
        declarativeRows.renderPdfParserSidecarParseUrlRow(this, setting),
      renderLibraryConnectionRow: (setting) =>
        declarativeRows.renderLibraryConnectionRow(this, setting),
      renderLibraryDirectionsRow: (setting) =>
        declarativeRows.renderLibraryDirectionsRow(this, setting),
      renderLibraryGuideRow: (setting) =>
        declarativeRows.renderLibraryGuideRow(this, setting),
      renderCategoryRow: (setting, index) =>
        declarativeRows.renderCategoryRow(this, setting, index),
      renderTopicRow: (setting, index) =>
        declarativeRows.renderTopicRow(this, setting, index),
      renderTimezoneRow: (setting) =>
        declarativeRows.renderTimezoneRow(this, setting),
      renderOutputDirectoryRow: (setting, key) =>
        declarativeRows.renderOutputDirectoryRow(this, setting, key),
      renderEmailSenderRow: (setting, key) =>
        declarativeRows.renderEmailSenderRow(this, setting, key),
      renderRunWindowRow: (setting) =>
        declarativeRows.renderRunWindowRow(this, setting),
      renderTickIntervalRow: (setting) =>
        declarativeRows.renderTickIntervalRow(this, setting),
      renderDailyPaperLimitRow: (setting) =>
        declarativeRows.renderDailyPaperLimitRow(this, setting),
      renderEmailGuideRow: (setting) =>
        declarativeRows.renderEmailGuideRow(this, setting),
      renderEmailModeRow: (setting) =>
        declarativeRows.renderEmailModeRow(this, setting),
      renderEmailToRow: (setting) =>
        declarativeRows.renderEmailToRow(this, setting),
      renderEmailApiKeyRow: (setting) =>
        declarativeRows.renderEmailApiKeyRow(this, setting),
      renderHostedTokenRow: (setting) =>
        declarativeRows.renderHostedTokenRow(this, setting),
      addCategory: () => void this.addCategory(),
      deleteCategory: (index) => void this.deleteCategory(index),
      addTopic: () => this.runAction("add topic", () => this.addTopic()),
    });
  }

  /** Resolve a flat declarative key against the nested settings object. */
  override getControlValue(key: string): unknown {
    return readSettingValue(this.plugin.settings, key);
  }

  /**
   * Persist a flat declarative key. Email To and From mirror display()'s
   * trimming on change; From name is written raw there, so it stays raw
   * here too.
   */
  override async setControlValue(key: string, value: unknown): Promise<void> {
    if (
      typeof value === "string" &&
      (key === SETTING_KEYS.email.to || key === SETTING_KEYS.email.fromEmail)
    ) {
      value = value.trim();
    }
    const revision = (this.declarativeKeyRevisions.get(key) ?? 0) + 1;
    this.declarativeKeyRevisions.set(key, revision);
    try {
      await this.changeSettingValue(key, value);
    } catch (error) {
      if (this.declarativeKeyRevisions.get(key) === revision) {
        this.refreshSettings();
      }
      throw error;
    }
  }

  public async changeSettingValue(key: string, value: unknown): Promise<void> {
    await this.plugin.settingsChanges.changeValue(key, value);
    await this.refreshLibraryAfterEmbeddingChange([key]);
  }

  public async changeSettingValues(
    changes: readonly SettingsValueChange[],
  ): Promise<void> {
    await this.plugin.settingsChanges.change({ changes });
    await this.refreshLibraryAfterEmbeddingChange(changes.map(({ key }) => key));
  }

  private async refreshLibraryAfterEmbeddingChange(keys: readonly string[]): Promise<void> {
    if (keys.some((key) => ["embedding.mode", "embedding.baseUrl", "embedding.model", "embedding.dimension"].includes(key))) {
      await this.plugin.refreshLibraryIndexTrace();
    }
  }

  /**
   * Track renderer drafts so an earlier queued failure cannot overwrite a
   * later draft already displayed by the same control.
   */
  public beginControlChange(control: object): number {
    const revision = (this.controlRevisions.get(control) ?? 0) + 1;
    this.controlRevisions.set(control, revision);
    return revision;
  }

  public isCurrentControlChange(control: object, revision: number): boolean {
    return this.controlRevisions.get(control) === revision;
  }

  /** Resolve the value visible after a rejection finishes. */
  public restoreCurrentControlValue(error: unknown, key: string): unknown {
    if (error instanceof SettingsChangeError) {
      const live = this.getControlValue(key);
      return live === undefined ? error.restoreValue(key) : live;
    }
    return this.getControlValue(key);
  }

  public restoreCurrentStringControlValue(
    error: unknown,
    key: string,
    fallback = "",
  ): string {
    const value = this.restoreCurrentControlValue(error, key);
    return typeof value === "string" ? value : fallback;
  }

  /** Resolve the old value carried by a rejected declarative transaction. */
  public restoreControlValue(error: unknown, key: string): unknown {
    const live = this.getControlValue(key);
    return live === undefined && error instanceof SettingsChangeError
      ? error.restoreValue(key)
      : live;
  }

  public restoreStringControlValue(
    error: unknown,
    key: string,
    fallback = "",
  ): string {
    const value = this.restoreControlValue(error, key);
    return typeof value === "string" ? value : fallback;
  }

  private async saveLegacyControl(
    control: object,
    action: string,
    key: string,
    changes: readonly SettingsValueChange[],
    restoreDisplayed: (value: unknown) => void,
  ): Promise<boolean> {
    const revision = this.beginControlChange(control);
    try {
      await this.changeSettingValues(changes);
      const current = this.isCurrentControlChange(control, revision);
      if (current) restoreDisplayed(this.getControlValue(key));
      return current;
    } catch (error) {
      if (this.isCurrentControlChange(control, revision)) {
        restoreDisplayed(this.restoreCurrentControlValue(error, key));
      }
      this.reportActionError(action, error);
      return false;
    }
  }

  /** Append an accessible circled "?" to a setting name. */
  private attachHelp(setting: Setting, text: string): Setting {
    setting.nameEl.createSpan({
      cls: "arxiv-daily-settings__help",
      text: "?",
      attr: { title: text, "aria-label": text },
    });
    return setting;
  }

  private reportActionError(action: string, error: unknown): void {
    const message = error instanceof Error ? error.message : String(error);
    this.plugin.logger.error(`settings: ${action} failed`, error);
    new Notice(`arXiv Daily: ${action} failed: ${message}`, 10_000);
  }

  public reportSettingsActionError(action: string, error: unknown): void {
    this.reportActionError(action, error);
  }

  public async runActionAndWait(
    action: string,
    operation: () => Promise<unknown>,
  ): Promise<void> {
    try {
      await operation();
    } catch (error) {
      this.reportActionError(action, error);
    }
  }

  public runAction(action: string, operation: () => Promise<unknown>): void {
    void this.runActionAndWait(action, operation);
  }

  /** Inline muted hint, used inside topic cards under a label. */
  private hint(parent: HTMLElement, text: string, id?: string): HTMLElement {
    const hint = parent.createDiv({
      cls: "arxiv-daily-settings__hint",
      text,
    });
    if (id) hint.id = id;
    return hint;
  }

  private sectionHeading(
    containerEl: HTMLElement,
    name: string,
    section: SettingsSection,
    desc?: string,
  ): Setting {
    const heading = new Setting(containerEl).setName(name).setHeading();
    if (desc) heading.setDesc(desc);
    heading.settingEl.addClass("arxiv-daily-settings__section", settingsSectionClass(section));
    heading.settingEl.setAttribute("data-arxiv-daily-section", section);
    // Ensure heading name is easy to scan in long settings pages.
    heading.nameEl?.addClass("arxiv-daily-settings__section-title");
    return heading;
  }

  /** Full-width guide block aligned with Setting name column (not indented desc). */
  /** Guide strip copy for the current email mode; shared with the 1.13+ rows. */
  public emailGuideContent(): { title: string; lines: string[] } {
    const hostedMode = this.plugin.settings.email.mode === "hosted";
    return {
      title: hostedMode ? "Official delivery (Beta)" : "Send yourself",
      lines: hostedMode
        ? [
            "1. Enter your email, then send a verification message.",
            "2. Open the link in that email and copy the code shown on the page.",
            "3. Paste the code below, send a test email, then turn on daily auto-send.",
            "Capacity is limited: only a few messages per inbox per day (tests count). For heavier use, switch to Send yourself.",
          ]
        : [
            "1. Create a free Resend account and an API key at resend.com.",
            "2. Paste the key below. For a quick start, put your Resend account email in Your email and leave From email empty.",
            "3. Send a test email, then turn on daily auto-send when it works.",
          ],
    };
  }

  /**
   * Shared renderer for the full-width guide boxes (email, library): same
   * layout and CSS, keyed by a class prefix so each section keeps its own
   * host/box/title/line classes (see styles.css).
   */
  private renderGuideBox(
    containerEl: HTMLElement,
    opts: { title: string; lines: string[] },
    clsPrefix: "email-guide" | "library-guide",
  ): void {
    const wrap = containerEl.createDiv({
      cls: `arxiv-daily-settings__${clsPrefix}`,
    });
    wrap.createDiv({
      cls: `arxiv-daily-settings__${clsPrefix}-title`,
      text: opts.title,
    });
    for (const line of opts.lines) {
      wrap.createDiv({
        cls: `arxiv-daily-settings__${clsPrefix}-line`,
        text: line,
      });
    }
  }

  private emailGuide(
    containerEl: HTMLElement,
    opts: { title: string; lines: string[] },
  ): void {
    this.renderGuideBox(containerEl, opts, "email-guide");
  }

  /**
   * Personal library intro box copy; shared with the 1.13+ row. Always
   * shown, like the email delivery guide — the section stays optional and
   * its steps stay worth restating even once a library is connected.
   */
  public libraryGuideContent(): { title: string; lines: string[] } {
    return {
      title: "How this works",
      lines: [
        "Optional — daily reports work the same without a library.",
        "1. Choose a folder of PDFs to automatically prepare a title-and-abstract search index. The first preparation scans the folder and queries arXiv with paper IDs or titles; network access is needed. Later preparations reuse the saved scan.",
        "2. Review suggestions (button below) generates topic suggestions on first use and opens saved suggestions afterwards, including PDFs without an arXiv match when their titles and abstracts can be extracted. Only directions you add to Research topics steer daily reports. Remote embedding and model processing always ask first.",
        "Local embedding is the default. Its model downloads once (about 130 MB), then runs locally. If preparation fails or is cancelled, use Retry preparation. Search from the command palette when ready.",
      ],
    };
  }

  private libraryGuide(
    containerEl: HTMLElement,
    opts: { title: string; lines: string[] },
  ): void {
    this.renderGuideBox(containerEl, opts, "library-guide");
  }

  public renderLibraryConnectionControls(setting: Setting): void {
    const indexStatus = this.plugin.libraryIndexStatus.snapshot();
    const row = libraryRowPresentation({
      status: this.plugin.getLibraryConnectionStatus(),
      embeddingMode: this.plugin.settings.embedding.mode,
      ...(indexStatus.activity ? { activity: indexStatus.activity } : {}),
      ...(indexStatus.lastRun ? { lastRun: indexStatus.lastRun } : {}),
      ...(indexStatus.preparationError ? { preparationError: indexStatus.preparationError } : {}),
    });
    setting.setDesc(row.description);
    setting.controlEl.addClass("arxiv-daily-settings__library-controls");
    // Mirrors styles.css's `.setting-item:has(> .arxiv-daily-settings__library-controls)`
    // as a static class on the row itself, added at the same moment the
    // control class above is, since the control is always a direct child of
    // this setting item for the lifetime of this row.
    setting.settingEl.addClass("arxiv-daily-settings__library-row");
    const live: LibraryRowElements = { descEl: setting.descEl };

    setting.addButton((button) =>
      button
        .setButtonText(row.chooseFolder.label)
        .setDisabled(row.chooseFolder.disabled)
        .onClick(() => this.runAction("choose personal library", () => this.chooseLibraryRoot())),
    );
    // There is no authorization button: remote consent is asked in place when
    // remote embedding is switched on, and otherwise in front of indexing.
    const primary = row.primary;
    if (primary) {
      setting.addButton((button) => {
        button
          .setButtonText(primary.label)
          .setCta()
          .setDisabled(primary.disabled)
          .onClick(() => this.runAction("index personal library titles and abstracts", () => this.indexPersonalLibraryFullText()));
        live.primary = button;
      });
    }
    // While a run is in flight the row's third button is its stop control, so
    // the row still offers at most three: folder, run, stop.
    const cancel = row.cancel;
    if (cancel) {
      setting.addButton((button) => {
        button.setButtonText(cancel.label);
        applyDestructiveButtonStyle(button);
        button
          .setDisabled(cancel.disabled)
          .onClick(() => this.cancelLibraryIndexing());
        live.cancel = button;
      });
    }
    // Secondary library actions (preview / scan / reload) live in the command
    // palette only, so this row never grows past three buttons and needs no
    // menu. Direction review has its own row below instead, since it is a
    // distinct, user-facing feature rather than upkeep.
    const revoke = row.revoke;
    if (revoke) {
      setting.addButton((button) =>
        button
          .setButtonText(revoke.label)
          .onClick(() => this.runAction("revoke personal library", () => this.revokeLibraryAuthorization())),
      );
    }

    this.libraryRowElements = live;
    this.libraryRowShowsRun = Boolean(row.cancel);
    this.libraryRowHasIndex = Boolean(indexStatus.lastRun?.papers);
    this.libraryRowPreparationError = indexStatus.preparationError;
    this.watchLibraryIndexStatus();
  }

  /**
   * Follow the indexing run while this tab is open.
   *
   * Subscribed from the row's own render rather than from a tab lifecycle hook,
   * because the row is the only thing that wants it and both render paths go
   * through here. The immediate replay the store performs on subscribe is
   * dropped: the render that is asking for the subscription has already drawn
   * that state, and re-entering `refreshSettings` from inside a render would
   * not terminate.
   */
  private watchLibraryIndexStatus(): void {
    if (this.libraryStatusUnsubscribe) return;
    let replaying = true;
    this.libraryStatusUnsubscribe = this.plugin.libraryIndexStatus.subscribe((status) => {
      if (replaying) return;
      this.onLibraryIndexStatusChange(status);
    });
    replaying = false;
  }

  /**
   * A run reports far more often than a settings page should re-render — once
   * per paper, which on a large library is several times a second.
   *
   * Two different updates hide behind that. Starting and stopping change which
   * buttons the row has, so they go through the full re-render (twice a run, and
   * `refreshSettings` keeps the scroll position). Everything in between only
   * changes text on buttons that are already there, so it is written straight
   * into the rendered elements on a timer — no re-render, which is also what
   * keeps a half-typed value in another row from being thrown away mid-run.
   */
  private onLibraryIndexStatusChange(status: LibraryIndexStatus): void {
    if (Boolean(status.activity) !== this.libraryRowShowsRun
      || (!status.activity && Boolean(status.lastRun?.papers) !== this.libraryRowHasIndex)
      || status.preparationError !== this.libraryRowPreparationError) {
      this.clearLibraryStatusFlush();
      this.refreshSettings();
      return;
    }
    this.pendingLibraryIndexStatus = status;
    if (this.libraryStatusFlushTimer !== undefined) return;
    const view = this.libraryRowElements?.descEl.ownerDocument.defaultView;
    if (!view) {
      this.flushLibraryIndexStatus();
      return;
    }
    this.libraryStatusFlushTimer = view.setTimeout(() => {
      this.libraryStatusFlushTimer = undefined;
      this.flushLibraryIndexStatus();
    }, LIBRARY_INDEX_PROGRESS_INTERVAL_MS);
  }

  /** Write the latest reported progress into the row that is already on screen. */
  private flushLibraryIndexStatus(): void {
    const status = this.pendingLibraryIndexStatus;
    const elements = this.libraryRowElements;
    this.pendingLibraryIndexStatus = undefined;
    if (!status || !elements || !elements.descEl.isConnected) return;
    const row = libraryRowPresentation({
      status: this.plugin.getLibraryConnectionStatus(),
      embeddingMode: this.plugin.settings.embedding.mode,
      ...(status.activity ? { activity: status.activity } : {}),
      ...(status.lastRun ? { lastRun: status.lastRun } : {}),
      ...(status.preparationError ? { preparationError: status.preparationError } : {}),
    });
    elements.descEl.textContent = row.description;
    if (elements.primary && row.primary) {
      elements.primary.setButtonText(row.primary.label).setDisabled(row.primary.disabled);
    }
    if (elements.cancel && row.cancel) {
      elements.cancel.setButtonText(row.cancel.label).setDisabled(row.cancel.disabled);
    }
  }

  private clearLibraryStatusFlush(): void {
    this.pendingLibraryIndexStatus = undefined;
    if (this.libraryStatusFlushTimer === undefined) return;
    this.libraryRowElements?.descEl.ownerDocument.defaultView?.clearTimeout(
      this.libraryStatusFlushTimer,
    );
    this.libraryStatusFlushTimer = undefined;
  }

  public cancelLibraryIndexing(): void {
    if (this.plugin.cancelPersonalLibraryIndexing()) return;
    // The row can be a frame behind the run it describes; say so rather than
    // leaving a press that did nothing.
    new Notice("arXiv Daily: that indexing run has already finished.");
    this.refreshSettings();
  }

  /**
   * Obsidian calls this when the tab is closed, on both render paths. The
   * subscription and its timer are the only things this tab leaves running.
   */
  override hide(): void {
    this.clearLibraryStatusFlush();
    this.libraryStatusUnsubscribe?.();
    this.libraryStatusUnsubscribe = undefined;
    this.libraryRowElements = undefined;
    super.hide();
  }

  public async chooseLibraryRoot(): Promise<void> {
    const result = await this.plugin.selectLibraryRoot();
    if (result === "unsupported") {
      new Notice("arXiv Daily: system folder selection is unavailable in this Obsidian desktop version.", 10_000);
      return;
    }
    if (result === "selected") {
      this.refreshSettings();
      await this.indexPersonalLibraryFullText();
    }
  }

  public renderLibrarySuggestionsControls(setting: Setting): void {
    const index = this.plugin.libraryIndexStatus.snapshot();
    setting.setDesc(libraryDirectionsRowDesc(this.plugin));
    setting.addButton((button) => button
      .setButtonText("Review suggestions")
      .setDisabled(Boolean(index.activity) || Boolean(index.preparationError) || !index.lastRun?.papers)
      .onClick(() => {
        const current = this.plugin.libraryIndexStatus.snapshot();
        if (current.activity || current.preparationError || !current.lastRun?.papers) return;
        this.runAction("open personal library direction review", async () => {
          this.plugin.openPersonalLibraryDirectionReview({ generateIfMissing: true });
        });
      }));
  }

  /**
   * The single title-and-abstract disclosure for remote embedding (ADR 0008), reached
   * from whichever moment comes first: switching to remote, selecting a folder
   * afterwards, moving the endpoint, or building the index while a legacy
   * remote configuration is still ungranted. `undisclosable` means the folder
   * or the endpoint is not known yet, so there is nothing honest to disclose.
   */
  private async requestRemoteFullTextConsent(
    options: { applyBeforeGrant?: () => Promise<void> } = {},
  ): Promise<"granted" | "declined" | "undisclosable"> {
    let disclosure;
    try {
      disclosure = this.plugin.getLibraryAuthorizationDisclosure({
        embeddingMode: "remote",
      });
    } catch {
      // An endpoint that cannot even be rendered as a URL cannot be disclosed;
      // the existing configuration check reports it when indexing runs.
      return "undisclosable";
    }
    if (!disclosure?.embeddingEndpoint) return "undisclosable";
    if (!await confirmLibraryAuthorization(this.app, disclosure)) return "declined";
    await options.applyBeforeGrant?.();
    await this.plugin.authorizeLibraryProcessing(
      disclosure.authorizationFingerprint,
    );
    this.refreshSettings();
    return "granted";
  }

  /**
   * Apply an Embedding mode change from either settings path. Turning remote
   * embedding on asks for title-and-abstract disclosure in place: confirming switches
   * and authorizes in one step, declining leaves the mode and the grant alone.
   */
  public async applyEmbeddingModeChange(next: "local" | "remote"): Promise<boolean> {
    const settings = this.plugin.settings;
    if (next === settings.embedding.mode) return false;
    const markChosen: SettingsValueChange[] = settings.embedding.initialChoiceDone
      ? []
      : [{ key: SETTING_KEYS.embedding.initialChoiceDone, value: true }];
    const applyMode = async () => {
      await this.changeSettingValues([
        { key: SETTING_KEYS.embedding.mode, value: next },
        ...markChosen,
      ]);
    };
    if (next === "local") {
      await applyMode();
      return true;
    }
    const consent = await this.requestRemoteFullTextConsent({
      // The grant is written against the live settings, so the mode has to be
      // remote before authorizing — but only once the disclosure is accepted.
      applyBeforeGrant: applyMode,
    });
    if (consent === "declined") return false;
    if (consent === "undisclosable") {
      // No folder or no endpoint to name yet: switch now, disclose at the
      // first moment that can name them (folder selection, or indexing).
      await applyMode();
      return true;
    }
    new Notice("arXiv Daily: remote embedding enabled and title-and-abstract processing authorized.");
    return true;
  }

  /**
   * Save an embedding field that can move where titles and abstracts is sent. When the
   * change invalidates a live grant, the same disclosure is shown for the new
   * destination; declining restores the authorized value, so an authorized
   * library never silently points somewhere the user did not agree to.
   * Returns the value the control should display.
   */
  public async saveEmbeddingEndpointField(
    key: typeof SETTING_KEYS.embedding.baseUrl | typeof SETTING_KEYS.embedding.model,
    next: string,
  ): Promise<string> {
    const read = () => key === SETTING_KEYS.embedding.baseUrl
      ? this.plugin.settings.embedding.baseUrl
      : this.plugin.settings.embedding.model;
    const previous = read();
    if (next === previous) return previous;
    const wasAuthorized = this.plugin.getLibraryConnectionStatus().kind === "authorized";
    await this.changeSettingValue(key, next);
    if (!wasAuthorized || this.plugin.settings.embedding.mode !== "remote") return read();
    if (this.plugin.getLibraryConnectionStatus().kind === "authorized") return read();
    const consent = await this.requestRemoteFullTextConsent();
    // `undisclosable` means the field was emptied, leaving no destination to
    // disclose; keep the edit and let the indexing gate report the gap.
    if (consent !== "declined") return read();
    await this.changeSettingValue(key, previous);
    this.refreshSettings();
    new Notice(
      "arXiv Daily: embedding endpoint change cancelled — the authorized endpoint is unchanged.",
      10_000,
    );
    return read();
  }

  public async revokeLibraryAuthorization(): Promise<void> {
    // Revoking a remote grant also returns embedding to local, so the plugin
    // is never left in a remote-but-unauthorized state nobody asked for.
    const switchesToLocal = this.plugin.settings.embedding.mode === "remote";
    if (!await confirmLibraryRevocation(this.app, { switchesToLocal })) return;
    await this.plugin.revokeLibraryProcessing();
    if (switchesToLocal) {
      await this.changeSettingValue(SETTING_KEYS.embedding.mode, "local");
    }
    new Notice(
      switchesToLocal
        ? "arXiv Daily: authorization revoked and embedding switched back to local. Rebuild the index when you want to search again."
        : "arXiv Daily: personal library authorization revoked.",
      10_000,
    );
    this.refreshSettings();
  }

  public async indexPersonalLibraryFullText(): Promise<void> {
    if (!await this.ensureRemoteEmbeddingConsent()) return;
    new Notice("arXiv Daily: indexing personal library titles and abstracts…");
    try {
      const summary = await this.plugin.indexPersonalLibraryFullText();
      new Notice(
        `arXiv Daily: ${describeFullTextIndexCompletion(summary, {
          onCompletionSuffix: "Search from the Dashboard.",
          libraryContext: this.plugin.getLastFullTextIndexLibraryContext(),
        })}`,
        10_000,
      );
    } catch (error) {
      // A run the reader stopped from this row is an outcome, not a fault: it
      // must not come back as "indexing failed" over a button they just pressed.
      if (isCancellationError(error)) {
        new Notice(
          "arXiv Daily: indexing cancelled. You can build the index again when ready.",
          10_000,
        );
        return;
      }
      this.plugin.logger.error("settings: personal library title-and-abstract indexing failed", error);
      throw error;
    }
  }

  /**
   * Last gate in front of remote title-and-abstract indexing. Configurations that were
   * remote before this consent flow existed — or whose grant an endpoint edit
   * invalidated — are asked here instead of being blocked by an error.
   */
  private async ensureRemoteEmbeddingConsent(): Promise<boolean> {
    if (this.plugin.settings.embedding.mode !== "remote") return true;
    if (this.plugin.getLibraryConnectionStatus().kind === "authorized") return true;
    const consent = await this.requestRemoteFullTextConsent();
    if (consent === "declined") {
      new Notice(
        "arXiv Daily: indexing cancelled. Remote embedding needs your confirmation before titles and abstracts can leave this device.",
        10_000,
      );
      return false;
    }
    // `undisclosable` means the remote endpoint is not configured yet; let the
    // existing configuration check report that instead of inventing a modal.
    return true;
  }

  /** Returns whether the new categories were saved (a failure is rolled back and reported). */
  public async setArxivCategories(categories: string[]): Promise<boolean> {
    const normalized = normalizeUniqueCategories(categories);
    return this.saveArxivEdit("save categories", (arxiv) => {
      arxiv.categories = normalized;
      if (normalized[0]) arxiv.category = normalized[0];
    });
  }

  /** Queue edits against the latest settings; failed saves leave them unchanged. */
  private async saveArxivEdit(
    action: string,
    mutate: (arxiv: PluginSettings["arxiv"]) => void,
  ): Promise<boolean> {
    try {
      await this.plugin.settingsChanges.changeComputed((current) => {
        mutate(current.arxiv);
        return { changes: [
          { key: "arxiv.category", value: current.arxiv.category },
          { key: "arxiv.categories", value: current.arxiv.categories },
          { key: "arxiv.topics", value: current.arxiv.topics },
        ] };
      });
      return true;
    } catch (error) {
      this.reportActionError(action, error);
      this.refreshSettings();
      return false;
    }
  }

  /** Re-render the tab: declarative update() on Obsidian 1.13+, display() otherwise. */
  public refreshSettings(): void {
    const scrollSnapshot = this.captureSettingsScroll();
    const focusSnapshot = this.captureTopicFocus();
    if (
      requireApiVersion("1.13.0") &&
      this.getSettingDefinitions().length > 0
    ) {
      this.update();
    } else {
      this.renderLegacySettings();
    }
    if (!this.pendingTopicFocusId && !this.pendingTopicDeletionAnchor) {
      this.restoreTopicFocus(focusSnapshot);
      this.restoreSettingsScroll(scrollSnapshot);
    }
  }

  /** Refresh external topic changes only after the visible edits have settled. */
  public async refreshAfterTopicChanges(): Promise<void> {
    await this.plugin.settingsChanges.changeComputed(() => ({ changes: [] }));
    // Input can continue while the first save is pending. Include those newer
    // edits and their failure restoration before replacing any controls.
    while (this.pendingTopicEdits.size > 0) {
      await Promise.allSettled([...this.pendingTopicEdits]);
    }
    this.refreshSettings();
  }

  private captureTopicFocus(): TopicFocusSnapshot | null {
    const input = this.containerEl.ownerDocument.activeElement;
    if (!(input instanceof HTMLInputElement || input instanceof HTMLTextAreaElement)
      || !this.containerEl.contains(input)) return null;
    const topicId = input.closest<HTMLElement>(".arxiv-daily-settings__topic-card")?.dataset.arxivDailyTopicId;
    if (!topicId) return null;
    const field = input.dataset.directionId ? "direction"
      : input.classList.contains("arxiv-daily-settings__topic-name-input") ? "name"
        : input.classList.contains("arxiv-daily-settings__topic-detail-checkbox") ? "detail" : null;
    if (!field) return null;
    return {
      topicId, field, directionId: input.dataset.directionId,
      selectionStart: input.selectionStart, selectionEnd: input.selectionEnd,
      selectionDirection: input.selectionDirection, scrollTop: input.scrollTop, scrollLeft: input.scrollLeft,
    };
  }

  private restoreTopicFocus(snapshot: TopicFocusSnapshot | null): void {
    if (!snapshot) return;
    const card = this.findTopicCard(snapshot.topicId);
    if (!card) return;
    const input = snapshot.field === "direction"
      ? Array.from(card.querySelectorAll<HTMLTextAreaElement>(".arxiv-daily-settings__topic-direction-input"))
        .find((item) => item.dataset.directionId === snapshot.directionId)
      : card.querySelector<HTMLInputElement>(snapshot.field === "name"
        ? ".arxiv-daily-settings__topic-name-input" : ".arxiv-daily-settings__topic-detail-checkbox");
    if (!input) return;
    input.focus({ preventScroll: true });
    if (snapshot.selectionStart !== null && snapshot.selectionEnd !== null) {
      input.setSelectionRange(snapshot.selectionStart, snapshot.selectionEnd, snapshot.selectionDirection ?? undefined);
    }
    input.scrollTop = snapshot.scrollTop;
    input.scrollLeft = snapshot.scrollLeft;
  }

  private captureSettingsScroll(): Array<{
    element: HTMLElement;
    top: number;
    left: number;
  }> {
    const snapshot: Array<{ element: HTMLElement; top: number; left: number }> = [];
    const view = this.containerEl.ownerDocument.defaultView;
    for (
      let element: HTMLElement | null = this.containerEl;
      element;
      element = element.parentElement
    ) {
      const overflowY = view?.getComputedStyle(element).overflowY;
      if (
        element.scrollTop !== 0 ||
        element.scrollLeft !== 0 ||
        overflowY === "auto" ||
        overflowY === "scroll"
      ) {
        snapshot.push({ element, top: element.scrollTop, left: element.scrollLeft });
      }
    }
    return snapshot;
  }

  private restoreSettingsScroll(
    snapshot: Array<{ element: HTMLElement; top: number; left: number }>,
  ): void {
    if (snapshot.length === 0) return;
    const restore = () => {
      for (const { element, top, left } of snapshot) {
        element.scrollTop = top;
        element.scrollLeft = left;
      }
    };
    restore();
  }

  /** Keep the Obsidian <1.13 fallback behind one explicit deprecated API call. */
  private renderLegacySettings(): void {
    const display = Reflect.get(this, "display") as (() => void) | undefined;
    display?.call(this);
  }

  /** Append a category (the first arXiv option not already in the list). */
  public async addCategory(): Promise<void> {
    const categories = arxivCategories(this.plugin.settings.arxiv);
    if (!await this.setArxivCategories([
      ...categories,
      nextCategoryCandidate(categories),
    ])) return;
    this.refreshSettings();
  }

  /** Remove a category by index; keeps the last remaining category. */
  public async deleteCategory(index: number): Promise<void> {
    const categories = arxivCategories(this.plugin.settings.arxiv);
    if (categories.length <= 1) return;
    if (!await this.setArxivCategories(categories.filter((_, j) => j !== index))) return;
    this.refreshSettings();
  }

  /** Append a blank, expanded topic card. */
  public async addTopic(): Promise<void> {
    const newId = crypto.randomUUID();
    const saved = await this.saveArxivEdit("add topic", ({ topics }) => {
      topics.push(normalizeTopic({
        id: newId,
        name: "",
        tag: autoTopicTag("", topics, newId),
        description: "",
        detail: false,
      }));
    });
    if (!saved) return;
    this.expandedTopics.add(newId);
    this.pendingTopicFocusId = newId;
    this.refreshSettings();
    this.focusPendingTopic();
  }

  /** Delete a topic after confirmation; returns whether it was deleted. */
  public async deleteTopic(topicRef: number | string): Promise<boolean> {
    const topics = this.plugin.settings.arxiv.topics;
    const index = typeof topicRef === "number" ? topicRef : topics.findIndex(({ id }) => id === topicRef);
    const topic = topics[index];
    if (!topic) return false;
    const topicId = topic.id;
    const topicName = topic.name.trim() || "(unnamed)";
    const confirmed = await this.confirmReplace(
      `Delete the research topic "${topicName}"? This cannot be undone.`,
      "Delete",
    );
    if (!confirmed) return false;
    this.pendingTopicDeletionAnchor = this.captureTopicDeletionAnchor(
      topics[index + 1]?.id ?? topics[index - 1]?.id,
    );
    const saved = await this.saveArxivEdit("delete topic", (arxiv) => {
      const currentIndex = arxiv.topics.findIndex(({ id }) => id === topicId);
      if (currentIndex >= 0) arxiv.topics.splice(currentIndex, 1);
    });
    if (!saved) {
      this.pendingTopicDeletionAnchor = undefined;
      return false;
    }
    this.expandedTopics.delete(topic.id);
    this.refreshSettings();
    this.restoreTopicDeletionAnchor();
    return true;
  }

  /**
   * Apply a quick-start template, replacing topics (and categories) after
   * confirmation when the current setup would be overwritten.
   */
  public async applyTopicTemplate(templateId: string): Promise<void> {
    const tpl = TOPIC_TEMPLATES.find((t) => t.id === templateId);
    if (!tpl) return;
    const settings = this.plugin.settings;
    const categories = arxivCategories(settings.arxiv);
    const apply = async () => {
      const saved = await this.saveArxivEdit("apply topic template", (arxiv) => {
        arxiv.category = tpl.category;
        arxiv.categories = [tpl.category];
        arxiv.topics.splice(
          0,
          arxiv.topics.length,
          ...tpl.topics.map(topicFromSeed),
        );
      });
      if (saved) this.refreshSettings();
    };
    const replacesCategories = categoriesWillChange(categories, [tpl.category]);
    if (settings.arxiv.topics.length === 0 && !replacesCategories) {
      await apply();
      return;
    }
    const confirmed = await this.confirmReplace(
      quickStartTemplateConfirmMessage(
        settings.arxiv.topics.length,
        tpl.name,
        replacesCategories,
      ),
    );
    if (confirmed) await apply();
  }

  /**
   * Changes for one sidecar endpoint edit. Both endpoints must share an
   * origin, and each field saves alone, so moving one endpoint to another
   * host or port moves the other to the same origin in the same change.
   */
  public sidecarUrlChanges(
    key: "pdfParserSidecar.capabilitiesUrl" | "pdfParserSidecar.parseUrl",
    next: string,
  ): SettingsValueChange[] {
    const changes: SettingsValueChange[] = [{ key, value: next }];
    const otherKey = key === SETTING_KEYS.pdfParserSidecar.capabilitiesUrl
      ? SETTING_KEYS.pdfParserSidecar.parseUrl
      : SETTING_KEYS.pdfParserSidecar.capabilitiesUrl;
    const otherValue = this.getControlValue(otherKey);
    const other = typeof otherValue === "string" ? otherValue : "";
    try {
      const nextUrl = new URL(next);
      const otherUrl = new URL(other);
      if (nextUrl.origin !== otherUrl.origin) {
        changes.push({
          key: otherKey,
          value: `${nextUrl.origin}${otherUrl.pathname}${otherUrl.search}`,
        });
      }
    } catch {
      // An unparsable URL is left to the settings validation to reject.
    }
    return changes;
  }

  public async saveTimezone(timezone: string): Promise<void> {
    await this.plugin.settingsChanges.changeValue("arxiv.timezone", timezone);
  }

  public bindTimezoneDraftInput(
    input: HTMLInputElement,
    select?: HTMLSelectElement,
  ): void {
    let saveQueue = Promise.resolve();
    let latestSave: { draft: string; promise: Promise<void> } | undefined;
    let latestSuccessful = this.plugin.settings.arxiv.timezone;
    const syncSelect = (timezone: string) => {
      if (!select) return;
      if (!Array.from(select.options).some((option) => option.value === timezone)) {
        select.createEl("option", {
          value: timezone,
          text: `${timezone} — custom`,
        });
      }
      select.value = timezone;
    };
    const commit = (): Promise<void> => {
      const draft = input.value.trim();
      if (!draft) return Promise.resolve();
      if (latestSave?.draft === draft) return latestSave.promise;
      if (!isValidTimezone(draft)) {
        input.setCustomValidity("Invalid timezone");
        input.addClass("is-invalid");
        return Promise.resolve();
      }
      input.setCustomValidity("");
      input.removeClass("is-invalid");
      const revision = this.beginControlChange(input);
      const operation = saveQueue.then(async () => {
        try {
          await this.saveTimezone(draft);
          latestSuccessful = draft;
          if (this.isCurrentControlChange(input, revision)) {
            input.value = "";
            input.setCustomValidity("");
            input.removeClass("is-invalid");
            syncSelect(draft);
          }
        } catch (error) {
          if (this.isCurrentControlChange(input, revision)) {
            const restored = this.restoreCurrentStringControlValue(
              error,
              SETTING_KEYS.arxiv.timezone,
              latestSuccessful,
            );
            latestSuccessful = restored;
            input.value = restored;
            syncSelect(restored);
          }
          throw error;
        }
      });
      let tracked: Promise<void>;
      tracked = operation.finally(() => {
        if (latestSave?.promise === tracked) latestSave = undefined;
      });
      latestSave = { draft, promise: tracked };
      saveQueue = tracked.catch(() => undefined);
      return tracked;
    };
    input.addEventListener("input", () => {
      // Typing a distinct draft supersedes any older save's UI result even
      // before blur/change queues the new transaction.
      this.beginControlChange(input);
      input.setCustomValidity("");
      input.removeClass("is-invalid");
    });
    input.addEventListener("change", () => this.runAction("save timezone", commit));
    input.addEventListener("blur", () => this.runAction("save timezone", commit));
    input.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      this.runAction("save timezone", commit);
    });
  }

  public async saveTickInterval(value: string | number): Promise<number> {
    const interval = Math.max(1, Number(value) || 20);
    await this.plugin.settingsChanges.changeValue(
      "schedule.tickIntervalMin",
      interval,
    );
    return interval;
  }

  public bindTickIntervalInput(input: HTMLInputElement): void {
    let saveQueue = Promise.resolve();
    let latestDraft: string | undefined;
    let latestSave = Promise.resolve();
    const commit = (): Promise<void> => {
      const draft = input.value;
      if (draft === latestDraft) return latestSave;
      latestDraft = draft;
      const revision = this.beginControlChange(input);
      const operation = saveQueue.then(async () => {
        try {
          const next = await this.saveTickInterval(draft);
          if (this.isCurrentControlChange(input, revision)) {
            input.value = String(next);
          }
        } catch (error) {
          if (this.isCurrentControlChange(input, revision)) {
            input.value = String(
              this.restoreCurrentControlValue(error, SETTING_KEYS.schedule.tickIntervalMin),
            );
          }
          throw error;
        }
      }).finally(() => {
        if (latestDraft === draft) latestDraft = undefined;
      });
      saveQueue = operation.catch(() => undefined);
      latestSave = operation;
      return operation;
    };
    input.addEventListener("change", () =>
      this.runAction("save tick interval", commit));
    input.addEventListener("blur", () =>
      this.runAction("save tick interval", commit));
    input.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      this.runAction("save tick interval", commit);
    });
  }

  public async saveLogLevel(value: string): Promise<boolean> {
    if (!isLogLevel(value)) return false;
    await this.plugin.settingsChanges.changeValue("advanced.logLevel", value);
    return true;
  }

  public async saveRunWindowTime(
    key: "runAtLocal" | "runUntilLocal",
    value: string,
  ): Promise<void> {
    try {
      await this.plugin.settingsChanges.changeValue(`schedule.${key}`, value);
    } catch (error) {
      this.reportActionError("save run window", error);
      throw error;
    }
  }

  public async applyOutputDirectoryDraft(
    key: "dailyDir" | "papersDir",
    draft: string,
    input: HTMLInputElement,
  ): Promise<void> {
    const siblingKey = key === "dailyDir" ? "papersDir" : "dailyDir";
    const validation = validateOutputDirectoryDraft(
      draft,
      this.plugin.settings.output[siblingKey],
    );
    input.setCustomValidity(validation.ok ? "" : (validation.reason ?? "Invalid path."));
    input.toggleClass("is-invalid", !validation.ok);
    if (!validation.ok || !validation.value) return;

    const settingKey = `output.${key}`;
    if (validation.value === this.plugin.settings.output[key]) return;
    const revision = this.beginControlChange(input);
    try {
      await this.plugin.settingsChanges.changeValue(settingKey, validation.value);
      if (this.isCurrentControlChange(input, revision)) input.value = validation.value;
    } catch (error) {
      if (this.isCurrentControlChange(input, revision)) {
        input.value = this.restoreCurrentStringControlValue(error, settingKey);
        input.setCustomValidity("");
        input.removeClass("is-invalid");
      }
      this.plugin.logger.error(`settings: rejected ${key}`, error);
      const message = error instanceof Error ? error.message : String(error);
      new Notice(`arXiv Daily: output path was not changed: ${message}`, 10_000);
    }
  }

  display(): void {
    const { containerEl } = this;
    const s = this.plugin.settings;
    containerEl.empty();
    containerEl.addClass("arxiv-daily-settings");

    this.renderSetupGuide(containerEl);

    // ─── Enable toggle (top) ─────────────────────────
    new Setting(containerEl)
      .setName(`Enable · ${s.schedule.enabled ? "Running" : "Paused"}`)
      .setDesc("When on, daily reports run automatically on weekdays (weekends are skipped).")
      .addToggle((t) =>
        t.setValue(s.schedule.enabled).onChange(async (v) => {
          const revision = this.beginControlChange(t);
          try {
            const changed = await this.plugin.setScheduleEnabled(v);
            if (this.isCurrentControlChange(t, revision) && !changed) {
              t.setValue(this.plugin.settings.schedule.enabled);
            }
          } catch (error) {
            if (this.isCurrentControlChange(t, revision)) {
              t.setValue(this.plugin.settings.schedule.enabled);
              this.reportActionError("save schedule enabled", error);
            }
          } finally {
            if (this.isCurrentControlChange(t, revision)) this.renderLegacySettings();
          }
        }),
      );

    // ─── LLM ──────────────────────────────────────────
    this.sectionHeading(containerEl, "LLM", "llm");

    // Base URL — always editable, default to DeepSeek
    new Setting(containerEl)
      .setName("API base URL")
      .setDesc("Where chat requests are sent. Change this only if you use another provider.")
      .addText((t) => {
        t.inputEl.addClass("arxiv-daily-settings__llm-input");
        t.setPlaceholder("Provider URL")
          .setValue(s.llm.baseUrl || "https://api.deepseek.com/v1")
          .onChange(async (v) => {
            const next = v.trim();
            const saved = await this.saveLegacyControl(
              t,
              "save API base URL",
              SETTING_KEYS.llm.baseUrl,
              [{ key: SETTING_KEYS.llm.baseUrl, value: next }],
              (value) => {
                const restored = typeof value === "string" ? value : "";
                t.setValue(restored);
                renderLlmHttpWarning(restored);
              },
            );
            if (saved) this.refreshSetupGuide();
          });
      });
    const llmWarningEl = containerEl.createDiv({
      cls: "arxiv-daily-settings__llm-http-warning",
    });
    const renderLlmHttpWarning = (baseUrl: string) => {
      const warning = llmHttpWarning(baseUrl);
      llmWarningEl.empty();
      llmWarningEl.toggleClass("is-visible", Boolean(warning));
      if (warning) llmWarningEl.setText(warning.message);
    };
    renderLlmHttpWarning(s.llm.baseUrl || "https://api.deepseek.com/v1");

    this.renderApiKeySetting(containerEl);

    // Model — typed name with Get models suggestions (shared with 1.13+)
    const modelSetting = new Setting(containerEl)
      .setName("Model")
      .setDesc("Type a model name, or click get models to see what your provider offers.");
    declarativeRows.renderModelRow(this, modelSetting);

    // Thinking mode — desc varies by provider
    const thinkingDesc = s.llm.provider === "anthropic"
      ? "Let the model spend extra effort on harder questions (Anthropic)."
      : s.llm.provider === "deepseek"
        ? "Let the model spend extra effort on harder questions (DeepSeek reasoning)."
        : "Let the model spend extra effort on harder questions when the provider supports it.";

    let thinkingToggle: { setValue(value: boolean): unknown } | undefined;
    new Setting(containerEl)
      .setName("Thinking mode")
      .setDesc(thinkingDesc)
      .addToggle((t) =>
        (thinkingToggle = t).setValue(s.llm.thinkingMode).onChange(async (v) => {
          await this.saveLegacyControl(
            t,
            "save thinking mode",
            SETTING_KEYS.llm.thinkingMode,
            v
              ? [
                  { key: SETTING_KEYS.llm.thinkingMode, value: true },
                  {
                    key: SETTING_KEYS.llm.reasoningEffort,
                    value: s.llm.reasoningEffort || "medium",
                  },
                ]
              : [{ key: SETTING_KEYS.llm.thinkingMode, value: false }],
            (value) => {
              t.setValue(Boolean(value));
            },
          );
        }),
      );

    // Reasoning effort — provider-specific options + custom input
    const efforts = Object.keys(REASONING_EFFORT_OPTIONS).filter(value => value !== "none");
    new Setting(containerEl)
      .setName("Reasoning effort")
      .setDesc("How hard the model tries when thinking mode is on. Higher may be slower and cost more.")
      .addDropdown((d) => {
        for (const e of efforts) {
          d.addOption(e, e);
        }
        d.setValue(efforts.includes(s.llm.reasoningEffort) ? s.llm.reasoningEffort : efforts[0]!)
          .onChange(async (v) => {
            // Choosing an effort also turns thinking on; keep that toggle in step.
            const saved = await this.saveLegacyControl(
              d,
              "save reasoning effort",
              SETTING_KEYS.llm.reasoningEffort,
              [
                { key: SETTING_KEYS.llm.thinkingMode, value: true },
                { key: SETTING_KEYS.llm.reasoningEffort, value: v },
              ],
              (value) => {
                d.setValue(typeof value === "string" ? value : efforts[0]!);
              },
            );
            if (saved) thinkingToggle?.setValue(this.plugin.settings.llm.thinkingMode);
          });
      })
      .addText((t) => {
        t.setPlaceholder("Or enter custom value")
          .setValue("")
          .onChange(async (v) => {
            const next = v.trim();
            if (!next) return;
            await this.saveLegacyControl(
              t,
              "save custom reasoning effort",
              SETTING_KEYS.llm.reasoningEffort,
              [
                { key: SETTING_KEYS.llm.thinkingMode, value: true },
                { key: SETTING_KEYS.llm.reasoningEffort, value: next },
              ],
              (value) => {
                t.setValue(typeof value === "string" ? value : "");
              },
            );
          });
      });

    // ─── arXiv ────────────────────────────────────────
    this.sectionHeading(containerEl, "arXiv categories", "arxiv");

    const categories = arxivCategories(s.arxiv);
    new Setting(containerEl)
      .setName("Paper categories")
      .setDesc("Which paper subject areas to watch. You can add several; the same paper is only kept once.")
      .setHeading();

    for (let i = 0; i < categories.length; i++) {
      const category = categories[i];
      if (!category) continue;
      new Setting(containerEl)
        .setName(`Category ${i + 1}`)
        .addDropdown((d) => {
          addCategoryOptions(d.selectEl, category);
          d.setValue(category).onChange(async (v) => {
            if (categories.some((other, j) => j !== i && other === v)) {
              new Notice(`arXiv Daily: ${v} is already in the list.`);
              d.setValue(category);
              return;
            }
            const next = [...categories];
            next[i] = v;
            await this.setArxivCategories(next);
            this.renderLegacySettings();
          });
        })
        .addText((t) => {
          t.setPlaceholder("Or enter custom category").setValue("");
          // Commit when editing ends: each keystroke would become a category
          // and the re-render would remove the input being typed in.
          t.inputEl.addEventListener("change", () => {
            const v = t.inputEl.value.trim();
            if (!v) return;
            const next = [...categories];
            next[i] = v;
            this.runAction("save category", async () => {
              await this.setArxivCategories(next);
              this.renderLegacySettings();
            });
          });
        })
        .addButton((b) =>
          b
            .setButtonText("Remove")
            .setDisabled(categories.length === 1)
            .onClick(() => void this.deleteCategory(i)),
        );
    }

    new Setting(containerEl).addButton((b) =>
      b.setButtonText("Add category").onClick(() => void this.addCategory()),
    );

    // ─── Research Topics ─────────────────────────────
    this.sectionHeading(
      containerEl,
      "Research topics",
      "topics",
      "Each topic becomes one section in the daily report.",
    );

    new Setting(containerEl)
      .setName("Quick start")
      .setDesc("Load a preset bundle of topics or add one manually.")
      .addDropdown((d) => {
        d.addOption("", "Load template…");
        for (const tpl of TOPIC_TEMPLATES) {
          d.addOption(tpl.id, tpl.name);
        }
        d.onChange(async (id) => {
          if (!id) return;
          d.setValue("");
          await this.runActionAndWait("apply topic template", () => this.applyTopicTemplate(id));
        });
      })
      .addButton((b) => {
        b.setButtonText("Add topic").onClick(() => this.runAction("add topic", () => this.addTopic()));
      });

    const topicsContainer = containerEl.createDiv();
    if (s.arxiv.topics.length === 0) {
      const empty = topicsContainer.createDiv({
        cls: "arxiv-daily-settings__empty-topics",
      });
      empty.createEl("strong", { text: "No topics yet." });
      empty.createDiv({
        text: "Generate topics from your library, pick a template, or add a topic. Daily reports need at least one topic.",
      });
    }
    for (let i = 0; i < s.arxiv.topics.length; i++) {
      this.renderTopicCard(topicsContainer, s.arxiv.topics, i);
    }

    new Setting(containerEl)
      .setName("Automatic detail notes")
      .setDesc(
        "How often the plugin writes a longer note for a paper. Only topics with detail report turned on are considered. Manual “summarize paper” is unchanged.",
      )
      .addDropdown((d) => {
        addBusinessOptions(d, "detailProfile", { detailProfile: s.detailSelection.profile });
        d.setValue(s.detailSelection.profile).onChange(async (profile) => {
          if (
            profile !== "conservative" &&
            profile !== "balanced" &&
            profile !== "broad"
          ) return;
          const preset = detailSelectionPreset(profile);
          const saved = await this.saveLegacyControl(
            d,
            "save automatic detail notes",
            SETTING_KEYS.detailSelection.profile,
            [
              { key: SETTING_KEYS.detailSelection.profile, value: preset.profile },
              { key: "detailSelection.normalThreshold", value: preset.normalThreshold },
              { key: "detailSelection.exceptionalThreshold", value: preset.exceptionalThreshold },
              { key: "detailSelection.softLimit", value: preset.softLimit },
            ],
            (value) => {
              d.setValue(typeof value === "string" ? value : "balanced");
            },
          );
          if (saved) this.renderLegacySettings();
        });
      });

    new Setting(containerEl)
      .setName("Timezone")
      .addDropdown((d) => {
        for (const zone of TIMEZONE_OPTIONS) {
          d.addOption(zone.value, zone.label);
        }
        d.setValue(s.arxiv.timezone).onChange(async (v) => {
          await this.saveLegacyControl(
            d,
            "save timezone",
            SETTING_KEYS.arxiv.timezone,
            [{ key: SETTING_KEYS.arxiv.timezone, value: v }],
            (value) => {
              d.setValue(typeof value === "string" ? value : "");
            },
          );
        });
      })
      .addText((t) => {
        t.setPlaceholder("Or enter custom timezone").setValue("");
        this.bindTimezoneDraftInput(t.inputEl);
      });

    // ─── Output & Schedule ────────────────────────────
    this.sectionHeading(containerEl, "Output & schedule", "schedule");

    declarativeRows.renderDailyPaperLimitRow(
      this,
      new Setting(containerEl)
        .setName("Daily paper limit")
        .setDesc("Maximum papers across all topics in each daily report. Default is 20."),
    );

    new Setting(containerEl)
      .setName("Daily reports folder")
      .setDesc("Folder in this vault for daily report notes (relative path).")
      .addText((t) => {
        t.setValue(s.output.dailyDir);
        t.inputEl.addEventListener("input", () => {
          const validation = validateOutputDirectoryDraft(t.inputEl.value);
          t.inputEl.setCustomValidity(validation.ok ? "" : (validation.reason ?? "Invalid path."));
          t.inputEl.toggleClass("is-invalid", !validation.ok);
        });
        t.inputEl.addEventListener("change", () => {
          this.runAction("update daily path", () =>
            this.applyOutputDirectoryDraft("dailyDir", t.inputEl.value, t.inputEl));
        });
      });

    new Setting(containerEl)
      .setName("Paper notes folder")
      .setDesc("Folder in this vault for per-paper notes (relative path).")
      .addText((t) => {
        t.setValue(s.output.papersDir);
        t.inputEl.addEventListener("input", () => {
          const validation = validateOutputDirectoryDraft(t.inputEl.value);
          t.inputEl.setCustomValidity(validation.ok ? "" : (validation.reason ?? "Invalid path."));
          t.inputEl.toggleClass("is-invalid", !validation.ok);
        });
        t.inputEl.addEventListener("change", () => {
          this.runAction("update papers path", () =>
            this.applyOutputDirectoryDraft("papersDir", t.inputEl.value, t.inputEl));
        });
      });

    new Setting(containerEl)
      .setName("Link style")
      .setDesc("How links between notes are written in daily reports.")
      .addDropdown((d) =>
        addBusinessOptions(d, "linkStyle")
          .setValue(s.output.linkStyle ?? "wikilink")
          .onChange(async (v) => {
            await this.saveLegacyControl(
              d,
              "save link style",
              SETTING_KEYS.output.linkStyle,
              [{
                key: SETTING_KEYS.output.linkStyle,
                value: v === "relative" ? "relative" : "wikilink",
              }],
              (value) => {
                d.setValue(value === "relative" ? "relative" : "wikilink");
              },
            );
          }),
      );

    new Setting(containerEl)
      .setName("Summary language")
      .setDesc("Language for daily reports and paper notes.")
      .addDropdown((d) =>
        addBusinessOptions(d, "summaryLanguage")
          .setValue(s.output.summaryLanguage ?? "zh")
          .onChange(async (v) => {
            await this.saveLegacyControl(
              d,
              "save summary language",
              SETTING_KEYS.output.summaryLanguage,
              [{
                key: SETTING_KEYS.output.summaryLanguage,
                value: v === "en" ? "en" : "zh",
              }],
              (value) => {
                d.setValue(value === "en" ? "en" : "zh");
              },
            );
          }),
      );

    const runWindow = new Setting(containerEl)
      .setName("Run window")
      .setDesc("Local times when automatic runs may start (24-hour clock).");
    renderRunWindowTimeSelect(
      runWindow.controlEl,
      "Start",
      "arxiv-daily-run-window-start",
      s.schedule.runAtLocal,
      (value) => this.saveRunWindowTime("runAtLocal", value),
    );
    renderRunWindowTimeSelect(
      runWindow.controlEl,
      "End",
      "arxiv-daily-run-window-end",
      s.schedule.runUntilLocal,
      (value) => this.saveRunWindowTime("runUntilLocal", value),
    );

    this.attachHelp(
      new Setting(containerEl).setName("Check every (minutes)").addText((t) => {
        t.setValue(String(s.schedule.tickIntervalMin));
        this.bindTickIntervalInput(t.inputEl);
      }),
      "How often the plugin looks for a day that still needs a report. Default is 20 minutes.",
    );

    // ─── Personal library ─────────────────────────────
    this.sectionHeading(containerEl, "Personal library", "library");
    this.libraryGuide(containerEl, this.libraryGuideContent());
    const librarySetting = new Setting(containerEl)
      .setName("Library")
      .setDesc("Choose a folder of PDFs to prepare its search index automatically. Only suggestions you accept change daily reports.");
    this.renderLibraryConnectionControls(librarySetting);

    if (this.plugin.getLibraryConnectionStatus().kind !== "disconnected") {
      const suggestions = new Setting(containerEl).setName("Topics from library");
      this.renderLibrarySuggestionsControls(suggestions);
    }

    new Setting(containerEl)
      .setName("Embedding")
      .setDesc(
        s.embedding.mode === "remote"
          ? "Remote sends titles and abstracts to an embeddings API. Switching modes rebuilds the index."
          : "Local downloads its model once (about 130 MB) on the first index build, then embeds on this device. Switch to remote only if you have an embeddings API.",
      )
      .addDropdown((d) => {
        addBusinessOptions(d, "embeddingMode");
        d.setValue(s.embedding.mode);
        d.onChange(async (v) => {
          const next = v === "remote" ? "remote" : "local";
          try {
            const changed = await this.applyEmbeddingModeChange(next);
            d.setValue(this.plugin.settings.embedding.mode);
            if (changed) this.renderLegacySettings();
          } catch (error) {
            d.setValue(this.plugin.settings.embedding.mode);
            this.reportActionError("save embedding mode", error);
          }
        });
      });
    if (s.embedding.mode === "remote") {
      new Setting(containerEl)
        .setName("Embedding API base URL")
        .setDesc("OpenAI-compatible embeddings endpoint.")
        .addText((t) => {
          t.setPlaceholder("https://api.openai.com/v1").setValue(s.embedding.baseUrl);
          // On change only: each save may re-ask for remote consent.
          t.inputEl.addEventListener("change", () => {
            void (async () => {
              try {
                t.setValue(await this.saveEmbeddingEndpointField(
                  SETTING_KEYS.embedding.baseUrl,
                  t.inputEl.value.trim(),
                ));
              } catch (error) {
                t.setValue(this.plugin.settings.embedding.baseUrl);
                this.reportActionError("save embedding base url", error);
              }
            })();
          });
        });
      new Setting(containerEl)
        .setName("Embedding API key")
        .setDesc("Saved only on this device.")
        .addText((t) => {
          t.inputEl.type = "password";
          t.setPlaceholder("Enter API key")
            .setValue(s.embedding.apiKey)
            .onChange(async (v) => {
              s.embedding.apiKey = v.trim();
              await this.plugin.saveSettings();
            });
        });
      new Setting(containerEl)
        .setName("Embedding model")
        .setDesc("Model name sent to the endpoint.")
        .addText((t) => {
          t.setPlaceholder("text-embedding-3-small").setValue(s.embedding.model);
          t.inputEl.addEventListener("change", () => {
            void (async () => {
              try {
                t.setValue(await this.saveEmbeddingEndpointField(
                  SETTING_KEYS.embedding.model,
                  t.inputEl.value.trim(),
                ));
              } catch (error) {
                t.setValue(this.plugin.settings.embedding.model);
                this.reportActionError("save embedding model", error);
              }
            })();
          });
        });
      new Setting(containerEl)
        .setName("Embedding dimension")
        .setDesc("Vector width of the remote model. Must match the model.")
        .addText((t) => {
          t.setPlaceholder("1536")
            .setValue(String(s.embedding.dimension))
            .onChange(async (v) => {
              const parsed = Number(v.trim());
              if (Number.isInteger(parsed) && parsed > 0) {
                s.embedding.dimension = parsed;
                await this.plugin.saveSettings();
              }
            });
        });
    }

    new Setting(containerEl)
      .setName("Better PDF parser")
      .setDesc("Optional local sidecar. Off by default; PDFs stay on this device either way.")
      .addToggle((toggle) => {
        toggle.setValue(s.pdfParserSidecar.enabled).onChange(async (enabled) => {
          try {
            await this.changeSettingValue("pdfParserSidecar.enabled", enabled);
            this.renderLegacySettings();
          } catch (error) {
            toggle.setValue(this.plugin.settings.pdfParserSidecar.enabled);
            this.reportActionError("save local parser sidecar", error);
          }
        });
      });
    if (s.pdfParserSidecar.enabled) {
      new Setting(containerEl)
        .setName("Sidecar capability URL")
        .setDesc("Local loopback endpoint that reports parser capabilities.")
        .addText((text) => {
          text.setPlaceholder("HTTP://127.0.0.1:5001/v1/capabilities").setValue(s.pdfParserSidecar.capabilitiesUrl);
          text.inputEl.addEventListener("change", () => {
            this.runAction("save local parser sidecar URL", async () => {
              try {
                await this.changeSettingValues(this.sidecarUrlChanges(
                  "pdfParserSidecar.capabilitiesUrl",
                  text.inputEl.value.trim(),
                ));
                this.renderLegacySettings();
              } catch (error) {
                text.setValue(this.plugin.settings.pdfParserSidecar.capabilitiesUrl);
                throw error;
              }
            });
          });
        });
      new Setting(containerEl)
        .setName("Sidecar parse URL")
        .setDesc("Same-origin local loopback endpoint that accepts one PDF byte buffer.")
        .addText((text) => {
          text.setPlaceholder("HTTP://127.0.0.1:5001/v1/parse").setValue(s.pdfParserSidecar.parseUrl);
          text.inputEl.addEventListener("change", () => {
            this.runAction("save local parser sidecar URL", async () => {
              try {
                await this.changeSettingValues(this.sidecarUrlChanges(
                  "pdfParserSidecar.parseUrl",
                  text.inputEl.value.trim(),
                ));
                this.renderLegacySettings();
              } catch (error) {
                text.setValue(this.plugin.settings.pdfParserSidecar.parseUrl);
                throw error;
              }
            });
          });
        });
    }

    // ─── Email ───────────────────────────────────────────
    this.sectionHeading(containerEl, "Email delivery", "email");

    const hostedMode = s.email.mode === "hosted";

    this.emailGuide(containerEl, this.emailGuideContent());

    new Setting(containerEl)
      .setName("How to send")
      .setDesc(
        hostedMode
          ? "Official delivery (Beta) is a shared free service with a small daily limit. Prefer Send yourself if you need many messages or reliable high volume."
          : "Send yourself uses your own Resend account (no project quota). Official delivery (Beta) is a limited free option for light personal use.",
      )
      .addDropdown((d) => {
        addBusinessOptions(d, "emailMode");
        d.setValue(hostedMode ? "hosted" : "self");
        d.onChange(async (value) => {
          const saved = await this.saveLegacyControl(
            d,
            "save email mode",
            SETTING_KEYS.email.mode,
            [{
              key: SETTING_KEYS.email.mode,
              value: value === "hosted" ? "hosted" : "self",
            }],
            (restored) => {
              d.setValue(restored === "hosted" ? "hosted" : "self");
            },
          );
          if (saved) this.renderLegacySettings();
        });
      });

    new Setting(containerEl)
      .setName("Your email")
      .setDesc(
        hostedMode
          ? "Where verification and daily digests are sent."
          : "Where digests are delivered. With From empty, use the email on your Resend account.",
      )
      .addText((t) => {
        t.setPlaceholder("you@example.com")
          .setValue(s.email.to)
          .onChange(async (v) => {
            await this.saveLegacyControl(
              t,
              "save email address",
              SETTING_KEYS.email.to,
              [{ key: SETTING_KEYS.email.to, value: v.trim() }],
              (value) => {
                t.setValue(typeof value === "string" ? value : "");
              },
            );
          });
      });

    if (hostedMode) {
      new Setting(containerEl)
        .setName("Send verification email")
        .setDesc("Sends a one-time link to confirm this address is yours.")
        .addButton((b) =>
          b.setButtonText("Send verification email").onClick(() => {
            this.runAction("send verification email", async () => {
              const message = await this.plugin.sendHostedVerificationEmail();
              new Notice(message, 10_000);
            });
          }),
        );

      this.renderHostedTokenSetting(containerEl);
    } else {
      this.renderEmailApiKeySetting(containerEl);

      new Setting(containerEl)
        .setName("From email")
        .setDesc(
          "Optional. Leave blank for the simplest setup (mail may only go to your provider account email). Use an address on a verified domain to send more freely.",
        )
        .addText((t) => {
          t.setPlaceholder("Leave blank for simplest setup")
            .setValue(s.email.fromEmail)
            .onChange(async (v) => {
              await this.saveLegacyControl(
                t,
                "save From email",
                SETTING_KEYS.email.fromEmail,
                [{ key: SETTING_KEYS.email.fromEmail, value: v.trim() }],
                (value) => {
                  t.setValue(typeof value === "string" ? value : "");
                },
              );
            });
        });

      new Setting(containerEl)
        .setName("From name")
        .setDesc('Optional name shown as the sender. Leave blank to use the default.')
        .addText((t) => {
          t.setPlaceholder("Sender name")
            .setValue(s.email.fromName ?? "")
            .onChange(async (v) => {
              await this.saveLegacyControl(
                t,
                "save From name",
                SETTING_KEYS.email.fromName,
                [{ key: SETTING_KEYS.email.fromName, value: v }],
                (value) => {
                  t.setValue(typeof value === "string" ? value : "");
                },
              );
            });
        });
    }

    new Setting(containerEl)
      .setName("Send test email")
      .setDesc(
        hostedMode
          ? "Sends a sample digest now. Needs your email and verification code. Tests count toward the daily limit."
          : "Sends a sample digest now. Needs your email and Resend API key.",
      )
      .addButton((b) =>
        b.setButtonText("Send test").setCta().onClick(() => {
          this.runAction("send test email", async () => {
            const message = await this.plugin.sendTestEmail();
            new Notice(message, 10_000);
          });
        }),
      );

    new Setting(containerEl)
      .setName("Daily auto-send")
      .setDesc(dailyAutoSendDesc(hostedMode, this.plugin.automaticEmailSupported()))
      .addToggle((t) =>
        t.setValue(s.email.enabled).onChange(async (v) => {
          await this.saveLegacyControl(
            t,
            "save daily auto-send",
            SETTING_KEYS.email.enabled,
            [{ key: SETTING_KEYS.email.enabled, value: v }],
            (value) => {
              t.setValue(Boolean(value));
            },
          );
        }),
      );

    // ─── Advanced ─────────────────────────────────────
    this.sectionHeading(containerEl, "Advanced", "advanced");

    this.attachHelp(
      new Setting(containerEl).setName("Log level").addDropdown((d) =>
        addBusinessOptions(d, "logLevel")
          .setValue(s.advanced.logLevel)
          .onChange(async (value) => {
            if (!isLogLevel(value)) return;
            await this.saveLegacyControl(
              d,
              "save log level",
              SETTING_KEYS.advanced.logLevel,
              [{ key: SETTING_KEYS.advanced.logLevel, value }],
              (restored) => {
                d.setValue(typeof restored === "string" ? restored : "info");
              },
            );
          }),
      ),
      "How much detail appears in the developer console. Use debug only when troubleshooting; info is the default.",
    );

    // ─── Help & feedback ──────────────────────────────
    this.sectionHeading(
      containerEl,
      "Help & feedback",
      "advanced",
      "Documentation and GitHub issues. A short note is enough; do not paste API keys.",
    );

    new Setting(containerEl)
      .setName("Report a bug")
      .setDesc("Opens a blank GitHub issue with the plugin version. A short description is enough.")
      .addButton((b) =>
        b.setButtonText("Open bug report").onClick(() => {
          this.runAction("open bug report", async () => {
            await this.openExternalUrl(this.bugReportUrl());
          });
        }),
      );

    new Setting(containerEl)
      .setName("Request a feature")
      .setDesc("Opens a blank GitHub issue. Write freely.")
      .addButton((b) =>
        b.setButtonText("Open feature request").onClick(() => {
          this.runAction("open feature request", async () => {
            await this.openExternalUrl(buildFeatureRequestUrl());
          });
        }),
      );

    new Setting(containerEl)
      .setName("Documentation")
      .setDesc("Getting started guide on GitHub.")
      .addButton((b) =>
        b.setButtonText("Open docs").onClick(() => {
          this.runAction("open docs", async () => {
            await this.openExternalUrl(ARXIV_DAILY_DOCS_URL);
          });
        }),
      );

    new Setting(containerEl)
      .setName("Repository")
      .setDesc(ARXIV_DAILY_REPO_URL)
      .addButton((b) =>
        b.setButtonText("Open repository").onClick(() => {
          this.runAction("open repository", async () => {
            await this.openExternalUrl(ARXIV_DAILY_REPO_URL);
          });
        }),
      );
  }

  private bugReportUrl(): string {
    return buildBugReportUrl(this.plugin.manifest.version);
  }

  private async openExternalUrl(url: string): Promise<void> {
    await new ObsidianResourceOpener(this.app).openUrl(url);
  }

  private renderEmailApiKeySetting(containerEl: HTMLElement): void {
    const setting = new Setting(containerEl)
      .setName("Resend API key")
      .setDesc("From your mail provider account. Saved only on this device; masked in the input.");
    renderSensitiveInput(this, setting, {
      value: this.plugin.settings.email.apiKey ?? "",
      placeholder: "Paste your Resend API key",
      ariaLabel: "Resend API key",
      save: (next) => this.changeSettingValue("email.apiKey", next),
    });
  }
  private renderHostedTokenSetting(containerEl: HTMLElement): void {
    const setting = new Setting(containerEl)
      .setName("Verification code")
      .setDesc(
        "After you open the verification link, copy the long code shown on the web page (not the short code in the email link). Use the same email address as above.",
      );
    renderSensitiveInput(this, setting, {
      value: this.plugin.settings.email.hostedToken ?? "",
      placeholder: "Paste the code from the verification page",
      ariaLabel: "verification code",
      normalize: (value) => value.replace(/\s+/g, "").trim(),
      save: (next) => this.changeSettingValue("email.hostedToken", next),
    });
  }
  private renderApiKeySetting(containerEl: HTMLElement): void {
    const setting = new Setting(containerEl)
      .setName("API key")
      .setDesc("Saved only on this device; masked in the input.");
    renderSensitiveInput(this, setting, {
      value: this.plugin.settings.llm.apiKey,
      placeholder: "Enter API key",
      ariaLabel: "LLM API key",
      save: (next) => this.changeSettingValue("llm.apiKey", next),
    });
  }
  private renderSetupGuide(containerEl: HTMLElement): void {
    const guide = this.createSetupGuide();
    if (guide) containerEl.appendChild(guide);
  }

  /**
   * Whether the setup-guide row belongs in the settings page at all: always
   * true before the guide has ever completed (covers both the step-by-step
   * list and the one-time completion summary), and after that only true if
   * something is now broken and needs a compact warning (see
   * `createConfigurationWarning`).
   */
  public shouldShowSetupGuide(): boolean {
    const status = getSetupStatus(
      this.plugin.settings,
      this.plugin.stateStore.snapshot(),
    );
    if (!this.plugin.settings.onboarding.guideCompleted) return true;
    return !status.readyToRun || status.schedulerReasons.length > 0;
  }

  /** Remember the host row so the guide can update without replacing active inputs. */
  public setDeclarativeSetupGuideRow(setting: Setting): void {
    this.declarativeSetupGuideRow = setting;
  }

  public refreshDeclarativeSetupGuide(): void {
    const setting = this.declarativeSetupGuideRow;
    if (setting?.settingEl.isConnected) {
      declarativeRows.renderSetupGuideRow(this, setting);
      return;
    }
    // Only a guide that has to reappear needs the full update; re-rendering
    // for every keystroke in a topic card would replace the focused input.
    if (this.shouldShowSetupGuide()) this.refreshSettings();
  }

  public refreshSetupGuide(): void {
    if (requireApiVersion("1.13.0")) {
      this.refreshDeclarativeSetupGuide();
      return;
    }
    const current = this.containerEl.querySelector(".arxiv-daily-setup");
    const next = this.createSetupGuide();
    if (current instanceof HTMLElement) {
      if (next) {
        current.replaceWith(next);
      } else {
        current.remove();
      }
    } else if (next) {
      this.containerEl.prepend(next);
    }
  }

  /**
   * Set the persisted "guide completed" marker the moment every milestone is
   * true at once, and flush it to disk. A failed save is reported the same
   * way other in-place field edits are (the draft/marker is kept locally).
   */
  private persistSetupGuideCompletion(status: SetupStatus): boolean {
    if (!markSetupGuideCompleteIfDone(this.plugin.settings, status)) return false;
    void this.plugin.saveSettings().catch((error) => {
      this.reportActionError("save setup guide completion", error);
    });
    return true;
  }

  /**
   * Once the guide is retired, an already-onboarded user who later breaks
   * their configuration (or a scheduler setting) would otherwise get no
   * explanation anywhere on the settings page. Reuse the same compact
   * details disclosure the guide used, standalone, only while something is
   * actually wrong.
   */
  private createConfigurationWarning(status: SetupStatus): HTMLElement | null {
    if (status.readyToRun && status.schedulerReasons.length === 0) return null;
    const warning = this.containerEl.createEl("section", {
      cls: "arxiv-daily-setup",
      attr: { "aria-labelledby": "arxiv-daily-setup-title" },
    });
    warning.detach();
    warning.createDiv({
      cls: "arxiv-daily-setup__title",
      text: "Configuration needs attention",
      attr: {
        id: "arxiv-daily-setup-title",
        role: "heading",
        "aria-level": "2",
      },
    });
    this.renderConfigurationDetails(warning, [...status.reasons, ...status.schedulerReasons]);
    return warning;
  }

  public createSetupGuide(): HTMLElement | null {
    const status = getSetupStatus(
      this.plugin.settings,
      this.plugin.stateStore.snapshot(),
    );

    // Once the guide has completed once, it is retired for good; only a
    // compact warning (if anything is broken) may still appear.
    if (this.plugin.settings.onboarding.guideCompleted) {
      return this.createConfigurationWarning(status);
    }

    const guide = this.containerEl.createEl("section", {
      cls: "arxiv-daily-setup",
      attr: { "aria-labelledby": "arxiv-daily-setup-title" },
    });
    guide.detach();

    if (isSetupComplete(status)) {
      this.persistSetupGuideCompletion(status);
      guide.addClass("arxiv-daily-setup--complete");
      const summary = guide.createDiv({
        cls: "arxiv-daily-setup__complete-summary",
      });
      summary.createDiv({
        cls: "arxiv-daily-setup__title",
        text: "Setup complete",
        attr: {
          id: "arxiv-daily-setup-title",
          role: "heading",
          "aria-level": "2",
        },
      });
      summary.createDiv({
        cls: "arxiv-daily-setup__complete-date",
        text: `Latest completed report: ${status.latestCompletedReportDate ?? "Unknown"}`,
      });
      this.renderConfigurationDetails(guide, status.schedulerReasons);
      this.renderDashboardAction(guide);
      return guide;
    }

    const header = guide.createDiv({
      cls: "arxiv-daily-setup__header",
    });
    header.createDiv({
      cls: "arxiv-daily-setup__title",
      text: "Getting started",
      attr: {
        id: "arxiv-daily-setup-title",
        role: "heading",
        "aria-level": "2",
      },
    });
    const completedCount = [
      status.llmReady,
      status.categoriesReady,
      status.topicsReady,
      status.firstReportComplete,
      status.scheduleEnabled,
    ].filter(Boolean).length;
    header.createDiv({
      cls: "arxiv-daily-setup__progress-summary",
      text: `${completedCount} of 5 complete`,
      attr: { "aria-live": "polite" },
    });
    const progress = guide.createEl("progress", {
      cls: "arxiv-daily-setup__progress",
      attr: {
        max: "5",
        value: String(completedCount),
        "aria-label": "Setup progress",
      },
    });
    progress.setAttribute("value", String(completedCount));

    const list = guide.createEl("ol", {
      cls: "arxiv-daily-setup__list",
    });
    this.renderSetupItem(
      list,
      status.llmReady,
      "Connect AI",
      "Add an API key, API base URL, and model under LLM.",
      "Connect AI",
      () => this.scrollToSection("llm"),
    );
    this.renderSetupItem(
      list,
      status.categoriesReady,
      "Choose paper sources",
      "Select at least one arXiv category under arXiv categories.",
      "Choose sources",
      () => this.scrollToSection("arxiv"),
    );
    this.renderSetupItem(
      list,
      status.topicsReady,
      "Describe your research interests",
      "Use your library to suggest topics, or add your own under Research topics.",
      "Describe interests",
      () => this.scrollToSection("topics"),
    );
    this.renderSetupItem(
      list,
      status.firstReportComplete,
      "Generate your first report",
      status.readyToRun
        ? "Your configuration is ready. Generate a report to finish setup."
        : status.llmReady && status.categoriesReady && status.topicsReady
          ? `Fix before generating: ${status.reasons.join("; ")}.`
          : "Complete the earlier configuration steps before generating a report.",
      !status.readyToRun
        ? undefined
        : this.firstReportRunning
          ? "Generating…"
          : "Generate first report",
      status.readyToRun
        ? () => {
            this.runAction("generate first report", () => this.generateFirstReport());
          }
        : undefined,
      this.firstReportRunning,
    );
    this.renderSetupItem(
      list,
      status.scheduleEnabled,
      "Turn on daily reports",
      status.readyToRun
        ? "Reports then run by themselves on weekdays, inside the run window below."
        : "Available once the configuration above is complete.",
      status.readyToRun ? "Turn on daily reports" : undefined,
      status.readyToRun
        ? () => {
            this.runAction("turn on daily reports", () => this.enableDailyReports());
          }
        : undefined,
    );

    this.renderConfigurationDetails(guide, status.schedulerReasons);
    this.renderDashboardAction(guide);
    return guide;
  }

  private renderSetupItem(
    parent: HTMLElement,
    done: boolean,
    title: string,
    description: string,
    actionLabel?: string,
    onAction?: () => void,
    busy = false,
  ): void {
    const current = !done && !parent.querySelector(".is-current");
    const item = parent.createEl("li", {
      cls: `arxiv-daily-setup__item ${done ? "is-done" : "is-pending"}`,
    });
    if (current) {
      item.addClass("is-current");
      item.setAttribute("aria-current", "step");
    }
    const body = item.createDiv({ cls: "arxiv-daily-setup__item-body" });
    body.createDiv({
      cls: "arxiv-daily-setup__label",
      text: title,
    });
    if (current) {
      body.createDiv({
        cls: "arxiv-daily-setup__description",
        text: description,
      });
    }
    if (done || current) {
      item.createSpan({
        cls: "arxiv-daily-setup__status",
        text: done ? "Complete" : "Next",
      });
    }
    if (current && actionLabel && onAction) {
      const action = item.createEl("button", {
        cls: "arxiv-daily-setup__link",
        text: actionLabel,
        attr: { type: "button" },
      });
      if (busy) {
        action.disabled = true;
        action.setAttribute("aria-busy", "true");
      }
      action.addEventListener("click", onAction);
    }
  }

  private renderConfigurationDetails(
    parent: HTMLElement,
    reasons: readonly string[],
  ): void {
    if (reasons.length === 0) return;
    const details = parent.createEl("details", {
      cls: "arxiv-daily-setup__details",
    });
    details.createEl("summary", { text: "Configuration details" });
    const list = details.createEl("ul");
    for (const reason of reasons) list.createEl("li", { text: reason });
  }

  private renderDashboardAction(parent: HTMLElement): void {
    const actions = parent.createDiv({
      cls: "arxiv-daily-setup__actions",
    });
    const dashboard = actions.createEl("button", {
      text: "Open dashboard",
      attr: { type: "button" },
    });
    dashboard.addEventListener("click", () => {
      this.runAction("open dashboard", () => openDashboardView(this.plugin));
    });
  }

  /**
   * Bring a guide step's section into view and put the cursor where the step
   * still needs input. The section is the 1.13+ group element or, in
   * display(), its heading row; both carry settingsSectionClass().
   */
  private scrollToSection(section: "llm" | "arxiv" | "topics"): void {
    const target = this.containerEl.querySelector<HTMLElement>(
      `.${settingsSectionClass(section)}`,
    );
    if (!target) {
      const heading = { llm: "LLM", arxiv: "arXiv categories", topics: "Research topics" }[section];
      this.plugin.logger.warn(`settings: setup guide could not find the ${section} section`);
      new Notice(`arXiv Daily: scroll down to "${heading}" to continue setup.`);
      return;
    }
    const view = target.ownerDocument.defaultView;
    const reduceMotion = view?.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    target.scrollIntoView({
      block: "start",
      behavior: reduceMotion ? "auto" : "smooth",
    });
    if (section === "topics") {
      if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
      target.focus({ preventScroll: true });
      this.focusIncompleteTopic();
      return;
    }
    const field = this.firstPendingField(section);
    if (field) {
      field.focus({ preventScroll: true });
      return;
    }
    if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
    target.focus({ preventScroll: true });
  }

  private firstPendingField(section: "llm" | "arxiv"): HTMLElement | null {
    const query = (selector: string) =>
      this.containerEl.querySelector<HTMLElement>(selector);
    if (section === "arxiv") {
      return query(".arxiv-daily-settings__category-select");
    }
    const { llm } = this.plugin.settings;
    if (!llm.apiKey.trim()) {
      const apiKey = query('input[aria-label="LLM API key"]');
      if (apiKey) return apiKey;
    }
    if (!llm.baseUrl.trim()) {
      const baseUrl = query(".arxiv-daily-settings__llm-url-input");
      if (baseUrl) return baseUrl;
    }
    return query(".arxiv-daily-settings__model-input");
  }

  /** Open the first topic missing a field and focus that field; add one if there are none. */
  private focusIncompleteTopic(): void {
    const topics = this.plugin.settings.arxiv.topics;
    if (topics.length === 0) {
      this.runAction("add topic", () => this.addTopic());
      return;
    }
    const topic = topics.find(
      (candidate) =>
        !candidate.name.trim() ||
        !candidate.tag.trim() ||
        !candidate.directions.some((direction) => direction.text.trim()),
    ) ?? topics[0]!;
    const card = this.findTopicCard(topic.id);
    if (!card) return;
    const form = card.querySelector<HTMLElement>(".arxiv-daily-settings__topic-form");
    if (form?.hidden) {
      card.querySelector<HTMLElement>(".arxiv-daily-settings__topic-header")?.click();
    }
    const direction = Array.from(card.querySelectorAll<HTMLTextAreaElement>(
      ".arxiv-daily-settings__topic-direction-input",
    )).find((input) => !input.value.trim());
    const field = !topic.name.trim()
      ? card.querySelector<HTMLElement>(".arxiv-daily-settings__topic-name-input")
      : direction ?? card.querySelector<HTMLElement>(".arxiv-daily-settings__topic-direction-add");
    field?.focus({ preventScroll: true });
  }

  /**
   * The first report should succeed whatever the day: weekends and the hours
   * before arXiv's announcement have no listing for today, so use the latest
   * announced day that is not after today. Falls back to today when the
   * announced days cannot be loaded.
   */
  private async latestAnnouncedDate(today: string): Promise<string> {
    try {
      await this.plugin.recentDates.refresh();
    } catch (error) {
      this.plugin.logger.warn("settings: could not load announced arXiv days", error);
      return today;
    }
    const announced = Array.from(this.plugin.recentDates.snapshot().dates)
      .filter((date) => date <= today)
      .sort();
    return announced.at(-1) ?? today;
  }

  /** Guide step 5: the same path as the Enable toggle (validation + run-today choice). */
  public async enableDailyReports(): Promise<void> {
    await this.plugin.setScheduleEnabled(true);
    this.refreshSettings();
  }

  public async generateFirstReport(): Promise<void> {
    if (this.firstReportRunning) return;
    this.firstReportRunning = true;
    this.refreshSetupGuide();
    try {
      const today = formatDate(
        todayInTz(new Date(), this.plugin.settings.arxiv.timezone),
      );
      const date = await this.latestAnnouncedDate(today);
      this.plugin.logger.info(`settings: first report requested for ${date}`);
      new Notice(
        date === today
          ? `arXiv Daily: running for ${date}…`
          : `arXiv Daily: today's papers are not announced yet, so the first report uses the latest announced day, ${date}…`,
      );
      const result = await this.plugin.scheduler.runForDateNow(date);
      new Notice(`arXiv Daily ${date}: ${describeResult(result)}`);
      await refreshOpenDashboardViews(this.plugin).catch((error: unknown) => {
        this.plugin.logger.warn("settings: dashboard refresh after first report failed", error);
      });
    } finally {
      this.firstReportRunning = false;
      this.refreshSetupGuide();
    }
  }

  /** Reveal and focus a topic created by Add topic after the settings list updates. */
  private focusPendingTopic(): void {
    const topicId = this.pendingTopicFocusId;
    if (!topicId) return;

    const focus = () => {
      const card = Array.from(
        this.containerEl.querySelectorAll<HTMLElement>(
          ".arxiv-daily-settings__topic-card",
        ),
      ).find((candidate) => candidate.dataset.arxivDailyTopicId === topicId);
      if (!card) return;
      this.pendingTopicFocusId = undefined;

      card.scrollIntoView?.({
        block: "nearest",
        behavior: "auto",
      });
      const nameInput = card.querySelector<HTMLInputElement>(
        ".arxiv-daily-settings__topic-name-input",
      );
      nameInput?.focus({ preventScroll: true });
    };

    queueMicrotask(() => {
      if (this.pendingTopicFocusId !== topicId) return;
      if (this.containerEl.querySelector(
        `[data-arxiv-daily-topic-id="${topicId}"]`,
      )) {
        focus();
        return;
      }
      const view = this.containerEl.ownerDocument.defaultView;
      if (view?.requestAnimationFrame) view.requestAnimationFrame(focus);
      else if (view?.setTimeout) view.setTimeout(focus, 0);
      else queueMicrotask(focus);
    });
  }

  /** Keep a surviving neighbor visually fixed when a topic card is removed. */
  private captureTopicDeletionAnchor(topicId?: string):
    | { topicId: string; scroller: HTMLElement; top: number }
    | undefined {
    if (!topicId) return undefined;
    const card = this.findTopicCard(topicId);
    if (!card) return undefined;
    const scroller = this.captureSettingsScroll().find(({ element }) =>
      element.contains(card),
    )?.element;
    if (!scroller) return undefined;
    return { topicId, scroller, top: card.getBoundingClientRect().top };
  }

  private restoreTopicDeletionAnchor(): void {
    const anchor = this.pendingTopicDeletionAnchor;
    if (!anchor) return;
    const restore = (): boolean => {
      const card = this.findTopicCard(anchor.topicId);
      if (!card) return false;
      this.pendingTopicDeletionAnchor = undefined;
      anchor.scroller.scrollTop += card.getBoundingClientRect().top - anchor.top;
      return true;
    };
    if (restore()) return;
    queueMicrotask(() => {
      if (restore()) return;
      const view = this.containerEl.ownerDocument.defaultView;
      if (view?.requestAnimationFrame) view.requestAnimationFrame(() => {
        restore();
      });
    });
  }

  private findTopicCard(topicId: string): HTMLElement | undefined {
    return Array.from(
      this.containerEl.querySelectorAll<HTMLElement>(
        ".arxiv-daily-settings__topic-card",
      ),
    ).find((candidate) => candidate.dataset.arxivDailyTopicId === topicId);
  }

  /** Render the topic card for one index into a declarative list row. */
  public renderTopicRow(setting: Setting, index: number): void {
    // Re-renders reuse the same row; drop the previous card first.
    for (const el of Array.from(
      setting.settingEl.querySelectorAll(".arxiv-daily-settings__topic-card"),
    )) {
      el.remove();
    }
    // Scopes the 1.13+ card layout (full-width row, aligned header, grid
    // form) without touching the <1.13 display() styling.
    setting.settingEl.addClass("arxiv-daily-settings__topic-host");
    this.renderTopicCard(
      setting.settingEl,
      this.plugin.settings.arxiv.topics,
      index,
      true,
    );
  }

  /** Apply only this edit to the latest queued topic, preserving other changes. */
  private async changeTopic(
    topicId: string,
    edit: (topic: Topic, topics: Topic[], index: number) => void,
  ): Promise<void> {
    await this.plugin.settingsChanges.changeComputed((current) => {
      const topics = current.arxiv.topics;
      const index = topics.findIndex(({ id }) => id === topicId);
      const topic = topics[index];
      if (!topic) throw new Error("This topic no longer exists. Reopen settings to refresh the list.");
      edit(topic, topics, index);
      // Empty rows remain editable drafts, as in the existing editor. Use the
      // shared shadow rule without normalizing those rows out while typing.
      topic.description = deriveTopicDescription(topic.directions);
      return { changes: [{ key: "arxiv.topics", value: topics }] };
    });
  }

  private renderTopicCard(
    container: HTMLElement,
    topics: Topic[],
    index: number,
    compact = false,
  ): void {
    const storedTopic = topics[index];
    if (!storedTopic) return;
    // Renderers never own live settings objects: another queued transaction
    // can replace their contents while the user continues typing here.
    const topic: Topic = {
      ...storedTopic,
      directions: storedTopic.directions.map((direction) => ({ ...direction })),
    };
    const createdDirections = new Set<string>();
    const liveTopic = () => this.plugin.settings.arxiv.topics.find(({ id }) => id === topic.id);
    const persistEdit = (
      control: object,
      action: string,
      edit: (current: Topic, topics: Topic[], index: number) => void,
      restore: () => void,
    ): Promise<void> => {
      const revision = this.beginControlChange(control);
      const saving = (async () => {
        try {
          await this.changeTopic(topic.id, edit);
          if (this.isCurrentControlChange(control, revision)) this.refreshSetupGuide();
        } catch (error) {
          if (this.isCurrentControlChange(control, revision)) restore();
          this.reportActionError(action, error);
        }
      })();
      this.pendingTopicEdits.add(saving);
      const settled = () => { this.pendingTopicEdits.delete(saving); };
      void saving.then(settled, settled);
      return saving;
    };
    const isExpanded = this.expandedTopics.has(topic.id);
    const idPrefix = `arxiv-daily-topic-${stableDomId(topic.id)}`;
    const formId = `${idPrefix}-form`;

    const card = container.createDiv({
      cls: "arxiv-daily-settings__topic-card",
      attr: { "data-arxiv-daily-topic-id": topic.id },
    });

    // ─── Header row (always visible, clickable) ────────────
    const header = card.createEl("button", {
      cls: "arxiv-daily-settings__topic-header",
      attr: {
        type: "button",
        "aria-expanded": String(isExpanded),
        "aria-controls": formId,
      },
    });

    const caret = header.createSpan({
      cls: "arxiv-daily-settings__topic-caret",
      text: isExpanded ? "▾" : "▸",
    });

    const titleSpan = header.createSpan({
      cls: "arxiv-daily-settings__topic-title",
      text: topic.name.trim() || "(unnamed)",
      attr: { title: topic.name },
    });
    titleSpan.toggleClass("is-muted", !topic.name.trim());

    // The header carries the name alone: a topic name and its machine tag
    // together outran the row, and the tag is derived from the name anyway.
    let star: HTMLElement | null = null;
    if (topic.detail) {
      star = header.createSpan({
        cls: "arxiv-daily-settings__topic-star",
        text: "★",
        attr: { title: "Detail report enabled" },
      });
    }

    // ─── Expanded form (toggled via display) ────────────────
    const form = card.createDiv({
      cls: "arxiv-daily-settings__topic-form",
    });
    form.id = formId;
    form.hidden = !isExpanded;
    form.toggleClass("is-collapsed", !isExpanded);

    // Name row
    const nameRow = form.createDiv({
      cls: "arxiv-daily-settings__topic-row",
    });
    const nameId = `${idPrefix}-name`;
    const nameHintId = `${nameId}-hint`;
    nameRow.createEl("label", {
      cls: "arxiv-daily-settings__topic-label",
      text: getTopicSettingField("name").name,
      attr: { for: nameId },
    });
    if (!compact) {
      this.hint(nameRow, getTopicSettingField("name").description, nameHintId);
    }
    const nameInput = nameRow.createEl("input", {
      cls: "arxiv-daily-settings__topic-name-input",
      type: "text",
      attr: compact
        ? { id: nameId }
        : { id: nameId, "aria-describedby": nameHintId },
    });
    nameInput.value = topic.name;
    nameInput.placeholder = getTopicSettingField("name").placeholder;

    const refreshHeader = () => {
      titleSpan.textContent = topic.name.trim() || "(unnamed)";
      titleSpan.title = topic.name;
      titleSpan.toggleClass("is-muted", !topic.name.trim());
    };

    // Settings no longer shows the tag, so the name is the only thing left
    // that can produce one. A tag that is still the machine form of the old
    // name follows the rename; one that was typed by hand while the field
    // existed is left alone, because nothing on screen could restore it.
    nameInput.oninput = async () => {
      const name = nameInput.value;
      topic.name = name;
      refreshHeader();
      await persistEdit(nameInput, "rename topic", (current, currentTopics, currentIndex) => {
        const tagFollowsName = isDerivedTopicTag(current.tag, current.name)
          || isPlaceholderTopicTag(current.tag) || !current.tag;
        current.name = name;
        if (tagFollowsName) {
          current.tag = uniqueTopicTag(currentTopics, currentIndex, slugify(name) || `topic-${currentIndex + 1}`);
        }
      }, () => {
        topic.name = liveTopic()?.name ?? storedTopic.name;
        nameInput.value = topic.name;
        refreshHeader();
      });
    };

    // Directions
    // The row always holds the directions list created just below (it is
    // only ever emptied/repopulated, never removed), so the CSS that makes
    // this row span the grid can key off a static class here instead of a
    // `:has()` selector.
    const dirRow = form.createDiv({
      cls: ["arxiv-daily-settings__topic-row", "arxiv-daily-settings__topic-row--has-directions"],
    });
    const dirId = `${idPrefix}-directions`;
    const dirHintId = `${dirId}-hint`;
    dirRow.createEl("label", {
      cls: "arxiv-daily-settings__topic-label",
      text: "Directions",
      attr: { for: dirId },
    });
    if (!compact) {
      this.hint(dirRow, "One line per specific thread you follow inside this topic. The AI matches papers against these.", dirHintId);
    }
    const dirList = dirRow.createDiv({
      cls: "arxiv-daily-settings__topic-directions",
      attr: compact
        ? { id: dirId }
        : { id: dirId, "aria-describedby": dirHintId },
    });

    // Re-checked when the card expands: a hidden field measures as zero, so
    // the truncation marker cannot be decided until it is on screen.
    const directionOverflowChecks: Array<() => void> = [];

    const appendDirection = async () => {
      const direction: Topic["directions"][number] = {
        id: crypto.randomUUID(),
        text: "",
        origin: "manual",
      };
      createdDirections.add(direction.id);
      topic.directions.push(direction);
      renderDirections();
      dirList
        .querySelectorAll<HTMLTextAreaElement>(
          ".arxiv-daily-settings__topic-direction-input",
        )
        .item(topic.directions.length - 1)
        ?.focus();
      await persistEdit(direction, "add direction", (current) => {
        current.directions.push({ id: direction.id, text: "", origin: "manual" });
      }, () => {
        topic.directions = topic.directions.filter(({ id }) => id !== direction.id);
        createdDirections.delete(direction.id);
        renderDirections();
      });
    };

    const renderDirections = () => {
      dirList.empty();
      directionOverflowChecks.length = 0;
      topic.directions.forEach((direction, directionIndex) => {
        const line = dirList.createDiv({
          cls: "arxiv-daily-settings__topic-direction",
        });
        // The field grows with its content by replicating the text into a
        // hidden ::after in the same grid cell, so no height is ever assigned
        // from script and a field hidden inside a collapsed card still sizes
        // itself correctly the moment it is shown.
        const field = line.createDiv({
          cls: "arxiv-daily-settings__topic-direction-field is-collapsed",
        });
        const dirInput = field.createEl("textarea", {
          cls: "arxiv-daily-settings__topic-direction-input",
        });
        dirInput.dataset.directionId = direction.id;
        dirInput.value = direction.text;
        dirInput.rows = 1;
        dirInput.placeholder = "One specific direction";
        // A textarea cannot render a marker over its own text, so the "there
        // is more" badge is a sibling, shown only when the collapsed field is
        // really cut off. It counts lines because that is what can be measured
        // exactly; the title spells the number out.
        const more = field.createSpan({
          cls: "arxiv-daily-settings__topic-direction-more",
        });

        const syncField = () => {
          field.dataset.replicatedValue = dirInput.value;
          const collapsed = field.classList.contains("is-collapsed");
          const lineHeight = Number.parseFloat(
            getComputedStyle(dirInput).lineHeight,
          );
          const hidden = collapsed && Number.isFinite(lineHeight) && lineHeight > 0
            ? Math.round((field.scrollHeight - field.clientHeight) / lineHeight)
            : 0;
          more.toggleClass("is-visible", hidden > 0);
          more.setText(hidden > 0 ? `+${hidden}` : "");
          more.title = hidden === 1 ? "1 more line" : `${hidden} more lines`;
        };
        directionOverflowChecks.push(syncField);
        syncField();

        dirInput.oninput = async () => {
          // The filter prompt joins topics with "\n", so a pasted newline
          // would break that line structure. Keep the stored value one line.
          const flattened = dirInput.value.replace(/\s*\n+\s*/g, " ");
          if (flattened !== dirInput.value) dirInput.value = flattened;
          direction.text = flattened;
          syncField();
          await persistEdit(direction, "edit direction", (current) => {
            const existing = current.directions.find(({ id }) => id === direction.id);
            if (existing) existing.text = flattened;
            else if (createdDirections.has(direction.id)) {
              // The blank-row save may have failed ahead of this newer text.
              // Keep the explicit Add action recoverable through its stable id.
              current.directions.push({ id: direction.id, text: flattened, origin: "manual" });
            } else {
              throw new Error("This direction no longer exists. Reopen settings to refresh the list.");
            }
          }, () => {
            const saved = liveTopic()?.directions.find(({ id }) => id === direction.id);
            if (!saved) {
              topic.directions = topic.directions.filter(({ id }) => id !== direction.id);
              renderDirections();
              return;
            }
            direction.text = saved.text;
            const visibleInput = Array.from(dirList.querySelectorAll<HTMLTextAreaElement>("textarea"))
              .find((input) => input.dataset.directionId === direction.id);
            if (visibleInput) visibleInput.value = direction.text;
            for (const check of directionOverflowChecks) check();
          });
        };
        dirInput.onfocus = () => {
          field.removeClass("is-collapsed");
          syncField();
        };
        dirInput.onblur = () => {
          field.addClass("is-collapsed");
          syncField();
        };
        dirInput.onkeydown = (event: KeyboardEvent) => {
          // A direction is one line of text: the filter prompt joins topics
          // with newlines, so one stored here would break that structure.
          // Enter cannot insert one, so it confirms instead — the field
          // collapses back and gives up focus. Adding a direction stays the
          // Add button's job.
          if (event.key !== "Enter") return;
          event.preventDefault();
          dirInput.blur();
        };

        const removeBtn = line.createEl("button", {
          cls: "arxiv-daily-settings__topic-direction-remove",
          text: "×",
          attr: {
            type: "button",
            "aria-label": `Remove direction ${directionIndex + 1}`,
          },
        });
        removeBtn.onclick = async () => {
          topic.directions = topic.directions.filter(({ id }) => id !== direction.id);
          renderDirections();
          await persistEdit(direction, "remove direction", (current) => {
            current.directions = current.directions.filter(({ id }) => id !== direction.id);
          }, () => {
            const saved = liveTopic()?.directions.find(({ id }) => id === direction.id);
            if (saved && !topic.directions.some(({ id }) => id === direction.id)) {
              topic.directions.splice(Math.min(directionIndex, topic.directions.length), 0, { ...saved });
              renderDirections();
            }
          });
        };
      });
      const addBtn = dirList.createEl("button", {
        cls: "arxiv-daily-settings__topic-direction-add",
        text: "Add direction",
        attr: { type: "button" },
      });
      addBtn.onclick = () => void appendDirection();
    };
    renderDirections();

    // Detail toggle + delete (right-aligned, only visible when expanded)
    if (!compact) {
      this.hint(form, "Detail report = generate a full, deep-dive markdown file for primary contributions to this topic. Delete = remove this topic.");
    }
    const footer = form.createDiv({
      cls: "arxiv-daily-settings__topic-footer",
    });

    const detailLabel = footer.createEl("label", {
      cls: "arxiv-daily-settings__topic-detail-label",
    });
    const detailCheckbox = detailLabel.createEl("input", { type: "checkbox" });
    detailCheckbox.checked = topic.detail;
    detailCheckbox.addClass("arxiv-daily-settings__topic-detail-checkbox");
    detailLabel.appendText(getTopicSettingField("detail").name);
    const refreshDetail = () => {
      // Refresh the header star indicator without a full re-render.
      star?.remove();
      star = null;
      if (topic.detail) {
        star = header.createSpan({
          cls: "arxiv-daily-settings__topic-star",
          text: "★",
          attr: { title: "Detail report enabled" },
        });
      }
    };
    detailCheckbox.onchange = async () => {
      const detail = detailCheckbox.checked;
      topic.detail = detail;
      refreshDetail();
      await persistEdit(detailCheckbox, "change detail report", (current) => {
        current.detail = detail;
      }, () => {
        topic.detail = liveTopic()?.detail ?? storedTopic.detail;
        detailCheckbox.checked = topic.detail;
        refreshDetail();
      });
    };

    const delBtn = footer.createEl("button", {
      text: "Delete",
      attr: { type: "button" },
    });
    delBtn.classList.add("mod-warning");
    delBtn.onclick = async (e) => {
      e.stopPropagation();
      await this.runActionAndWait("delete topic", () => this.deleteTopic(topic.id));
    };

    // Toggle expand/collapse on header click
    header.onclick = () => {
      const expanded = !this.expandedTopics.has(topic.id);
      if (expanded) this.expandedTopics.add(topic.id);
      else this.expandedTopics.delete(topic.id);
      form.hidden = !expanded;
      form.toggleClass("is-collapsed", !expanded);
      header.setAttribute("aria-expanded", String(expanded));
      caret.textContent = expanded ? "▾" : "▸";
      // Directions could not be measured while the form was hidden.
      if (expanded) for (const check of directionOverflowChecks) check();
    };
  }

  public confirmReplace(message: string, confirmLabel = "Replace"): Promise<boolean> {
    return new Promise((resolve) => {
      const modal = new Modal(this.app);
      modal.titleEl.setText("Confirm");
      modal.contentEl.createEl("p", { text: message });
      const btns = modal.contentEl.createDiv({
        cls: "arxiv-daily-modal-button-row",
      });
      const cancel = btns.createEl("button", { text: "Cancel" });
      const ok = btns.createEl("button", { text: confirmLabel });
      ok.classList.add("mod-warning");
      let settled = false;
      const finish = (value: boolean) => {
        if (settled) return;
        settled = true;
        resolve(value);
        modal.close();
      };
      cancel.onclick = () => finish(false);
      ok.onclick = () => finish(true);
      modal.onClose = () => finish(false);
      modal.open();
    });
  }

  private textareaSetting(
    container: HTMLElement,
    name: string,
    desc: string,
    value: string,
    onChange: (v: string) => Promise<void>,
  ): Setting {
    return new Setting(container)
      .setName(name)
      .setDesc(desc)
      .addTextArea((t) => {
        t.setValue(value).onChange((v) => onChange(v));
        t.inputEl.rows = 6;
        t.inputEl.addClass("arxiv-daily-settings__textarea");
      });
  }
}

/**
 * Tag a topic gets from its name: the slug, or `topic-N` when the name has
 * no usable characters, with a numeric suffix until no other topic has it.
 */
export function autoTopicTag(name: string, topics: readonly Topic[], selfId: string): string {
  const taken = new Set(
    topics.filter((topic) => topic.id !== selfId).map((topic) => topic.tag.trim()),
  );
  const base = slugify(name);
  if (!base) {
    const position = topics.findIndex((topic) => topic.id === selfId);
    let n = (position >= 0 ? position : topics.length) + 1;
    while (taken.has(`topic-${n}`)) n += 1;
    return `topic-${n}`;
  }
  if (!taken.has(base)) return base;
  let suffix = 2;
  while (taken.has(`${base}-${suffix}`)) suffix += 1;
  return `${base}-${suffix}`;
}

function stableDomId(value: string): string {
  const normalized = value.replace(/[^A-Za-z0-9_-]/g, "-");
  return normalized || "unnamed";
}

/** The stand-in tag a blank topic starts with, before it has a name. */
export function isPlaceholderTopicTag(tag: string): boolean {
  return /^topic-\d+$/.test(tag);
}

/**
 * Whether `tag` is the machine form of `name`: the slug, or the slug plus the
 * numeric suffix uniqueness adds. Retiring the tag field made this the test
 * for "nobody chose this tag on purpose", so renaming may replace it.
 */
export function isDerivedTopicTag(tag: string, name: string): boolean {
  const base = slugify(name);
  if (!base) return false;
  // `base` is a slug, so it holds no regular-expression metacharacters.
  return tag === base || new RegExp(`^${base}-\\d+$`).test(tag);
}

/**
 * First free tag for `base`, ignoring the topic at `index` so a topic keeps
 * its own tag. Duplicate tags fail validation outright, and the settings page
 * no longer offers a field to resolve a collision by hand.
 */
export function uniqueTopicTag(
  topics: readonly Topic[],
  index: number,
  base: string,
): string {
  const taken = new Set(
    topics.filter((_, position) => position !== index).map(({ tag }) => tag),
  );
  if (!taken.has(base)) return base;
  for (let suffix = 2; ; suffix += 1) {
    const candidate = `${base}-${suffix}`;
    if (!taken.has(candidate)) return candidate;
  }
}

export { isValidLocalTime, runWindowTimeOptions, type RunWindowTimeOption } from "@arxiv-daily/core";

export function renderRunWindowTimeSelect(
  parent: HTMLElement,
  labelText: string,
  id: string,
  current: string,
  onChange: (value: string) => Promise<void>,
): void {
  const field = parent.createDiv({
    cls: "arxiv-daily-settings__time-field",
  });
  field.createEl("label", {
    cls: "arxiv-daily-settings__time-label",
    text: labelText,
    attr: { for: id },
  });
  const select = field.createEl("select", {
    cls: "dropdown arxiv-daily-settings__time-select",
    attr: { id },
  });
  for (const option of runWindowTimeOptions(current)) {
    const optionEl = select.createEl("option", {
      value: option.value,
      text: option.label,
    });
    optionEl.disabled = !option.valid;
  }
  select.value = current;
  let revision = 0;
  let latestSuccessful = current;
  let saveQueue = Promise.resolve();
  select.addEventListener("change", () => {
    const next = select.value;
    if (!isValidLocalTime(next)) return;
    const changeRevision = ++revision;
    const operation = saveQueue.then(async () => {
      try {
        await onChange(next);
        latestSuccessful = next;
        if (revision === changeRevision) select.value = next;
      } catch (error) {
        if (revision === changeRevision) select.value = latestSuccessful;
        throw error;
      }
    });
    saveQueue = operation.catch(() => undefined);
  });
}

/** Timezone presets for the arXiv section; shared by display() and the 1.13+ rows. */

export function addCategoryOptions(
  selectEl: HTMLSelectElement,
  current?: string,
): void {
  let hasCurrent = false;
  for (const group of ARXIV_CATEGORIES) {
    const optgroup = selectEl.createEl("optgroup");
    optgroup.label = group.label;
    for (const cat of group.categories) {
      if (cat.id === current) hasCurrent = true;
      const opt = optgroup.createEl("option");
      opt.value = cat.id;
      opt.textContent = `${cat.id} — ${cat.name}`;
    }
  }
  if (current && !hasCurrent) {
    const opt = selectEl.createEl("option");
    opt.value = current;
    opt.textContent = `${current} — custom`;
  }
}

function normalizeUniqueCategories(categories: string[]): string[] {
  const out: string[] = [];
  for (const value of categories) {
    const category = value.trim();
    if (!category || out.includes(category)) continue;
    out.push(category);
  }
  return out;
}

function nextCategoryCandidate(existing: string[]): string {  const seen = new Set(existing);
  for (const group of ARXIV_CATEGORIES) {
    for (const category of group.categories) {
      if (!seen.has(category.id)) return category.id;
    }
  }
  return "cs.LG";
}

function categoriesWillChange(current: string[], next: string[]): boolean {
  if (current.length !== next.length) return true;
  return current.some((category, index) => category !== next[index]);
}

function quickStartTemplateConfirmMessage(
  topicCount: number,
  templateName: string,
  replacesCategories: boolean,
): string {
  if (topicCount > 0 && replacesCategories) {
    return `Replace your ${topicCount} topic(s) and arXiv categories with the "${templateName}" template?`;
  }
  if (topicCount > 0) {
    return `Replace your ${topicCount} topic(s) with the "${templateName}" template?`;
  }
  return `Replace your arXiv categories with the "${templateName}" template?`;
}
