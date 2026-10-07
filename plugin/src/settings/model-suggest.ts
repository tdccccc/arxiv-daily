import { AbstractInputSuggest, type App } from "obsidian";

/**
 * Type-ahead for the model field: one box you can both type into and pick
 * from, instead of a separate control that only appears after a fetch.
 *
 * `selectSuggestion` is implemented here rather than left to
 * AbstractInputSuggest's own default (which only exists from Obsidian
 * 1.6.6, per its `@since` tag in obsidian.d.ts) so the suggest works on
 * the lower floor `getSuggestions` itself needs (1.5.7, likewise by its
 * `@since` tag) — see the minAppVersion bump in manifest.json.
 */
export class ModelInputSuggest extends AbstractInputSuggest<string> {
  private models: string[] = [];
  private currentModel = "";

  constructor(
    app: App,
    private readonly inputEl: HTMLInputElement,
    private readonly onPick: (model: string) => void,
  ) {
    super(app, inputEl);
  }

  /** Replaces the fetched list and the model it should be ordered/marked against. */
  setModels(models: string[], currentModel: string): void {
    this.models = models;
    this.currentModel = currentModel;
  }

  /** Whether there is anything to suggest at all — false means type as plain text. */
  hasModels(): boolean {
    return this.models.length > 0;
  }

  /** Re-opens with the full list, e.g. right after a fetch completes. */
  showAll(): void {
    if (!this.hasModels()) return;
    this.inputEl.focus();
    // AbstractInputSuggest listens for the input element's own "input"
    // event to recompute and open suggestions; there is no public method
    // to trigger that directly.
    this.inputEl.dispatchEvent(new Event("input"));
  }

  /**
   * Exposes `getSuggestions` (protected, per the abstract contract) for
   * tests: happy-dom does not render the real floating suggestion popup,
   * so unit tests assert on the computed list directly instead.
   */
  previewSuggestions(query: string): string[] {
    return this.getSuggestions(query);
  }

  protected getSuggestions(query: string): string[] {
    if (!this.hasModels()) return [];
    const ordered = this.currentModel && this.models.includes(this.currentModel)
      ? [this.currentModel, ...this.models.filter((model) => model !== this.currentModel)]
      : this.models;
    const trimmed = query.trim();
    if (trimmed === "" || trimmed === this.currentModel) return ordered;
    const needle = trimmed.toLowerCase();
    return ordered.filter((model) => model.toLowerCase().includes(needle));
  }

  renderSuggestion(value: string, el: HTMLElement): void {
    el.setText(value);
    if (value === this.currentModel) {
      el.createSpan({ cls: "arxiv-daily-settings__model-suggestion-current", text: " (current)" });
    }
  }

  selectSuggestion(value: string): void {
    this.setValue(value);
    this.close();
    this.onPick(value);
  }
}
