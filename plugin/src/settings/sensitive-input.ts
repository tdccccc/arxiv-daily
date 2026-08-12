import { setIcon, type Setting } from "obsidian";
import type { ArxivDailySettingTab } from "./tab";

/**
 * Shared masked-input row for secrets (LLM API key, Resend API key,
 * hosted verification code): the persisted value is shown masked in the
 * input with a Show/Hide toggle, and edits commit transactionally on
 * blur or Enter.
 */
export interface SensitiveInputOptions {
  value: string;
  placeholder: string;
  ariaLabel: string;
  normalize?: (value: string) => string;
  save: (value: string) => Promise<void>;
  onCommitted?: () => void;
}

export function renderSensitiveInput(
  tab: ArxivDailySettingTab,
  setting: Setting,
  options: SensitiveInputOptions,
): () => Promise<void> {
  const input = setting.controlEl.createEl("input", {
    cls: "arxiv-daily-settings__llm-input",
    type: "password",
    attr: {
      placeholder: options.placeholder,
      autocomplete: "off",
      "aria-label": options.ariaLabel,
    },
  });
  input.value = options.value;

  const reveal = setting.controlEl.createEl("button", {
    cls: "arxiv-daily-settings__reveal-key",
    attr: {
      type: "button",
      "aria-label": `Show ${options.ariaLabel}`,
      title: `Show ${options.ariaLabel}`,
    },
  });
  setIcon(reveal, "eye");

  const setRevealed = (revealed: boolean) => {
    input.type = revealed ? "text" : "password";
    const action = revealed ? "Hide" : "Show";
    reveal.setAttribute("aria-label", `${action} ${options.ariaLabel}`);
    reveal.title = `${action} ${options.ariaLabel}`;
    reveal.empty();
    setIcon(reveal, revealed ? "eye-off" : "eye");
  };
  reveal.addEventListener("pointerdown", (event) => event.preventDefault());
  reveal.addEventListener("click", () => setRevealed(input.type === "password"));

  let savedValue = input.value;
  let saveQueue = Promise.resolve();
  let latestSave: { value: string; promise: Promise<void> } | undefined;
  const save = (): Promise<void> => {
    const next = (options.normalize ?? ((value: string) => value.trim()))(input.value);
    if (next === savedValue) return Promise.resolve();
    if (latestSave?.value === next) return latestSave.promise;
    const revision = tab.beginControlChange(input);
    const operation = saveQueue.then(async () => {
      try {
        await options.save(next);
        if (tab.isCurrentControlChange(input, revision)) {
          savedValue = next;
          input.value = next;
          options.onCommitted?.();
        }
      } catch (error) {
        if (tab.isCurrentControlChange(input, revision)) input.value = savedValue;
        throw error;
      }
    });
    let tracked: Promise<void>;
    tracked = operation.finally(() => {
      if (latestSave?.promise === tracked) latestSave = undefined;
    });
    latestSave = { value: next, promise: tracked };
    saveQueue = tracked.catch(() => undefined);
    return tracked;
  };
  input.addEventListener("blur", () => tab.runAction(`save ${options.ariaLabel}`, save));
  input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    tab.runAction(`save ${options.ariaLabel}`, save);
  });
  return save;
}
