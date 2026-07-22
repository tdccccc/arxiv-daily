import { App, Modal, Setting } from "obsidian";

export type DatePickerNotice = (message: string, timeoutMs?: number) => void;

export interface DatePickerOptions {
  title?: string;
  desc?: string;
  buttonText?: string;
}

export function isValidCalendarDate(value: string): boolean {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return false;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const date = new Date(Date.UTC(year, month - 1, day));
  return (
    date.getUTCFullYear() === year &&
    date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day
  );
}

export function bindEnterToButton(
  input: HTMLInputElement,
  button: HTMLButtonElement,
): void {
  input.addEventListener("keydown", (evt) => {
    if (evt.key !== "Enter") return;
    evt.preventDefault();
    button.click();
  });
}

export function openDatePickerModal(
  app: App,
  onSubmit: (date: string) => void,
  opts: DatePickerOptions,
  notice: DatePickerNotice,
): void {
  new DatePickerModal(app, onSubmit, opts, notice).open();
}

class DatePickerModal extends Modal {
  private value = "";

  constructor(
    app: App,
    private onSubmit: (date: string) => void,
    private opts: DatePickerOptions,
    private notice: DatePickerNotice,
  ) {
    super(app);
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.createEl("h2", { text: this.opts.title ?? "Run arXiv Daily for date" });
    let inputEl: HTMLInputElement | null = null;
    let submitButton: HTMLButtonElement | null = null;
    const dateSetting = new Setting(contentEl)
      .setName("Date")
      .setDesc(this.opts.desc ?? "Choose a real calendar date within the supported arXiv window.")
      .addText((t) => {
        inputEl = t.inputEl;
        t.inputEl.type = "date";
        t.inputEl.setAttribute("aria-describedby", "arxiv-daily-date-error");
        t.onChange((v) => {
          this.value = v.trim();
          refreshValidation();
        });
      });
    const errorEl = dateSetting.descEl.createEl("div", {
      attr: { id: "arxiv-daily-date-error", "aria-live": "polite" },
    });
    const refreshValidation = () => {
      const valid = isValidCalendarDate(this.value);
      errorEl.textContent = this.value && !valid ? "Enter a valid calendar date." : "";
      if (submitButton) submitButton.disabled = !valid;
      inputEl?.setAttribute("aria-invalid", String(Boolean(this.value) && !valid));
    };
    new Setting(contentEl).addButton((b) => {
      submitButton = b.buttonEl;
      b
        .setButtonText(this.opts.buttonText ?? "Run")
        .setCta()
        .setDisabled(true)
        .onClick(() => {
          if (!isValidCalendarDate(this.value)) {
            refreshValidation();
            this.notice("Invalid calendar date");
            return;
          }
          this.close();
          this.onSubmit(this.value);
        });
    });
    if (inputEl && submitButton) bindEnterToButton(inputEl, submitButton);
  }

  onClose() {
    this.contentEl.empty();
  }
}
