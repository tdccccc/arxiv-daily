import { App, Modal } from "obsidian";

export interface ChooseOption {
  label: string;
  value: string;
  cta?: boolean;
  warning?: boolean;
}

/**
 * Show a modal with arbitrary buttons. Resolves with the chosen value, or `null`
 * if the user closes the modal without clicking any button (Esc, click-out).
 */
export function chooseModal(
  app: App,
  title: string,
  message: string,
  options: ChooseOption[],
): Promise<string | null> {
  return new Promise((resolve) => {
    const modal = new Modal(app);
    modal.titleEl.setText(title);
    modal.contentEl.createEl("p", { text: message });
    const btnRow = modal.contentEl.createDiv({
      cls: "arxiv-daily-modal-button-row",
    });

    let settled = false;
    const finish = (value: string | null) => {
      if (settled) return;
      settled = true;
      resolve(value);
      modal.close();
    };

    for (const opt of options) {
      const btn = btnRow.createEl("button", { text: opt.label });
      if (opt.cta) btn.classList.add("mod-cta");
      if (opt.warning) btn.classList.add("mod-warning");
      btn.onclick = () => finish(opt.value);
    }

    modal.onClose = () => finish(null);
    modal.open();
  });
}
