import { App, Modal } from "obsidian";

/**
 * Shared marker every modal this plugin opens carries on `modalEl`, alongside
 * each modal's own class. Lets tooling scoped to this plugin's UI (e.g.
 * acceptance exploration) find every plugin modal.
 */
export const ARXIV_DAILY_MODAL_CLASS = "arxiv-daily-modal";

/**
 * Adds the shared marker to a modal constructed via a bare `new Modal(app)`
 * (this plugin has several one-off confirmation/info dialogs that do not
 * warrant their own subclass).
 */
export function markArxivDailyModal(modal: Modal): void {
  modal.modalEl.addClass(ARXIV_DAILY_MODAL_CLASS);
}

/**
 * Base class for this plugin's `Modal` subclasses: adds the shared marker
 * once, in the constructor, so subclasses don't each have to remember to.
 */
export class ArxivDailyModal extends Modal {
  constructor(app: App) {
    super(app);
    markArxivDailyModal(this);
  }
}
