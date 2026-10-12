import { Menu } from "obsidian";

/**
 * Shared marker on the root element of this plugin's "More" `Menu`, mirroring
 * `ARXIV_DAILY_MODAL_CLASS`, so tooling scoped to this plugin's UI can tell it
 * apart from Obsidian's and other plugins' menus. `Menu.dom` is not in the
 * public typings but is created in the constructor; if it is ever missing,
 * the menu is simply left unmarked.
 */
export const ARXIV_DAILY_MENU_CLASS = "arxiv-daily-menu";

/** Adds the shared marker to this plugin's "More" menu, right after construction. */
export function markArxivDailyMenu(menu: Menu): void {
  (menu as unknown as { dom?: HTMLElement }).dom?.addClass(ARXIV_DAILY_MENU_CLASS);
}
