import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Menu } from "obsidian";
import { ARXIV_DAILY_MENU_CLASS, markArxivDailyMenu } from "../src/plugin-menu";

const dashboardViewSource = readFileSync(
  resolve(process.cwd(), "src/dashboard/view.ts"),
  "utf-8",
);

describe("markArxivDailyMenu (plugin/src/plugin-menu.ts)", () => {
  it("adds the shared marker class to the Menu's own dom", () => {
    const menu = new Menu();
    expect((menu as unknown as { dom: HTMLElement }).dom.classList.contains(ARXIV_DAILY_MENU_CLASS)).toBe(false);
    markArxivDailyMenu(menu);
    expect((menu as unknown as { dom: HTMLElement }).dom.classList.contains(ARXIV_DAILY_MENU_CLASS)).toBe(true);
  });

  it("keeps the real Menu's own \"menu\" class alongside the shared marker", () => {
    const menu = new Menu();
    markArxivDailyMenu(menu);
    const dom = (menu as unknown as { dom: HTMLElement }).dom;
    expect(dom.classList.contains("menu")).toBe(true);
    expect(dom.classList.contains(ARXIV_DAILY_MENU_CLASS)).toBe(true);
  });

  it("is a no-op (does not throw) if dom is somehow missing, rather than crashing menu construction", () => {
    const bareMenu = {} as Menu;
    expect(() => markArxivDailyMenu(bareMenu)).not.toThrow();
  });
});

describe("the dashboard's \"More\" menu is marked (plugin/src/dashboard/view.ts showMoreMenu)", () => {
  // Source guard, same convention as plugin-modal.test.ts's "real plugin
  // modals carry the shared marker" section: constructing a real ItemView to
  // exercise showMoreMenu() (private, needs a full WorkspaceLeaf/app) is
  // heavy for a one-call-site marker; asserting the marker call sits right
  // next to the single `new Menu()` in the whole plugin is cheap and catches
  // the exact regression that matters (the marker call being removed, or a
  // second unmarked Menu being introduced elsewhere).
  it("showMoreMenu marks its Menu immediately after constructing it", () => {
    const match = dashboardViewSource.match(/const menu = new Menu\(\);\s*\n\s*markArxivDailyMenu\(menu\);/);
    expect(match, "showMoreMenu must call markArxivDailyMenu(menu) right after `new Menu()`").not.toBeNull();
  });

  it("is the only Menu construction in the plugin (so the marker is never missed on a second one)", () => {
    const occurrences = dashboardViewSource.match(/new Menu\(\)/g) ?? [];
    expect(occurrences.length).toBe(1);
  });
});
