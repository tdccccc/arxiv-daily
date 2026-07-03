import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const styles = readFileSync(resolve(process.cwd(), "styles.css"), "utf-8");
const settingsTab = readFileSync(resolve(process.cwd(), "src/settings/tab.ts"), "utf-8");

describe("dashboard and settings styles", () => {
  it("uses compact LLM and model controls", () => {
    expect(styles).toMatch(
      /\.arxiv-daily-settings__llm-input\s*\{[^}]*width:\s*260px;/s,
    );
    expect(styles).toMatch(
      /\.arxiv-daily-settings__model-select\s*\{[^}]*min-width:\s*160px;/s,
    );
  });

  it("renders the schedule run window controls", () => {
    expect(settingsTab).toContain('.setName("Run window")');
    expect(settingsTab).toContain("runUntilLocal");
    expect(settingsTab).toContain('setPlaceholder("09:00")');
    expect(settingsTab).toContain('setPlaceholder("18:00")');
    expect(settingsTab).not.toContain('.setName("Run time (HH:MM)")');
  });

  it("does not render a custom calendar hover tooltip from aria-label", () => {
    expect(styles).not.toContain(
      ".arxiv-daily-dashboard__calendar-day[aria-label]::after",
    );
    expect(styles).not.toContain(
      ".arxiv-daily-dashboard__calendar-day[aria-label]:hover::after",
    );
  });

  it("uses no-relevant-papers calendar class naming", () => {
    expect(styles).toContain(".arxiv-daily-dashboard__calendar-day.no-relevant-papers");
    expect(styles).not.toContain(".arxiv-daily-dashboard__calendar-day.no-papers");
  });

  it("preserves the user-tuned runnable play icon dimensions", () => {
    expect(styles).toMatch(
      /\.arxiv-daily-dashboard__calendar-day\.is-runnable \.arxiv-daily-dashboard__calendar-day-icon\s*\{[^}]*right:\s*3px;[^}]*bottom:\s*3px;[^}]*width:\s*12px;[^}]*height:\s*12px;/s,
    );
    expect(styles).toMatch(
      /\.arxiv-daily-dashboard__calendar-day\.is-runnable \.arxiv-daily-dashboard__calendar-day-icon svg\s*\{[^}]*width:\s*12px;[^}]*height:\s*12px;/s,
    );
  });

  it("keeps calendar marker text at least 11px and readable", () => {
    expect(styles).toMatch(
      /\.arxiv-daily-dashboard__calendar-day\.no-relevant-papers\s*\{[^}]*font-size:\s*11px;[^}]*opacity:\s*0\.85;/s,
    );
    expect(styles).toMatch(
      /\.arxiv-daily-dashboard__calendar-day-count\s*\{[^}]*font-size:\s*11px;/s,
    );
  });

  it("does not disable text selection inside dashboard inputs", () => {
    expect(styles).not.toMatch(
      /\.arxiv-daily-dashboard button,\s*\.arxiv-daily-dashboard input,\s*\.arxiv-daily-dashboard select\s*\{[^}]*user-select:\s*none;/s,
    );
  });
});
