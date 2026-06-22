import { describe, it, expect, vi } from "vitest";
import { appendSettingsButton } from "../../src/dashboard/view";

describe("Settings Button", () => {
  it("creates a button with the correct class", () => {
    const parent = document.createElement("div");
    appendSettingsButton(parent, () => {});
    const button = parent.querySelector("button");
    expect(button).not.toBeNull();
    expect(button!.classList.contains("arxiv-daily-dashboard__settings-btn")).toBe(true);
  });

  it("sets the aria-label for accessibility", () => {
    const parent = document.createElement("div");
    appendSettingsButton(parent, () => {});
    const button = parent.querySelector("button")!;
    expect(button.getAttribute("aria-label")).toBe("Open arXiv Daily settings");
  });

  it("renders a Settings text label inside the button", () => {
    const parent = document.createElement("div");
    appendSettingsButton(parent, () => {});
    const span = parent.querySelector("button > span");
    expect(span).not.toBeNull();
    expect(span!.textContent).toBe("Settings");
  });

  it("calls the onClick handler when clicked", () => {
    const parent = document.createElement("div");
    const onClick = vi.fn();
    appendSettingsButton(parent, onClick);
    const button = parent.querySelector("button")!;
    button.click();
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
