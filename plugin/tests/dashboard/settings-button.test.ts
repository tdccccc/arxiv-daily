import { describe, it, expect, vi } from "vitest";
import { appendSettingsButton } from "../../src/dashboard/view";

function createParent(): HTMLElement {
  const parent = document.createElement("div");
  (parent as any).createEl = (
    tag: string,
    options: { cls?: string; attr?: Record<string, string> } = {},
  ) => {
    const child = parent.ownerDocument.createElement(tag);
    if (options.cls) child.className = options.cls;
    for (const [name, value] of Object.entries(options.attr ?? {})) {
      child.setAttribute(name, value);
    }
    (child as any).createSpan = ({ text }: { text: string }) => {
      const span = child.ownerDocument.createElement("span");
      span.textContent = text;
      child.append(span);
      return span;
    };
    parent.append(child);
    return child;
  };
  return parent;
}

describe("Settings Button", () => {
  it("creates a button with the correct class", () => {
    const parent = createParent();
    appendSettingsButton(parent, () => {});
    const button = parent.querySelector("button");
    expect(button).not.toBeNull();
    expect(button!.classList.contains("arxiv-daily-dashboard__settings-btn")).toBe(true);
  });

  it("sets the aria-label for accessibility", () => {
    const parent = createParent();
    appendSettingsButton(parent, () => {});
    const button = parent.querySelector("button")!;
    expect(button.getAttribute("aria-label")).toBe("Open arXiv Daily settings");
  });

  it("renders a Settings text label inside the button", () => {
    const parent = createParent();
    appendSettingsButton(parent, () => {});
    const span = parent.querySelector("button > span");
    expect(span).not.toBeNull();
    expect(span!.textContent).toBe("Settings");
  });

  it("calls the onClick handler when clicked", () => {
    const parent = createParent();
    const onClick = vi.fn();
    appendSettingsButton(parent, onClick);
    const button = parent.querySelector("button")!;
    button.click();
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
