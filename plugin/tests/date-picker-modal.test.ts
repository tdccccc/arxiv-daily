import { afterEach, describe, expect, it, vi } from "vitest";
import { Modal, type App } from "obsidian";
import { openDatePickerModal } from "../src/date-picker-modal";

// The host extends HTMLElement; keep only the DOM helpers exercised by this modal.
Object.assign(HTMLElement.prototype, {
  createEl(this: HTMLElement, tag: string, options: { text?: string; cls?: string; attr?: Record<string, string> } = {}) {
    const element = document.createElement(tag);
    if (options.text) element.textContent = options.text;
    if (options.cls) element.className = options.cls;
    for (const [name, value] of Object.entries(options.attr ?? {})) element.setAttribute(name, value);
    this.appendChild(element); return element;
  },
  createDiv(this: HTMLElement, options = {}) { return this.createEl("div", options); },
  empty(this: HTMLElement) { this.replaceChildren(); },
});

afterEach(() => { Modal.opened.length = 0; document.body.replaceChildren(); });

describe("date picker with the host button's internal disabled guard", () => {
  it.each(["click", "Enter"])("submits a valid date through %s after starting disabled", trigger => {
    const submit = vi.fn(), notice = vi.fn();
    openDatePickerModal({} as App, submit, {}, notice);
    const modal = Modal.opened.at(-1)!;
    document.body.appendChild(modal.contentEl);
    const input = modal.contentEl.querySelector<HTMLInputElement>('input[type="date"]')!;
    const button = modal.contentEl.querySelector<HTMLButtonElement>("button")!;
    expect(button.disabled).toBe(true);
    button.click(); expect(submit).not.toHaveBeenCalled();
    input.value = "2026-10-01";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    expect(button.disabled).toBe(false);
    if (trigger === "click") button.click();
    else input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
    expect(submit).toHaveBeenCalledExactlyOnceWith("2026-10-01");
    expect(modal.contentEl.childElementCount).toBe(0);
    expect(notice).not.toHaveBeenCalled();
  });
});
