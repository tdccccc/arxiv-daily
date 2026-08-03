import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { Modal, type App } from "obsidian";
import {
  confirmLibraryAuthorization,
  showLibraryInventoryPreview,
} from "../src/library/modal";

beforeAll(() => {
  type CreateOptions = { cls?: string; text?: string };
  const proto = HTMLElement.prototype as HTMLElement & {
    addClass?: (...classes: string[]) => void;
    createEl?: (tag: string, options?: CreateOptions) => HTMLElement;
    createDiv?: (options?: CreateOptions) => HTMLElement;
  };
  proto.addClass ??= function (...classes: string[]) {
    this.classList.add(...classes);
  };
  proto.createEl ??= function (tag, options = {}) {
    const element = document.createElement(tag);
    if (options.cls) element.className = options.cls;
    if (options.text) element.textContent = options.text;
    this.appendChild(element);
    return element;
  };
  proto.createDiv ??= function (options = {}) {
    return this.createEl!("div", options);
  };
  (proto as HTMLElement & { setText?: (text: string) => void }).setText ??=
    function (text: string) { this.textContent = text; };
});

beforeEach(() => {
  Modal.opened.length = 0;
});

const disclosure = {
  selectedRoot: "/private/papers",
  eligibleExtensions: [".pdf"],
  processingDepth: "metadata-and-abstracts" as const,
  endpoint: "https://example.com/v1",
  authorizationFingerprint: `sha256:${"a".repeat(64)}`,
};

describe("personal library authorization modal", () => {
  it("renders the exact scope and resolves authorization", async () => {
    const result = confirmLibraryAuthorization({} as App, disclosure);
    const modal = Modal.opened.at(-1)!;

    expect(modal.titleEl.textContent).toBe("Authorize personal library");
    expect(modal.contentEl.textContent).toContain("/private/papers");
    expect(modal.contentEl.textContent).toContain(".pdf");
    expect(modal.contentEl.textContent).toContain("Metadata and abstracts only");
    expect(modal.contentEl.textContent).toContain("https://example.com/v1");
    expect(modal.contentEl.textContent).toContain("local, read-only");
    expect(modal.contentEl.textContent).toContain("does not require this authorization");
    const buttons = [...modal.contentEl.querySelectorAll<HTMLButtonElement>("button")];
    expect(buttons.map((button) => button.textContent)).toEqual(["Cancel", "Authorize"]);
    expect(buttons[1]?.classList.contains("mod-cta")).toBe(true);
    buttons[1]?.click();
    await expect(result).resolves.toBe(true);
  });

  it("resolves false when cancelled or closed outside the buttons", async () => {
    const cancelled = confirmLibraryAuthorization({} as App, disclosure);
    Modal.opened.at(-1)!.contentEl.querySelector<HTMLButtonElement>("button")!.click();
    await expect(cancelled).resolves.toBe(false);

    const closed = confirmLibraryAuthorization({} as App, disclosure);
    Modal.opened.at(-1)!.close();
    await expect(closed).resolves.toBe(false);
  });
});

describe("personal library inventory modal", () => {
  it("renders counts, truncation, reasons, and caps each path group", () => {
    showLibraryInventoryPreview({} as App, {
      eligible: Array.from({ length: 102 }, (_, index) => ({
        path: `paper-${index}.pdf`,
      })),
      ignored: Array.from({ length: 101 }, (_, index) => ({
        path: `note-${index}.md`,
        reason: "Unsupported file type",
      })),
      folders: 7,
      truncated: true,
    });
    const modal = Modal.opened.at(-1)!;

    expect(modal.titleEl.textContent).toBe("Personal library inventory");
    expect(modal.contentEl.textContent).toContain("102 eligible PDFs, 101 ignored, 7 folders");
    expect(modal.contentEl.textContent).toContain("truncated");
    expect(modal.contentEl.textContent).toContain("note-0.md — Unsupported file type");
    expect(modal.contentEl.textContent).toContain("…and 2 more");
    expect(modal.contentEl.textContent).toContain("…and 1 more");
    expect(modal.contentEl.querySelectorAll("ul")[0]?.children).toHaveLength(101);
    expect(modal.contentEl.querySelectorAll("ul")[1]?.children).toHaveLength(101);
  });
});
