import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import { Modal, type App } from "obsidian";
import {
  confirmLibraryAuthorization,
  confirmLibraryRevocation,
  showLibraryInventoryPreview,
  showPersonalLibraryCatalogSummary,
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

    expect(modal.titleEl.textContent).toBe("Send titles and abstracts off this device?");
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

  it("discloses destination, purpose, containment, and reversibility at full-text depth", async () => {
    const result = confirmLibraryAuthorization({} as App, {
      ...disclosure,
      processingDepth: "full-text",
      embeddingEndpoint: "https://embed.example.com/v1/embeddings",
    });
    const modal = Modal.opened.at(-1)!;
    const text = modal.contentEl.textContent ?? "";

    expect(modal.titleEl.textContent).toBe("Send full text off this device?");
    expect(text).toContain("/private/papers");
    expect(text).toContain("Full text");
    expect(text).toContain("https://embed.example.com/v1/embeddings");
    expect(text).toMatch(/similarity vectors/i);
    expect(text).toMatch(/nothing else/i);
    expect(text).toMatch(/revoke/i);
    modal.contentEl.querySelectorAll<HTMLButtonElement>("button")[1]?.click();
    await expect(result).resolves.toBe(true);
  });

  /**
   * The desktop acceptance run finds this dialog by this class, precisely so it
   * does not have to find it by its heading — the heading is copy that follows
   * the processing depth. The literal is spelled out here rather than imported
   * so that renaming the class fails here instead of silently agreeing with
   * itself, and the acceptance run's own literal is checked in
   * `scripts/tests/desktop-acceptance-library-settings.test.mjs`.
   */
  it("marks its root with the stable class the acceptance run locates it by", async () => {
    const result = confirmLibraryAuthorization({} as App, disclosure);
    const modal = Modal.opened.at(-1)!;

    expect(modal.modalEl.classList.contains("arxiv-daily-library-authorization-modal")).toBe(true);
    modal.close();
    await expect(result).resolves.toBe(false);
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

describe("personal library revocation modal", () => {
  it("says revoking also switches embedding back to local and invalidates the index", async () => {
    const result = confirmLibraryRevocation({} as App, { switchesToLocal: true });
    const modal = Modal.opened.at(-1)!;
    const text = `${modal.titleEl.textContent ?? ""}\n${modal.contentEl.textContent ?? ""}`;

    expect(text).toMatch(/local embedding/i);
    expect(text).toMatch(/rebuilt/i);
    const buttons = [...modal.contentEl.querySelectorAll<HTMLButtonElement>("button")];
    expect(buttons.map((button) => button.textContent)).toEqual(["Cancel", "Revoke"]);
    buttons[1]?.click();
    await expect(result).resolves.toBe(true);
  });

  it("omits the embedding switch for a library that is already local", async () => {
    const result = confirmLibraryRevocation({} as App, { switchesToLocal: false });
    const modal = Modal.opened.at(-1)!;
    expect(modal.contentEl.textContent ?? "").not.toMatch(/switch/i);
    modal.contentEl.querySelector<HTMLButtonElement>("button")!.click();
    await expect(result).resolves.toBe(false);
  });

  it("treats dismissal as cancel", async () => {
    const result = confirmLibraryRevocation({} as App, { switchesToLocal: true });
    Modal.opened.at(-1)!.close();
    await expect(result).resolves.toBe(false);
  });
});

describe("personal library catalog modal", () => {
  it("renders revision and every scan count without an absolute root", () => {
    showPersonalLibraryCatalogSummary({} as App, {
      schemaVersion: 1,
      revision: 7,
      scopeFingerprint: `sha256:${"a".repeat(64)}`,
      identificationFingerprint: `sha256:${"b".repeat(64)}`,
      updatedAt: "2026-08-03T00:00:00.000Z",
      lastScan: {
        ready: 3,
        papers: 2,
        unresolved: 4,
        unrelated: 5,
        failed: 6,
        truncated: true,
      },
      files: {},
      papers: {},
    });
    const modal = Modal.opened.at(-1)!;
    expect(modal.titleEl.textContent).toBe("Personal library catalog");
    expect(modal.contentEl.textContent).toContain("Revision7");
    expect(modal.contentEl.textContent).toContain("Ready files3");
    expect(modal.contentEl.textContent).toContain("Papers2");
    expect(modal.contentEl.textContent).toContain("Unresolved4");
    expect(modal.contentEl.textContent).toContain("Unrelated5");
    expect(modal.contentEl.textContent).toContain("Failed6");
    expect(modal.contentEl.textContent).toContain("TruncatedYes");
    expect(modal.contentEl.textContent).not.toContain("/private");
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
