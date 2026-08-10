import { App, Modal } from "obsidian";
import type { PersonalLibraryCatalog } from "@arxiv-daily/core";
import type {
  LibraryAuthorizationDisclosure,
  LibraryInventoryPreview,
} from "./connection";

export function confirmLibraryAuthorization(
  app: App,
  disclosure: LibraryAuthorizationDisclosure,
): Promise<boolean> {
  return new Promise((resolve) => {
    const modal = new Modal(app);
    modal.titleEl.setText("Authorize personal library");
    modal.contentEl.createEl("p", {
      text: "Review exactly what arXiv Daily may process through your configured model endpoints.",
    });
    const list = modal.contentEl.createEl("dl");
    addDisclosure(list, "Folder", disclosure.selectedRoot);
    addDisclosure(list, "Files", disclosure.eligibleExtensions.join(", "));
    addDisclosure(
      list,
      "Depth",
      disclosure.processingDepth === "full-text"
        ? "Full text (remote embedding sends full-text chunks to the embedding endpoint)"
        : "Metadata and abstracts only",
    );
    addDisclosure(list, "Endpoint", disclosure.endpoint);
    if (disclosure.embeddingEndpoint) {
      addDisclosure(list, "Embedding endpoint", disclosure.embeddingEndpoint);
    }
    modal.contentEl.createEl("p", {
      text: disclosure.processingDepth === "full-text"
        ? "Authorizing permits full-text chunks to be sent to the endpoints above for similarity indexing in later steps. Changing the folder, endpoints, file types, or depth invalidates it."
        : "Inventory preview is local, read-only, and does not require this authorization. Authorizing permits eligible metadata and abstracts to be sent to the endpoint in later profile-building steps. Changing the folder, endpoint, file types, or depth invalidates it.",
    });
    const actions = modal.contentEl.createDiv({ cls: "arxiv-daily-modal-button-row" });
    let settled = false;
    const finish = (value: boolean) => {
      if (settled) return;
      settled = true;
      resolve(value);
      modal.close();
    };
    actions.createEl("button", { text: "Cancel" }).onclick = () => finish(false);
    const authorize = actions.createEl("button", { text: "Authorize" });
    authorize.addClass("mod-cta");
    authorize.onclick = () => finish(true);
    modal.onClose = () => finish(false);
    modal.open();
  });
}

export function showPersonalLibraryCatalogSummary(
  app: App,
  catalog: PersonalLibraryCatalog,
): void {
  const modal = new Modal(app);
  modal.titleEl.setText("Personal library catalog");
  const summary = catalog.lastScan ?? {
    ready: 0,
    papers: Object.keys(catalog.papers).length,
    unresolved: 0,
    unrelated: 0,
    failed: 0,
    truncated: false,
  };
  const list = modal.contentEl.createEl("dl");
  addDisclosure(list, "Revision", String(catalog.revision));
  addDisclosure(list, "Ready files", String(summary.ready));
  addDisclosure(list, "Papers", String(summary.papers));
  addDisclosure(list, "Unresolved", String(summary.unresolved));
  addDisclosure(list, "Unrelated", String(summary.unrelated));
  addDisclosure(list, "Failed", String(summary.failed));
  addDisclosure(list, "Truncated", summary.truncated ? "Yes" : "No");
  const actions = modal.contentEl.createDiv({ cls: "arxiv-daily-modal-button-row" });
  actions.createEl("button", { text: "Close" }).onclick = () => modal.close();
  modal.open();
}

export function showLibraryInventoryPreview(
  app: App,
  preview: LibraryInventoryPreview,
): void {
  const modal = new Modal(app);
  modal.titleEl.setText("Personal library inventory");
  modal.contentEl.createEl("p", {
    text: `${preview.eligible.length} eligible PDF${preview.eligible.length === 1 ? "" : "s"}, ${preview.ignored.length} ignored, ${preview.folders} folders.`,
  });
  if (preview.truncated) {
    modal.contentEl.createEl("p", {
      cls: "mod-warning",
      text: "Preview was truncated by the configured safety limits.",
    });
  }
  renderPaths(modal.contentEl, "Eligible papers", preview.eligible.map((entry) => entry.path));
  renderPaths(
    modal.contentEl,
    "Ignored entries",
    preview.ignored.map((entry) => `${entry.path} — ${entry.reason}`),
  );
  const actions = modal.contentEl.createDiv({ cls: "arxiv-daily-modal-button-row" });
  const close = actions.createEl("button", { text: "Close" });
  close.onclick = () => modal.close();
  modal.open();
}

function addDisclosure(list: HTMLElement, label: string, value: string): void {
  list.createEl("dt", { text: label });
  list.createEl("dd", { text: value });
}

function renderPaths(parent: HTMLElement, heading: string, paths: string[]): void {
  if (paths.length === 0) return;
  parent.createEl("h3", { text: heading });
  const list = parent.createEl("ul");
  for (const path of paths.slice(0, 100)) list.createEl("li", { text: path });
  if (paths.length > 100) list.createEl("li", { text: `…and ${paths.length - 100} more` });
}
