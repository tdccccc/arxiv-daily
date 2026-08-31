import { App, Modal } from "obsidian";
import type { PersonalLibraryCatalog } from "@arxiv-daily/core";
import type {
  LibraryAuthorizationDisclosure,
  LibraryInventoryPreview,
} from "./connection";

/**
 * Stable handle on the disclosure dialog, for anything that has to find it
 * from the outside (the desktop acceptance run drives it through this).
 * Deliberately not derived from the heading: the heading is product copy that
 * follows the processing depth, so locating by it would turn every wording
 * change into a "dialog not found" failure instead of a wording failure.
 */
export const LIBRARY_AUTHORIZATION_MODAL_CLASS = "arxiv-daily-library-authorization-modal";

/**
 * Stable handles on the two answers, for the same reason as the class above:
 * the confirm button's label is product copy that follows the processing
 * depth, so anything that clicks it from the outside has to find it by mark,
 * not by wording. Clicking by label is what turns a copy change into "the
 * button is missing" — the dialog-level version of that mistake is documented
 * on LIBRARY_AUTHORIZATION_MODAL_CLASS.
 */
export const LIBRARY_AUTHORIZATION_CONFIRM_CLASS = "arxiv-daily-library-authorization-confirm";
export const LIBRARY_AUTHORIZATION_CANCEL_CLASS = "arxiv-daily-library-authorization-cancel";

/**
 * The words the dialog asks and answers with, taken from the depth in one
 * place.
 *
 * The heading names what leaves the device, and it has to name the depth the
 * grant actually covers: a fixed "full text" heading would misdescribe a
 * metadata-and-abstracts grant exactly the way the depth and endpoint fields
 * used to before they were made to follow the scope being asked about.
 *
 * The confirm button has to answer that heading in the same words. "Authorize"
 * answers a question nobody was asked: the heading asks whether to send
 * something, so the affirmative is sending it, not granting an authorization
 * the reader then has to translate back. Heading and button are returned
 * together, from a single branch on the depth, so a third depth cannot be
 * added to one of them and forgotten in the other.
 */
export function libraryAuthorizationCopy(
  processingDepth: LibraryAuthorizationDisclosure["processingDepth"],
): { title: string; confirm: string } {
  return processingDepth === "full-text"
    ? { title: "Send full text off this device?", confirm: "Send full text" }
    : {
        title: "Send titles and abstracts off this device?",
        confirm: "Send titles and abstracts",
      };
}

export function confirmLibraryAuthorization(
  app: App,
  disclosure: LibraryAuthorizationDisclosure,
): Promise<boolean> {
  return new Promise((resolve) => {
    const modal = new Modal(app);
    const copy = libraryAuthorizationCopy(disclosure.processingDepth);
    modal.modalEl.addClass(LIBRARY_AUTHORIZATION_MODAL_CLASS);
    modal.titleEl.setText(copy.title);
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
        ? "Authorizing permits full-text chunks from that folder to be sent to the endpoints above, where they are turned into similarity vectors for library search. Nothing else on this device is sent, and you can revoke this from settings at any time. Changing the folder, endpoints, file types, or depth invalidates it."
        : "Inventory preview is local, read-only, and does not require this authorization. Authorizing permits eligible metadata and abstracts to be sent to the endpoint in later profile-building steps. Nothing else on this device is sent, and you can revoke this from settings at any time. Changing the folder, endpoint, file types, or depth invalidates it.",
    });
    const actions = modal.contentEl.createDiv({ cls: "arxiv-daily-modal-button-row" });
    let settled = false;
    const finish = (value: boolean) => {
      if (settled) return;
      settled = true;
      resolve(value);
      modal.close();
    };
    const cancel = actions.createEl("button", { text: "Cancel" });
    cancel.addClass(LIBRARY_AUTHORIZATION_CANCEL_CLASS);
    cancel.onclick = () => finish(false);
    const confirm = actions.createEl("button", { text: copy.confirm });
    confirm.addClass(LIBRARY_AUTHORIZATION_CONFIRM_CLASS);
    confirm.addClass("mod-cta");
    confirm.onclick = () => finish(true);
    modal.onClose = () => finish(false);
    modal.open();
  });
}

/**
 * Confirms revoking the personal-library grant. Revoking a remote-embedding
 * grant also returns embedding to local, so no configuration is left in the
 * dead "remote but unauthorized" state; that consequence — including the
 * invalidated index — is disclosed here before anything changes.
 */
export function confirmLibraryRevocation(
  app: App,
  options: { switchesToLocal: boolean },
): Promise<boolean> {
  return new Promise((resolve) => {
    const modal = new Modal(app);
    modal.titleEl.setText(
      options.switchesToLocal
        ? "Revoke authorization and return to local embedding?"
        : "Revoke personal library authorization?",
    );
    modal.contentEl.createEl("p", {
      text: options.switchesToLocal
        ? "arXiv Daily stops sending anything from your library folder to the model endpoints, and embedding returns to local (offline) on this device."
        : "arXiv Daily stops sending anything from your library folder to the model endpoints. Local embedding keeps working offline.",
    });
    if (options.switchesToLocal) {
      modal.contentEl.createEl("p", {
        text: "The existing search index was built with the remote embedding model. Local and remote vectors cannot be mixed, so the index stops being usable and has to be rebuilt on this device (that takes hours for a large library).",
      });
    }
    const actions = modal.contentEl.createDiv({ cls: "arxiv-daily-modal-button-row" });
    let settled = false;
    const finish = (value: boolean) => {
      if (settled) return;
      settled = true;
      resolve(value);
      modal.close();
    };
    actions.createEl("button", { text: "Cancel" }).onclick = () => finish(false);
    const revoke = actions.createEl("button", { text: "Revoke" });
    revoke.addClass("mod-warning");
    revoke.onclick = () => finish(true);
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
  modal.contentEl.createEl("p", {
    cls: "mod-hint",
    text: "Unresolved files have no arXiv id; they are indexed and searchable as local "
      + "documents by the full-text index (Run index-personal-library-fulltext).",
  });
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

/**
 * First-time guided choice between local and remote embedding (ADR 0008),
 * shown right after a library is first connected — before the user hits the
 * long local index. Dismissing the modal keeps the local default; the choice
 * is always changeable later in settings (switching rebuilds the index).
 */
export function confirmEmbeddingMode(app: App): Promise<"local" | "remote"> {
  return new Promise((resolve) => {
    const modal = new Modal(app);
    modal.titleEl.setText("Embedding mode");
    modal.contentEl.createEl("p", {
      text: "How should arXiv Daily turn your library's full text into similarity vectors? "
        + "You can change this later in settings — switching modes rebuilds the index.",
    });
    const list = modal.contentEl.createEl("dl");
    addDisclosure(
      list,
      "Local (offline, default)",
      "Embeds on this device with a bundled model. Private and offline, but indexing a large library takes a long time (hours).",
    );
    addDisclosure(
      list,
      "Remote (fast)",
      "Sends full-text chunks to an embeddings API. Indexing takes minutes; requires model authorization at full-text depth; full text leaves this device.",
    );
    const actions = modal.contentEl.createDiv({ cls: "arxiv-daily-modal-button-row" });
    let settled = false;
    const finish = (mode: "local" | "remote") => {
      if (settled) return;
      settled = true;
      resolve(mode);
      modal.close();
    };
    actions.createEl("button", { text: "Local" }).onclick = () => finish("local");
    const remote = actions.createEl("button", { text: "Remote" });
    remote.addClass("mod-cta");
    remote.onclick = () => finish("remote");
    modal.onClose = () => finish("local");
    modal.open();
  });
}
