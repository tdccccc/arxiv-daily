const { DEFAULT_SETTINGS, PaperIndexStore, isPaperStatus } = require("@arxiv-daily/core");
const { NodeStorageAdapter } = require("@arxiv-daily/node-runtime/storage-adapter");
const { PAPER_INDEX_PATH } = require("./workspace");

/**
 * The Paper Index as Obsidian and the CLI use it: core's PaperIndexStore over
 * the Node storage adapter rooted at the vault folder. Writes therefore take
 * the same cross-process lock (NodeFileLock keyed by the vault's real path and
 * "paper-index:<papers.json path>") and use the same backup, temporary-file and
 * schema handling, so the three hosts exclude one another instead of losing each
 * other's changes.
 *
 * @param {{ vaultRootUri: { fsPath: string, scheme?: string } }} vault
 * @param {{ lockRoot?: string, onStaleOwner?: (message: string) => void }} [options]
 *   Test seams: a private lock directory, and where stale-lock notices go.
 */
function createPaperIndexStore(vault, options = {}) {
  const root = vault.vaultRootUri;
  if (root.scheme && root.scheme !== "file") {
    throw new Error(
      `the arXiv Daily Paper Index needs a local folder, not a ${root.scheme}: workspace`,
    );
  }
  const storage = new NodeStorageAdapter(root.fsPath, {
    lockRoot: options.lockRoot,
    onStaleOwner: options.onStaleOwner,
  });
  const store = new PaperIndexStore(storage, DEFAULT_SETTINGS.output);
  if (store.paths.papersJsonPath !== PAPER_INDEX_PATH) {
    throw new Error(
      `Paper Index location ${store.paths.papersJsonPath} does not match ${PAPER_INDEX_PATH}`,
    );
  }
  return store;
}

/** Loads the (normalized) index; a vault with no papers.json yet yields an empty one. */
async function loadPaperIndex(store) {
  return await store.load();
}

/**
 * Sets one paper's reading status through core's locked, queued transaction.
 * `id` is a papers.json map key ("arxiv:<id>") or a bare arXiv id.
 */
async function updatePaperStatus(store, id, status) {
  if (!isPaperStatus(status)) throw new Error(`invalid status: ${status}`);
  // Core reads an unreadable or newer-schema papers.json as corrupt and then
  // falls back to papers.json.bak, which a save would promote over it. Only
  // write when the primary file itself is the document being edited.
  const { recoveredFromBackup } = await store.inspect();
  if (recoveredFromBackup) {
    throw new Error(
      "papers.json cannot be read by this extension (damaged, or written by a newer version); open it in Obsidian or run the arxiv-daily CLI instead",
    );
  }
  const entry = await store.setStatus(id, status);
  if (!entry) throw new Error(`${id} is not in papers.json`);
  return entry;
}

module.exports = {
  createPaperIndexStore,
  loadPaperIndex,
  updatePaperStatus,
};
