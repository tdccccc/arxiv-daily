import esbuild from "esbuild";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createTestVault } from "./test-vault.mjs";

// Obsidian and the CLI take core's cross-process lock before they edit
// papers.json. This test plays one of those hosts, holds that lock the way they
// would, and checks that the extension's status write waits for it instead of
// racing it, and that both hosts' changes survive.
const here = path.dirname(fileURLToPath(import.meta.url));
const { loadExtensionBundle } = await import("./load-bundle.mjs");
const { createPaperIndexStore, updatePaperStatus } = loadExtensionBundle().internals;

// The "other host" is built straight from the workspace sources, independently
// of the extension bundle, so it cannot share the extension's lock settings.
const peerDir = mkdtempSync(path.join(tmpdir(), "arxiv-daily-peer-"));
const peerBundle = path.join(peerDir, "peer.cjs");
await esbuild.build({
  stdin: {
    contents: 'module.exports = { NodeFileLock: require("@arxiv-daily/node-runtime/file-lock").NodeFileLock };',
    resolveDir: here,
    sourcefile: "peer-entry.js",
  },
  outfile: peerBundle,
  loader: { ".md": "text" },
  bundle: true,
  platform: "node",
  format: "cjs",
  target: "node20",
  logLevel: "warning",
});
const { NodeFileLock } = createRequire(import.meta.url)(peerBundle);

const entry = (id, overrides = {}) => ({
  paperKey: `arxiv:${id}`,
  source: "arxiv",
  externalId: id,
  arxivId: id,
  title: `Paper ${id}`,
  authors: ["Jane Doe"],
  published: "2026-06-13",
  updated: "2026-06-13",
  category: "astro-ph.CO",
  categories: ["astro-ph.CO"],
  summary: {},
  topics: ["photo-z"],
  primaryTopic: "photo-z",
  detail: false,
  status: "to_read",
  priority: "normal",
  seenDates: ["2026-06-13"],
  dailyReports: [],
  discoveryProvenanceByReport: {},
  paperPath: "",
  arxivUrl: `https://arxiv.org/abs/${id}`,
  pdfUrl: `https://arxiv.org/pdf/${id}`,
  pdfPath: "",
  zoteroKey: "",
  zoteroUri: "",
  citationKey: "",
  projects: [],
  ...overrides,
});
const document = {
  schemaVersion: 5,
  updatedAt: "2026-06-13T00:00:00.000Z",
  papers: {
    "arxiv:2606.00001": entry("2606.00001"),
    "arxiv:2606.00002": entry("2606.00002"),
  },
};

const testVault = createTestVault({ index: document });
try {
  const store = createPaperIndexStore(testVault.vault, testVault.storeOptions);

  // What Obsidian (ObsidianStorageAdapter) and the CLI (NodeStorageAdapter) do:
  // NodeFileLock rooted at the vault folder, keyed by the Paper Index path.
  const hostKey = "paper-index:arxiv-daily/.index/papers.json";
  const peer = new NodeFileLock(testVault.root, { lockRoot: testVault.lockRoot });
  const held = await peer.acquire(hostKey, { wait: true });
  assert(held, "the other host should take the lock first");

  const pending = updatePaperStatus(store, "arxiv:2606.00001", "reading");
  let settled = false;
  pending.then(() => { settled = true; }, () => { settled = true; });
  await new Promise((resolve) => setTimeout(resolve, 400));
  assert.equal(settled, false, "the extension must wait while another host holds the lock");
  assert.equal(testVault.readIndex().papers["arxiv:2606.00001"].status, "to_read");

  // The other host edits a different field while it holds the lock. A write
  // that raced it (read before, write after) would erase this change.
  const edited = structuredClone(document);
  edited.papers["arxiv:2606.00002"].priority = "high";
  writeFileSync(testVault.indexFile, JSON.stringify(edited, null, 2));
  await held.release();

  await pending;
  const result = testVault.readIndex();
  assert.equal(result.papers["arxiv:2606.00001"].status, "reading", "the extension's change is kept");
  assert.equal(result.papers["arxiv:2606.00002"].priority, "high", "the other host's change is kept");

  // Same lock namespace: the directory name is derived from the vault's real
  // path and the key, and the extension's acquisition is a later generation
  // in the directory the other host used.
  const rootIdentity = process.platform === "win32"
    ? realpathSync(testVault.root).toLowerCase()
    : realpathSync(testVault.root);
  const lockName = createHash("sha256").update(`${rootIdentity}\0${hostKey}`).digest("hex");
  assert.deepEqual(readdirSync(testVault.lockRoot), [lockName], "both hosts must use one lock directory");
  const generations = readdirSync(path.join(testVault.lockRoot, lockName))
    .filter((name) => /^g\d+\.owner\.json$/.test(name));
  assert(generations.length >= 2, "the extension took its own generation of the same lock");
  const owners = generations.map((name) =>
    JSON.parse(readFileSync(path.join(testVault.lockRoot, lockName, name), "utf8")).pid);
  assert.deepEqual(new Set(owners), new Set([process.pid]));
} finally {
  testVault.cleanup();
  rmSync(peerDir, { recursive: true, force: true });
}

console.log("arXiv Daily VS Code lock interop OK");
