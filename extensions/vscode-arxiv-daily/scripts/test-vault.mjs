import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

export const INDEX_PATH = "arxiv-daily/.index/papers.json";

/**
 * A real vault folder in the OS temp directory, plus a private lock directory
 * so the tests never touch the user's ~/.arxiv-daily/host-locks. The paper index
 * goes through core's Node storage, so it has to exist on disk.
 */
export function createTestVault({ index, backup } = {}) {
  const base = mkdtempSync(path.join(tmpdir(), "arxiv-daily-vscode-"));
  const root = path.join(base, "vault");
  const lockRoot = path.join(base, "locks");
  mkdirSync(path.join(root, "arxiv-daily", ".index"), { recursive: true });
  mkdirSync(path.join(root, "arxiv-daily", "daily"), { recursive: true });
  mkdirSync(path.join(root, "arxiv-daily", "papers"), { recursive: true });
  const indexFile = path.join(root, ...INDEX_PATH.split("/"));
  if (index !== undefined) {
    writeFileSync(indexFile, typeof index === "string" ? index : JSON.stringify(index, null, 2));
  }
  if (backup !== undefined) writeFileSync(`${indexFile}.bak`, JSON.stringify(backup, null, 2));
  return {
    base,
    root,
    lockRoot,
    indexFile,
    vault: { vaultRootUri: { fsPath: root, path: root, scheme: "file" } },
    storeOptions: { lockRoot, onStaleOwner() {} },
    readIndex: () => JSON.parse(readFileSync(indexFile, "utf8")),
    readIndexText: () => readFileSync(indexFile, "utf8"),
    cleanup: () => rmSync(base, { recursive: true, force: true }),
  };
}
