import fsPromises from "node:fs/promises";
import path from "node:path";

/**
 * A settings store shaped the way it was before the sidecar existed. Loading it
 * is what gives the migration something to prove.
 */
export function legacySettingsFixture() {
  return {
    settings: {
      llm: { provider: "openai", model: "gpt-4o-mini", apiKey: "" },
      arxiv: { categories: ["astro-ph.GA"], timezone: "UTC" },
      output: { dailyDir: "arxiv-daily/daily", papersDir: "arxiv-daily/papers" },
      schedule: { enabled: false },
      advanced: {},
      email: {},
      detailSelection: {},
    },
  };
}

/**
 * A settings store whose personal library is connected to `libraryRoot` but not
 * authorized, with remote embedding configured but not selected.
 *
 * That is the state the library settings page is worth looking at in: the row
 * has a folder to name, a next step to offer, and a remote switch that must ask
 * before anything leaves the device. `rootIdentity` is the folder's real
 * `dev:ino` — the plugin re-stats the root before reading it, so a made-up
 * identity would make the connection unusable rather than merely untidy.
 */
export function connectedLibraryFixture({ libraryRoot, rootIdentity, embeddingMode = "local" }) {
  if (typeof libraryRoot !== "string" || !path.isAbsolute(libraryRoot)) {
    throw new TypeError(`library root must be absolute: ${String(libraryRoot)}`);
  }
  if (!/^\d+:\d+$/.test(String(rootIdentity))) {
    throw new TypeError(`library root identity must be "dev:ino", received ${JSON.stringify(rootIdentity)}`);
  }
  return {
    settings: {
      llm: {
        provider: "openai",
        model: "gpt-4o-mini",
        apiKey: "",
        baseUrl: "http://127.0.0.1:11434/v1",
      },
      arxiv: { categories: ["astro-ph.GA"], timezone: "UTC" },
      output: { dailyDir: "arxiv-daily/daily", papersDir: "arxiv-daily/papers" },
      schedule: { enabled: false },
      advanced: {},
      email: {},
      detailSelection: {},
      // A configured endpoint is what makes remote consent disclosable: without
      // one there is nothing honest to name, and the modal is skipped.
      embedding: {
        mode: embeddingMode,
        provider: "",
        baseUrl: "http://127.0.0.1:11434/v1",
        apiKey: "",
        model: "nomic-embed-text",
        dimension: 768,
        initialChoiceDone: true,
      },
    },
    libraryConnection: {
      schemaVersion: 1,
      selectedRoot: libraryRoot,
      rootIdentity,
      eligibleExtensions: [".pdf"],
      processingDepth: "metadata-and-abstracts",
    },
  };
}

/**
 * The plugin re-stats the library root and refuses it unless the device and
 * inode still match, so the fixture has to carry the folder's real identity.
 */
export async function readRootIdentity(root, fs = fsPromises) {
  const info = await fs.stat(root);
  if (!info.isDirectory()) throw new Error(`library root is not a directory: ${root}`);
  return `${info.dev}:${info.ino}`;
}

/**
 * A folder of PDFs inside the test vault to point the library at. Staying
 * inside the vault matters: the harness must not open, enumerate or reference
 * anything of the user's outside the vault it was given.
 */
export async function resolveLibraryRoot({ vaultPath, fs = fsPromises }) {
  const entries = await fs.readdir(vaultPath, { withFileTypes: true });
  const candidates = entries
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
    .map((entry) => path.join(vaultPath, entry.name));
  for (const candidate of candidates) {
    const files = await fs.readdir(candidate);
    if (files.some((name) => name.toLowerCase().endsWith(".pdf"))) return candidate;
  }
  throw new Error(
    `no folder of PDFs found in ${vaultPath}; the personal library settings page needs one to describe`,
  );
}

export async function installSettingsFixture({ vaultPath, pluginId, fs = fsPromises, data }) {
  if (typeof vaultPath !== "string" || !path.isAbsolute(vaultPath)) {
    throw new TypeError(`vault path must be absolute: ${String(vaultPath)}`);
  }
  const target = path.join(vaultPath, ".obsidian", "plugins", pluginId, "data.json");
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, JSON.stringify(data, null, 2));
  return target;
}
