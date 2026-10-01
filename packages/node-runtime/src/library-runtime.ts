export interface PrepareLibraryRuntimeOptions {
  cacheDir: string;
  localEmbedding: boolean;
  signal?: AbortSignal;
  install?: (directory: string, signal?: AbortSignal) => Promise<void>;
}

export function nodeLibraryRuntimePaths(cacheDir: string) {
  return {
    pdf: path.resolve(cacheDir, "runtimes", "pdfjs-5.4.624"),
    embedding: path.resolve(cacheDir, "runtimes", "transformers-4.2.0-cpu"),
  };
}

/** Explicit setup: installs only pinned runtime dependencies, never user paper content. */
export async function prepareNodeLibraryRuntime(options: PrepareLibraryRuntimeOptions): Promise<{ pdf: string; embedding?: string }> {
  const paths = nodeLibraryRuntimePaths(options.cacheDir);
  await prepare(paths.pdf, pdfPackage, pdfLock, options);
  if (options.localEmbedding) await prepare(paths.embedding, embeddingPackage, embeddingLock, options);
  return { pdf: paths.pdf, ...(options.localEmbedding ? { embedding: paths.embedding } : {}) };
}

async function prepare(directory: string, manifest: PackageManifest, lockfile: unknown, options: PrepareLibraryRuntimeOptions) {
  options.signal?.throwIfAborted();
  await fs.mkdir(directory, { recursive: true, mode: 0o700 });
  if ((await fs.lstat(directory)).isSymbolicLink()) throw new Error("Library runtime directory cannot be a symbolic link");
  const lease = await new NodeFileLock(directory, { lockRoot: path.resolve(options.cacheDir, "runtime-locks") })
    .acquire("prepare", { wait: true, signal: options.signal, timeoutMs: 300_000 });
  if (!lease) throw new Error("Library runtime preparation is busy");
  try {
    const fingerprint = createHash("sha256").update(JSON.stringify({ manifest, lockfile })).digest("hex");
    const marker = path.join(directory, ".prepared");
    const current = await fs.readFile(marker, "utf8").catch(() => "");
    if (current === fingerprint && await installed(directory, manifest)) return;
    await fs.writeFile(path.join(directory, "package.json"), JSON.stringify(manifest, null, 2), { mode: 0o600 });
    await fs.writeFile(path.join(directory, "package-lock.json"), JSON.stringify(lockfile, null, 2), { mode: 0o600 });
    await (options.install ?? installDependencies)(directory, options.signal);
    options.signal?.throwIfAborted();
    if (!await installed(directory, manifest)) throw new Error("Library runtime installation is incomplete; required packages are missing");
    await fs.writeFile(marker, fingerprint, { mode: 0o600 });
  } finally { await lease.release(); }
}

interface PackageManifest { dependencies: Record<string, string> }
async function installed(directory: string, manifest: PackageManifest): Promise<boolean> {
  for (const [name, version] of Object.entries(manifest.dependencies)) {
    try {
      const file = path.join(directory, "node_modules", name, "package.json");
      const value = JSON.parse(await fs.readFile(file, "utf8"));
      if (value.name !== name || value.version !== version) return false;
    } catch { return false; }
  }
  return true;
}

async function installDependencies(directory: string, signal?: AbortSignal): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    let diagnostics = "";
    const child = spawn(process.platform === "win32" ? "npm.cmd" : "npm", [
      "ci", "--ignore-scripts", "--no-audit", "--no-fund", "--workspaces=false",
    ], { cwd: directory, stdio: ["ignore", "ignore", "pipe"], signal, shell: process.platform === "win32" });
    child.stderr?.on("data", chunk => { diagnostics = (diagnostics + String(chunk)).slice(-2000); });
    child.once("error", reject);
    child.once("close", code => code === 0 ? resolve() : reject(new Error(`Library runtime install failed (${code}): ${diagnostics}`)));
  });
}

async function requireInstalled(directory: string, manifest: PackageManifest) {
  if (!await installed(directory, manifest)) throw new Error("Library runtime is not prepared. Run: arxiv-daily library prepare");
  return createRequire(path.join(directory, "package.json"));
}

export async function createNodeLibraryDocumentParser(cacheDir: string): Promise<PdfJsDocumentParser> {
  const { pdf } = nodeLibraryRuntimePaths(cacheDir);
  const require = await requireInstalled(pdf, pdfPackage);
  const root = path.dirname(require.resolve("pdfjs-dist/package.json"));
  const module = await import(pathToFileURL(require.resolve("pdfjs-dist/legacy/build/pdf.mjs")).href) as PdfJsLib;
  return new PdfJsDocumentParser(module, {
    provenance: { id: "node-pdfjs", version: "5.4.624" },
    cMapUrl: path.join(root, "cmaps") + path.sep,
    cMapPacked: true,
    standardFontDataUrl: path.join(root, "standard_fonts") + path.sep,
  });
}

export function createNodeLibraryEmbeddingModel(cacheDir: string, options: { signal?: AbortSignal; offline?: boolean } = {}) {
  return createTransformersEmbeddingModelFromLoader({
    signal: options.signal,
    async loadPipeline() {
      const { embedding } = nodeLibraryRuntimePaths(cacheDir);
      const require = await requireInstalled(embedding, embeddingPackage);
      const imported = await import(pathToFileURL(require.resolve("@huggingface/transformers")).href);
      const module = (imported.default ?? imported) as {
        env: { cacheDir: string; allowRemoteModels: boolean };
        pipeline(task: string, model: string, options: Record<string, unknown>): Promise<TransformersFeatureExtractor>;
      };
      module.env.cacheDir = path.resolve(cacheDir, "models");
      module.env.allowRemoteModels = !options.offline;
      return module.pipeline("feature-extraction", LOCAL_EMBEDDING_MODEL_REPO, { dtype: "q8", device: "cpu" });
    },
  });
}
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { spawn } from "node:child_process";
import {
  PdfJsDocumentParser, LOCAL_EMBEDDING_MODEL_REPO,
  createTransformersEmbeddingModelFromLoader,
  type PdfJsLib, type TransformersFeatureExtractor,
} from "@arxiv-daily/core";
import { NodeFileLock } from "./file-lock";
import pdfPackage from "../../../tools/node-library-runtime/pdf/package.json";
import pdfLock from "../../../tools/node-library-runtime/pdf/package-lock.json";
import embeddingPackage from "../../../tools/node-library-runtime/embedding/package.json";
import embeddingLock from "../../../tools/node-library-runtime/embedding/package-lock.json";
