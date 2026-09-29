import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { createRequire } from "node:module";
import { gunzipSync } from "node:zlib";
import type { NativeStorageBinding } from "./native-private-storage";

export interface NativeAsset { sha256: string; gzip: string; }
export type NativeAssets = Record<string, NativeAsset>;
declare const __ARXIV_DAILY_NATIVE_ASSETS__: NativeAssets;

/** Assets come from the product build, never the Vault, CWD, or a runtime download. */
export function loadNativeStorage(
  assets: NativeAssets,
  options: { cacheRoot?: string } = {},
): NativeStorageBinding | undefined {
  const asset = assets[`${process.platform}-${process.arch}`];
  if (!asset) return undefined;
  if (!/^[0-9a-f]{64}$/.test(asset.sha256) || typeof asset.gzip !== "string") {
    throw new Error("native storage asset integrity metadata is invalid");
  }
  const bytes = gunzipSync(Buffer.from(asset.gzip, "base64"), { maxOutputLength: 8 * 1024 * 1024 });
  if (digest(bytes) !== asset.sha256) throw new Error("native storage asset digest mismatch");
  const root = options.cacheRoot ?? path.join(os.homedir(), ".arxiv-daily", "native");
  privateDirectory(root);
  const directory = path.join(root, asset.sha256);
  privateDirectory(directory);
  const target = path.join(directory, "private-storage.node");
  if (!fs.existsSync(target)) {
    const temporary = path.join(directory, `${randomUUID()}.tmp`);
    try {
      const handle = fs.openSync(temporary, "wx", 0o500);
      try { fs.writeFileSync(handle, bytes); fs.fsyncSync(handle); }
      finally { fs.closeSync(handle); }
      try { fs.linkSync(temporary, target); }
      catch (error) { if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error; }
    } finally {
      fs.rmSync(temporary, { force: true });
    }
  }
  const info = fs.lstatSync(target);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("unsafe native storage cache file");
  if (digest(fs.readFileSync(target)) !== asset.sha256) throw new Error("native storage cache digest mismatch");
  const binding = createRequire(path.join(directory, "loader.cjs"))(target) as NativeStorageBinding;
  if (binding.version !== 1 || typeof binding.openDirectory !== "function") {
    throw new Error("native storage API version is incompatible");
  }
  return binding;
}

let initialized = false;
let binding: NativeStorageBinding | undefined;
export function getNativeStorageBinding(): NativeStorageBinding | undefined {
  if (!initialized) {
    binding = loadNativeStorage(typeof __ARXIV_DAILY_NATIVE_ASSETS__ === "undefined" ? {} : __ARXIV_DAILY_NATIVE_ASSETS__);
    initialized = true;
  }
  return binding;
}

function digest(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function privateDirectory(directory: string): void {
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
  const info = fs.lstatSync(directory);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("unsafe native storage cache directory");
  if (process.platform !== "win32" && ((info.mode & 0o077) !== 0 || info.uid !== process.getuid?.())) {
    throw new Error("native storage cache directory is not private");
  }
}
