import { lstatSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { randomUUID } from "node:crypto";
import { gzipSync } from "node:zlib";
import { NativeToolchainUnavailableError, buildNative, nativeRoot, nativeSourceHash, sha256 } from "./native-build.mjs";

export const NATIVE_TARGETS = ["linux-x64", "linux-arm64", "darwin-x64", "darwin-arm64", "win32-x64", "win32-arm64"];
const prebuilds = join(nativeRoot, "prebuilds");

export function readNativeAssets({ directory = prebuilds, targets = NATIVE_TARGETS, sourceHash = nativeSourceHash() } = {}) {
  const assets = {};
  for (const target of targets) {
    if (!NATIVE_TARGETS.includes(target)) throw new Error(`Unsupported native target: ${target}`);
    const binary = join(directory, `${target}.node`);
    const metadataPath = join(directory, `${target}.json`);
    let metadata, bytes;
    try {
      for (const file of [binary, metadataPath]) {
        if (!lstatSync(file).isFile()) throw new Error(`Unsafe native artifact for ${target}`);
      }
      metadata = JSON.parse(readFileSync(metadataPath, "utf8"));
      bytes = readFileSync(binary);
    } catch (error) {
      if (error.code === "ENOENT") throw new Error(`Missing native platform artifact: ${target}`);
      throw error;
    }
    if (metadata?.schemaVersion !== 1 || metadata.nodeApi !== 8 || !/^[0-9a-f]{64}$/.test(metadata.sha256)) {
      throw new Error(`Invalid native API metadata: ${target}`);
    }
    if (metadata.target !== target) throw new Error(`Native target mismatch: ${target}`);
    if (metadata.sourceHash !== sourceHash) throw new Error(`Native source revision mismatch: ${target}`);
    if (sha256(bytes) !== metadata.sha256) throw new Error(`Native artifact digest mismatch: ${target}`);
    assets[target] = { sha256: metadata.sha256, gzip: gzipSync(bytes).toString("base64") };
  }
  return assets;
}

export function exportNativeBuild({ directory = prebuilds, expectedTarget } = {}) {
  const result = buildNative();
  if (expectedTarget && expectedTarget !== result.target) throw new Error(`Native runner target mismatch: expected ${expectedTarget}, got ${result.target}`);
  mkdirSync(directory, { recursive: true });
  const metadata = { schemaVersion: 1, target: result.target, sourceHash: result.sourceHash, nodeApi: 8, sha256: result.sha256 };
  const binary = readFileSync(result.bindingPath);
  if (sha256(binary) !== metadata.sha256) throw new Error("Native build changed before export");
  for (const [name, content] of [[`${result.target}.node`, binary], [`${result.target}.json`, `${JSON.stringify(metadata, null, 2)}\n`]]) {
    const temporary = join(directory, `${name}.${randomUUID()}.tmp`);
    writeFileSync(temporary, content);
    renameSync(temporary, join(directory, name));
  }
  return result.target;
}

export function nativeAssetsForBuild({ release = process.env.ARXIV_DAILY_NATIVE_RELEASE === "1", directory = prebuilds } = {}) {
  if (release) return readNativeAssets({ directory, targets: NATIVE_TARGETS });
  let target;
  try {
    target = exportNativeBuild({ directory });
  } catch (error) {
    if (!(error instanceof NativeToolchainUnavailableError)) throw error;
    // Development/CI builds without a C++ toolchain still need to produce a
    // usable bundle: ship it with no embedded native asset rather than fail.
    // Runtime (native-storage-loader.ts / storage-adapter.ts) already treats
    // a missing platform asset as "no native backend" and falls back to the
    // Linux descriptor-anchored backend or fails closed per ADR 0009 — it
    // never silently degrades to an unsynced, backup-less write. Release
    // builds (ARXIV_DAILY_NATIVE_RELEASE=1) are unaffected: they always read
    // the full verified prebuilt matrix above and fail hard if it is missing.
    console.warn(`[arxiv-daily] ${error.message} Building without an embedded native storage backend for this platform: private writes will use the Linux fallback or fail closed (ADR 0009). Install CMake and a C++ toolchain to include it.`);
    return {};
  }
  return readNativeAssets({ directory, targets: [target] });
}

export function nativeAssetsForTests() {
  if (process.env.ARXIV_DAILY_TEST_NATIVE !== "1") return {};
  const result = buildNative();
  const bytes = readFileSync(result.bindingPath);
  return { [result.target]: { sha256: result.sha256, gzip: gzipSync(bytes).toString("base64") } };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  if (process.argv[2] === "export") {
    console.log(`Native asset exported: ${exportNativeBuild({ expectedTarget: process.argv[3] })}`);
  } else if (process.argv[2] === "verify") {
    readNativeAssets();
    console.log("Complete native release matrix verified");
  } else {
    throw new Error("Usage: node scripts/native-assets.mjs export [expected-target] | verify");
  }
}
