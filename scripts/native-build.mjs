import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";

export const nativeRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../packages/node-runtime/native");
export const nativeSourceFiles = ["CMakeLists.txt", "private-storage.cc", "win-delay-load.cc"];
export function nativeSourceHash() {
  const hash = createHash("sha256");
  for (const name of nativeSourceFiles) {
    // Git may check out identical C++/CMake source with CRLF on Windows.
    // Canonicalize line endings; binary asset digests still cover exact bytes.
    const source = readFileSync(join(nativeRoot, name), "utf8").replace(/\r\n/g, "\n");
    hash.update(name).update("\0").update(source).update("\0");
  }
  return hash.digest("hex");
}
export function sha256(bytes) { return createHash("sha256").update(bytes).digest("hex"); }

/** Reproduces the manually verified CMake build using installed tools only. */
export function buildNative({ buildDir = join(nativeRoot, "build"), force = false, stdio = "inherit" } = {}) {
  if (!["linux", "darwin", "win32"].includes(process.platform) || !["x64", "arm64"].includes(process.arch)) {
    throw new Error("Native storage build requires a supported native OS/architecture");
  }
  const target = `${process.platform}-${process.arch}`;
  const sourceHash = nativeSourceHash();
  const bindingPath = join(buildDir, "Release", "private_storage.node");
  const metadataPath = join(buildDir, "native-build.json");
  if (!force && existsSync(bindingPath) && existsSync(metadataPath)) {
    const metadata = JSON.parse(readFileSync(metadataPath, "utf8"));
    if (metadata.target === target && metadata.sourceHash === sourceHash && metadata.sha256 === sha256(readFileSync(bindingPath))) {
      return { ...metadata, bindingPath };
    }
  }
  const sdk = join(nativeRoot, "sdk", `node-v${process.versions.node}`);
  const candidates = [process.env.NODE_INCLUDE_DIR, join(sdk, "include", "node"),
    resolve(dirname(process.execPath), "../include/node"), "/usr/include/node"];
  const include = candidates.find(directory => directory && existsSync(join(directory, "node_api.h")));
  if (!include) throw new Error("Node-API headers not found; set NODE_INCLUDE_DIR or prepare the official native SDK");
  const args = ["-S", nativeRoot, "-B", buildDir, "-DCMAKE_BUILD_TYPE=Release", `-DNODE_INCLUDE_DIR=${include}`];
  if (process.platform === "win32") {
    const library = process.env.NODE_LIBRARY ?? join(sdk, `win-${process.arch}`, "node.lib");
    if (!existsSync(library)) throw new Error("Windows Node import library not found; set NODE_LIBRARY or prepare the official native SDK");
    args.push(`-DNODE_LIBRARY=${library}`, "-A", process.arch === "arm64" ? "ARM64" : "x64");
  } else if (process.platform === "darwin") {
    args.push(`-DCMAKE_OSX_ARCHITECTURES=${process.arch === "arm64" ? "arm64" : "x86_64"}`);
  }
  execFileSync("cmake", args, { stdio });
  execFileSync("cmake", ["--build", buildDir, "--config", "Release"], { stdio });
  const metadata = { target, sourceHash, sha256: sha256(readFileSync(bindingPath)), nodeApi: 8 };
  mkdirSync(buildDir, { recursive: true });
  writeFileSync(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);
  return { ...metadata, bindingPath };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) buildNative();
