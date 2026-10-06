import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync, existsSync, cpSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { gunzipSync } from "node:zlib";
import { pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";
import { afterEach, test } from "node:test";
import { nativeRoot, nativeSourceFiles, nativeSourceHash, sha256, NativeToolchainUnavailableError } from "../native-build.mjs";
import { readNativeAssets, NATIVE_TARGETS } from "../native-assets.mjs";

const roots = [];
const sourceHash = "a".repeat(64);
function fixture(targets = ["linux-x64"]) {
  const directory = mkdtempSync(join(tmpdir(), "arxiv-native-assets-"));
  roots.push(directory);
  for (const target of targets) {
    const bytes = Buffer.from(`fixture only, not executable: ${target}`);
    writeFileSync(join(directory, `${target}.node`), bytes);
    writeFileSync(join(directory, `${target}.json`), JSON.stringify({ schemaVersion: 1, target, sourceHash, nodeApi: 8, sha256: sha256(bytes) }));
  }
  return directory;
}
afterEach(() => { while (roots.length) rmSync(roots.pop(), { recursive: true, force: true }); });

test("native source identity is stable across LF and CRLF checkouts but detects code changes", async () => {
  const root = mkdtempSync(join(tmpdir(), "arxiv-native-source-"));
  roots.push(root);
  mkdirSync(join(root, "scripts"));
  const source = join(root, "packages/node-runtime/native");
  mkdirSync(source, { recursive: true });
  const module = join(root, "scripts/native-build.mjs");
  writeFileSync(module, readFileSync(resolve("scripts/native-build.mjs")));
  for (const name of nativeSourceFiles) {
    const text = readFileSync(join(nativeRoot, name), "utf8").replace(/\r\n/g, "\n");
    writeFileSync(join(source, name), text.replace(/\n/g, "\r\n"));
  }
  const checkout = await import(pathToFileURL(module).href);
  assert.equal(checkout.nativeSourceHash(), nativeSourceHash());
  writeFileSync(join(source, nativeSourceFiles[0]), "changed build semantics\r\n");
  assert.notEqual(checkout.nativeSourceHash(), nativeSourceHash());
});

test("native release target set covers both architectures on all three desktop systems", () => {
  assert.deepEqual(NATIVE_TARGETS, ["linux-x64", "linux-arm64", "darwin-x64", "darwin-arm64", "win32-x64", "win32-arm64"]);
});

test("bundles only source-matched and digest-checked platform bytes", () => {
  const directory = fixture();
  const assets = readNativeAssets({ directory, targets: ["linux-x64"], sourceHash });
  assert.equal(assets["linux-x64"].sha256, sha256(readFileSync(join(directory, "linux-x64.node"))));
  assert.equal(gunzipSync(Buffer.from(assets["linux-x64"].gzip, "base64")).toString(), "fixture only, not executable: linux-x64");
});

test("release assembly fails when any declared platform is missing", () => {
  assert.throws(() => readNativeAssets({ directory: fixture(), targets: NATIVE_TARGETS, sourceHash }), /missing|platform/i);
});

test("refuses artifacts from another source revision or platform", () => {
  const directory = fixture();
  assert.throws(() => readNativeAssets({ directory, targets: ["linux-x64"], sourceHash: "b".repeat(64) }), /source|revision/i);
  const metadata = JSON.parse(readFileSync(join(directory, "linux-x64.json"), "utf8"));
  metadata.target = "win32-x64";
  writeFileSync(join(directory, "linux-x64.json"), JSON.stringify(metadata));
  assert.throws(() => readNativeAssets({ directory, targets: ["linux-x64"], sourceHash }), /platform|target/i);
});

test("refuses corrupted native bytes and unsupported API metadata", () => {
  const directory = fixture();
  writeFileSync(join(directory, "linux-x64.node"), "corrupted");
  assert.throws(() => readNativeAssets({ directory, targets: ["linux-x64"], sourceHash }), /digest|integrity/i);
  const clean = fixture();
  const metadata = JSON.parse(readFileSync(join(clean, "linux-x64.json"), "utf8"));
  metadata.nodeApi = 999;
  writeFileSync(join(clean, "linux-x64.json"), JSON.stringify(metadata));
  assert.throws(() => readNativeAssets({ directory: clean, targets: ["linux-x64"], sourceHash }), /API|metadata/i);
});

test("a complete fixture matrix assembles every required target", () => {
  const directory = fixture(NATIVE_TARGETS);
  assert.deepEqual(Object.keys(readNativeAssets({ directory, targets: NATIVE_TARGETS, sourceHash })), NATIVE_TARGETS);
});

test("installed-package verification exercises native code without installer scripts or network", () => {
  const source = readFileSync(resolve("scripts/install-smoke.mjs"), "utf8");
  assert.match(source, /smokeNativePackage/);
  assert.match(source, /--ignore-scripts/);
  assert.match(source, /--offline/);
});

test("both product builds embed native assets through the guarded assembler", () => {
  for (const file of ["apps/cli/esbuild.config.mjs", "plugin/esbuild.config.mjs"]) {
    const source = readFileSync(resolve(file), "utf8");
    assert.match(source, /nativeAssetsForBuild/);
    assert.match(source, /__ARXIV_DAILY_NATIVE_ASSETS__/);
  }
});

// A PATH containing only a symlink to the running `node` binary: no `cmake`
// is reachable, but Node-API headers are still found (buildNative resolves
// them relative to the real, resolved `process.execPath`, not argv[0] or
// PATH), so only the cmake lookup itself is made to fail.
function pathWithoutCmake() {
  const directory = mkdtempSync(join(tmpdir(), "arxiv-no-cmake-path-"));
  roots.push(directory);
  symlinkSync(process.execPath, join(directory, "node"));
  return directory;
}

test("buildNative reports a dedicated, catchable error when cmake itself is missing from PATH", () => {
  const buildDir = mkdtempSync(join(tmpdir(), "arxiv-native-build-"));
  roots.push(buildDir);
  const script = `
    import("${pathToFileURL(resolve("scripts/native-build.mjs")).href}").then(({ buildNative, NativeToolchainUnavailableError }) => {
      try {
        buildNative({ buildDir: ${JSON.stringify(buildDir)}, force: true, stdio: "pipe" });
        console.log("ran-without-throwing");
      } catch (error) {
        console.log(error instanceof NativeToolchainUnavailableError ? "NativeToolchainUnavailableError" : error.constructor.name);
      }
    });
  `;
  const output = execFileSync(process.execPath, ["--input-type=module", "-e", script], {
    env: { ...process.env, PATH: pathWithoutCmake() },
    encoding: "utf8",
  });
  assert.equal(output.trim(), "NativeToolchainUnavailableError");
});

test("buildNative still throws a plain error (not the toolchain-missing error) for a real compile failure", () => {
  const buildDir = mkdtempSync(join(tmpdir(), "arxiv-native-build-"));
  roots.push(buildDir);
  const sourceRoot = mkdtempSync(join(tmpdir(), "arxiv-native-source-"));
  roots.push(sourceRoot);
  mkdirSync(join(sourceRoot, "scripts"));
  const brokenRoot = join(sourceRoot, "packages/node-runtime/native");
  mkdirSync(brokenRoot, { recursive: true });
  for (const name of nativeSourceFiles) {
    const text = readFileSync(join(nativeRoot, name), "utf8");
    writeFileSync(join(brokenRoot, name), name === "private-storage.cc" ? `${text}\n#error intentional break for test\n` : text);
  }
  const module = join(sourceRoot, "scripts/native-build.mjs");
  writeFileSync(module, readFileSync(resolve("scripts/native-build.mjs")));
  const script = `
    import(${JSON.stringify(pathToFileURL(module).href)}).then(({ buildNative, NativeToolchainUnavailableError }) => {
      try {
        buildNative({ buildDir: ${JSON.stringify(buildDir)}, force: true, stdio: "pipe" });
        console.log("ran-without-throwing");
      } catch (error) {
        console.log(error instanceof NativeToolchainUnavailableError ? "NativeToolchainUnavailableError" : error.constructor.name);
      }
    });
  `;
  const output = execFileSync(process.execPath, ["--input-type=module", "-e", script], { encoding: "utf8" });
  assert.equal(output.trim(), "Error");
});

test("non-release builds fall back to no embedded native asset (with a warning) when cmake is unavailable", () => {
  const realBuildDir = join(nativeRoot, "build");
  // Park (copy, don't move: tmpdir() may be a different filesystem) the
  // dev-machine's real native/build cache so this test actually exercises
  // the cmake-missing path instead of short-circuiting on a cache hit.
  const parked = existsSync(realBuildDir) ? mkdtempSync(join(tmpdir(), "arxiv-parked-native-build-")) : null;
  if (parked) {
    cpSync(realBuildDir, parked, { recursive: true });
    rmSync(realBuildDir, { recursive: true, force: true });
  }
  try {
    const outputDirectory = mkdtempSync(join(tmpdir(), "arxiv-native-prebuilds-"));
    roots.push(outputDirectory);
    const script = `
      console.warn = (...args) => console.log(...args);
      import(${JSON.stringify(pathToFileURL(resolve("scripts/native-assets.mjs")).href)}).then(({ nativeAssetsForBuild }) => {
        const assets = nativeAssetsForBuild({ release: false, directory: ${JSON.stringify(outputDirectory)} });
        console.log(JSON.stringify(assets));
      });
    `;
    const result = execFileSync(process.execPath, ["--input-type=module", "-e", script], {
      env: { ...process.env, PATH: pathWithoutCmake() },
      encoding: "utf8",
    });
    const lines = result.trim().split("\n");
    assert.match(lines.find(line => line.includes("cmake")) ?? "", /cmake was not found on PATH/);
    assert.deepEqual(JSON.parse(lines.at(-1)), {});
  } finally {
    if (parked) {
      rmSync(realBuildDir, { recursive: true, force: true });
      cpSync(parked, realBuildDir, { recursive: true });
      rmSync(parked, { recursive: true, force: true });
    }
  }
});
