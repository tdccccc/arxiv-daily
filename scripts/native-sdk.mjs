import { lstat, mkdir, mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";
import { nativeRoot, sha256 } from "./native-build.mjs";

export function sdkDownloadPlan({ version = process.versions.node, platform = process.platform, arch = process.arch } = {}) {
  if (!/^\d+\.\d+\.\d+$/.test(version)) throw new Error("An exact Node SDK version is required");
  if (!["x64", "arm64"].includes(arch)) throw new Error("Unsupported native SDK architecture");
  if (!["linux", "darwin", "win32"].includes(platform)) throw new Error("Unsupported native SDK platform");
  return {
    version,
    base: `https://nodejs.org/download/release/v${version}/`,
    files: [`node-v${version}-headers.tar.gz`, ...(platform === "win32" ? [`win-${arch}/node.lib`] : [])],
  };
}

export async function downloadSdkFile(url, checksum, fetchImpl = fetch) {
  if (new URL(url).origin !== "https://nodejs.org") throw new Error("Native SDK source origin is not allowed");
  if (!/^[0-9a-f]{64}$/.test(checksum)) throw new Error("Native SDK checksum is invalid");
  const response = await fetchImpl(url, { redirect: "error", signal: AbortSignal.timeout(60_000) });
  if (!response.ok) throw new Error(`Native SDK download failed: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length > 64 * 1024 * 1024 || sha256(bytes) !== checksum) throw new Error("Native SDK digest mismatch");
  return bytes;
}

/** Build-time headers/import library only; this never installs or runs package scripts. */
export async function prepareNativeSdk({
  version = process.versions.node, platform = process.platform, arch = process.arch,
  directory = join(nativeRoot, "sdk"), fetchImpl = fetch,
  extractArchive = async (archive, target) => { execFileSync("tar", ["-xzf", archive, "-C", target], { stdio: "inherit" }); },
} = {}) {
  const plan = sdkDownloadPlan({ version, platform, arch });
  const response = await fetchImpl(`${plan.base}SHASUMS256.txt`, { redirect: "error", signal: AbortSignal.timeout(60_000) });
  if (!response.ok) throw new Error(`Native SDK checksum list failed: HTTP ${response.status}`);
  const checksums = new Map();
  for (const line of (await response.text()).split(/\r?\n/)) {
    const match = /^([0-9a-f]{64})\s+\*?(.+)$/.exec(line.trim());
    if (match) checksums.set(match[2], match[1]);
  }
  for (const file of plan.files) if (!checksums.has(file)) throw new Error(`Native SDK checksum is missing: ${file}`);
  const target = join(directory, `node-v${version}`);
  const files = Object.fromEntries(plan.files.map(file => [file, checksums.get(file)]));
  try {
    const existing = JSON.parse(await readFile(join(target, "sdk-receipt.json"), "utf8"));
    if (existing.version !== version || plan.files.some(file => existing.files?.[file] !== files[file])) throw new Error("Existing native SDK receipt does not match the requested inputs");
    if (!(await lstat(join(target, "include/node/node_api.h"))).isFile()) throw new Error("Native SDK header is unsafe");
    if (platform === "win32") {
      const library = await readFile(join(target, `win-${arch}/node.lib`));
      if (sha256(library) !== files[`win-${arch}/node.lib`]) throw new Error("Cached native SDK library digest mismatch");
    }
    return target;
  } catch (error) { if (error.code !== "ENOENT") throw error; }
  if (await lstat(target).catch(error => { if (error.code === "ENOENT") return null; throw error; })) {
    throw new Error("Refusing to replace an incomplete or unrecognized native SDK directory");
  }
  await mkdir(directory, { recursive: true });
  const temporary = await mkdtemp(join(directory, "prepare-"));
  try {
    const archive = join(temporary, "headers.tar.gz");
    await writeFile(archive, await downloadSdkFile(`${plan.base}${plan.files[0]}`, files[plan.files[0]], fetchImpl));
    await extractArchive(archive, temporary);
    const prepared = join(temporary, `node-v${version}`);
    if (!(await lstat(join(prepared, "include/node/node_api.h"))).isFile()) throw new Error("Native SDK headers were not extracted safely");
    if (platform === "win32") {
      const library = `win-${arch}/node.lib`;
      const bytes = await downloadSdkFile(`${plan.base}${library}`, files[library], fetchImpl);
      await mkdir(join(prepared, `win-${arch}`), { recursive: true });
      await writeFile(join(prepared, library), bytes);
    }
    await writeFile(join(prepared, "sdk-receipt.json"), `${JSON.stringify({ version, files }, null, 2)}\n`);
    await rename(prepared, target);
    return target;
  } finally { await rm(temporary, { recursive: true, force: true }); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  console.log(`Official native SDK prepared: ${await prepareNativeSdk()}`);
}
