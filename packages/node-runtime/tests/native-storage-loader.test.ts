import { afterEach, describe, expect, it } from "vitest";
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";
import { loadNativeStorage, type NativeAssets } from "../src/native-storage-loader";

const roots: string[] = [];
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "arxiv-native-loader-"));
  roots.push(root);
  return root;
}
async function asset() {
  const bytes = await readFile(new URL("../native/build/Release/private_storage.node", import.meta.url));
  return { sha256: createHash("sha256").update(bytes).digest("hex"), gzip: gzipSync(bytes).toString("base64") };
}
const key = `${process.platform}-${process.arch}`;
afterEach(async () => {
  while (roots.length) {
    await rm(roots.pop()!, { recursive: true, force: true }).catch((error: NodeJS.ErrnoException) => {
      // Windows keeps loaded DLLs open until this test worker exits.
      if (process.platform !== "win32" || !["EPERM", "EACCES", "EBUSY"].includes(error.code ?? "")) throw error;
    });
  }
});

describe("bundled native storage loader", () => {
  it("leaves absent platform assets unavailable without downloading or writing code", async () => {
    const root = await fixture();
    expect(loadNativeStorage({}, { cacheRoot: root })).toBeUndefined();
  });

  it("loads only verified bundled bytes and reuses the content-addressed module", async () => {
    const root = await fixture();
    const record = await asset();
    const assets: NativeAssets = { [key]: record };
    const first = loadNativeStorage(assets, { cacheRoot: root });
    expect(first?.version).toBe(1);
    expect(first?.openDirectory).toBeTypeOf("function");
    expect(loadNativeStorage(assets, { cacheRoot: root })).toBe(first);
    const bytes = await readFile(join(root, record.sha256, "private-storage.node"));
    expect(createHash("sha256").update(bytes).digest("hex")).toBe(record.sha256);
  });

  it("rejects a payload digest mismatch before extracting or loading code", async () => {
    const root = await fixture();
    const record = await asset();
    expect(() => loadNativeStorage({ [key]: { ...record, sha256: "0".repeat(64) } }, { cacheRoot: root }))
      .toThrow(/integrity|digest/i);
  });

  it("refuses to replace a corrupt cached binary", async () => {
    const root = await fixture();
    const record = await asset();
    const dir = join(root, record.sha256);
    await mkdir(dir, { mode: 0o700 });
    const target = join(dir, "private-storage.node");
    await writeFile(target, "do not execute or overwrite");
    expect(() => loadNativeStorage({ [key]: record }, { cacheRoot: root })).toThrow(/integrity|digest/i);
    expect(await readFile(target, "utf8")).toBe("do not execute or overwrite");
  });

  it("refuses a linked cache directory even if its bytes would be valid", async () => {
    const root = await fixture();
    const outside = await fixture();
    const record = await asset();
    await symlink(outside, join(root, record.sha256), process.platform === "win32" ? "junction" : "dir");
    expect(() => loadNativeStorage({ [key]: record }, { cacheRoot: root })).toThrow(/unsafe|link/i);
  });
});
