import { afterEach, describe, expect, it } from "vitest";
import { mkdir, mkdtemp, readFile, rename, rm, stat, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createRequire } from "node:module";
import { NativePrivateStorage, type NativeStorageBinding } from "../src/native-private-storage";

const binding = createRequire(import.meta.url)("../native/build/Release/private_storage.node") as NativeStorageBinding;
const roots: string[] = [];
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "arxiv-native-private-"));
  roots.push(root);
  return root;
}
afterEach(async () => { while (roots.length) await rm(roots.pop()!, { recursive: true, force: true }); });

function assertPrivate(root: string, name: string) {
  const parts = name.split("/");
  const leaf = parts.pop()!;
  const dir = binding.openDirectory(root, parts.join("/"), false);
  try {
    const file = dir.openFile(leaf);
    expect(file).not.toBeNull();
    try { expect(file!.isPrivate()).toBe(true); } finally { file!.close(); }
  } finally { dir.close(); }
}

describe("native private storage", () => {
  it("publishes exactly one complete private exclusive file", async () => {
    const root = await fixture();
    const first = new NativePrivateStorage(root, binding);
    const second = new NativePrivateStorage(root, binding);
    const results = await Promise.all([
      first.createTextExclusive("claims/send.json", "first"),
      second.createTextExclusive("claims/send.json", "second"),
    ]);
    expect(results.filter(Boolean)).toHaveLength(1);
    expect(await readFile(join(root, "claims/send.json"), "utf8")).toBe(results[0] ? "first" : "second");
    assertPrivate(root, "claims/send.json");
  });

  it.each(["before", "after"])("rejects a parent swap %s file creation without leaving a claim", async (when) => {
    const root = await fixture();
    const outside = await fixture();
    const move = async () => {
      await rename(join(root, "claims"), join(outside, "moved"));
      await symlink(join(outside, "moved"), join(root, "claims"), process.platform === "win32" ? "junction" : "dir");
    };
    const storage = new NativePrivateStorage(root, binding, {
      ...(when === "before" ? { afterFinalParentOpened: move } : { afterTargetCreated: move }),
    });
    await expect(storage.createTextExclusive("claims/daily/send.json", "must not escape")).rejects.toThrow();
    expect(await readFile(join(outside, "moved/daily/send.json"), "utf8").catch(() => null)).toBeNull();
    expect(await readFile(join(root, "claims/daily/send.json"), "utf8").catch(() => null)).toBeNull();
  });

  it("keeps the old primary present until replacement and makes every artifact private", async () => {
    const root = await fixture();
    const artifacts: string[] = [];
    const storage = new NativePrivateStorage(root, binding, {
      afterTemporaryFileReady: async path => {
        artifacts.push(path);
        expect(await readFile(join(root, "state/delivery.json"), "utf8")).toBe("old");
        assertPrivate(root, path.slice(root.length + 1).replace(/\\/g, "/"));
      },
      afterBackupFileReady: async path => {
        artifacts.push(path);
        expect(await readFile(join(root, "state/delivery.json"), "utf8")).toBe("old");
        assertPrivate(root, path.slice(root.length + 1).replace(/\\/g, "/"));
      },
    });
    await mkdir(join(root, "state"));
    await writeFile(join(root, "state/delivery.json"), "old", { mode: 0o666 });
    await storage.writeTextAtomic("state/delivery.json", "new", 0o600);
    expect(artifacts).toHaveLength(2);
    expect(await readFile(join(root, "state/delivery.json"), "utf8")).toBe("new");
    assertPrivate(root, "state/delivery.json");
    expect(await Promise.all(artifacts.map(path => stat(path).catch(() => null)))).toEqual([null, null]);
  });

  it("does not replace the primary if the namespace changes after backup creation", async () => {
    const root = await fixture();
    const outside = await fixture();
    await mkdir(join(root, "state"));
    await writeFile(join(root, "state/delivery.json"), "old");
    const storage = new NativePrivateStorage(root, binding, {
      afterBackupFileReady: async () => {
        await rename(join(root, "state"), join(outside, "moved"));
        await symlink(join(outside, "moved"), join(root, "state"), process.platform === "win32" ? "junction" : "dir");
      },
    });
    await expect(storage.writeTextAtomic("state/delivery.json", "new", 0o600)).rejects.toThrow();
    const parent = await stat(join(outside, "moved")).then(() => join(outside, "moved"), () => join(root, "state"));
    expect(await readFile(join(parent, "delivery.json"), "utf8")).toBe("old");
  });

  it("recovers an authoritative backup, never promotes a lone temporary, and tightens privacy", async () => {
    const root = await fixture();
    await mkdir(join(root, "state"));
    await writeFile(join(root, "state/delivery.json.bak-old"), "not a recognized artifact");
    await writeFile(join(root, "state/delivery.json.bak"), "delivered", { mode: 0o666 });
    await writeFile(join(root, "state/delivery.json.tmp-deadbeef"), "uncommitted", { mode: 0o666 });
    const storage = new NativePrivateStorage(root, binding);
    await storage.recoverTextAtomic("state/delivery.json", 0o600);
    expect(await readFile(join(root, "state/delivery.json"), "utf8")).toBe("delivered");
    assertPrivate(root, "state/delivery.json");
    expect(await stat(join(root, "state/delivery.json.bak")).catch(() => null)).toBeNull();
    expect(await stat(join(root, "state/delivery.json.tmp-deadbeef")).catch(() => null)).toBeNull();
    expect(await readFile(join(root, "state/delivery.json.bak-old"), "utf8")).toBe("not a recognized artifact");
    await writeFile(join(root, "state/missing.json.tmp"), "future");
    await storage.recoverTextAtomic("state/missing.json", 0o600);
    expect(await stat(join(root, "state/missing.json")).catch(() => null)).toBeNull();
  });

  it("keeps the old primary if atomic rename fails and does not fall back to copy", async () => {
    const root = await fixture();
    await mkdir(join(root, "state"));
    await writeFile(join(root, "state/delivery.json"), "old");
    const storage = new NativePrivateStorage(root, binding, {
      renameAtomic: async () => { throw Object.assign(new Error("EXDEV"), { code: "EXDEV" }); },
    });
    await expect(storage.writeTextAtomic("state/delivery.json", "new", 0o600)).rejects.toThrow("EXDEV");
    expect(await readFile(join(root, "state/delivery.json"), "utf8")).toBe("old");
  });

  it("rejects a replaced claim namespace or prevents its move, and rejects a released guard", async () => {
    const root = await fixture();
    const outside = await fixture();
    const storage = new NativePrivateStorage(root, binding);
    await mkdir(join(root, "claims"));
    await storage.createTextExclusive("claims/send.json", "claim");
    const guard = await storage.guardClaimNamespace("claims/send.json");
    try {
      guard.assertCurrent();
      let moved = false;
      try {
        await rename(join(root, "claims"), join(outside, "moved"));
        moved = true;
      } catch (error) {
        if (process.platform !== "win32" || !["EPERM", "EACCES", "EBUSY"].includes((error as NodeJS.ErrnoException).code ?? "")) throw error;
      }
      if (moved) {
        await mkdir(join(root, "claims"));
        expect(() => guard.assertCurrent()).toThrow(/replaced|changed/i);
        expect(await readFile(join(outside, "moved/send.json"), "utf8")).toBe("claim");
      } else {
        guard.assertCurrent();
        expect(await readFile(join(root, "claims/send.json"), "utf8")).toBe("claim");
      }
    } finally { await guard.release(); }
    expect(() => guard.assertCurrent()).toThrow(/closed|released/i);
  });
});
