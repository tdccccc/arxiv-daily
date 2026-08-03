import { afterEach, describe, expect, it } from "vitest";
import { mkdtemp, mkdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { isCancellationError } from "@arxiv-daily/core";
import { openScopedLibrarySource } from "../src/scoped-library-source";

const tempDirs: string[] = [];

afterEach(async () => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir) await rm(dir, { recursive: true, force: true });
  }
});

async function makeTempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "arxiv-daily-library-"));
  tempDirs.push(dir);
  return dir;
}

describe("openScopedLibrarySource", () => {
  it("inventories and reads files without exposing mutation methods", async () => {
    const root = await makeTempDir();
    await mkdir(join(root, "papers"));
    await writeFile(join(root, "papers", "example.pdf"), new Uint8Array([1, 2, 3]));
    const source = await openScopedLibrarySource(root);

    await expect(source.inventory()).resolves.toEqual({
      entries: [
        { path: "papers", type: "folder" },
        { path: "papers/example.pdf", type: "file", size: 3 },
      ],
      truncated: false,
    });
    expect(Array.from(new Uint8Array(await source.readBinary("papers/example.pdf"))))
      .toEqual([1, 2, 3]);
    expect("writeText" in source).toBe(false);
    expect("remove" in source).toBe(false);
    expect("rename" in source).toBe(false);
  });

  it.each([
    "../outside.pdf",
    "papers/../outside.pdf",
    "/tmp/outside.pdf",
    "C:/outside.pdf",
    "C:\\outside.pdf",
    "//server/share.pdf",
    "papers//paper.pdf",
    "papers/./paper.pdf",
  ])("rejects unsafe logical path %s", async (logicalPath) => {
    const root = await makeTempDir();
    const source = await openScopedLibrarySource(root);

    await expect(source.readBinary(logicalPath)).rejects.toMatchObject({
      name: "LibrarySourceError",
      kind: "unsafe-path",
    });
  });

  it("does not follow a symbolic link that escapes the selected root", async () => {
    const parent = await makeTempDir();
    const root = join(parent, "root");
    const outside = join(parent, "outside");
    await mkdir(root);
    await mkdir(outside);
    await writeFile(join(outside, "private.pdf"), "private");
    await symlink(outside, join(root, "linked"), "dir");
    const source = await openScopedLibrarySource(root);

    await expect(source.inventory()).resolves.toEqual({
      entries: [{
        path: "linked",
        type: "ignored",
        ignoredReason: "symbolic-link",
      }],
      truncated: false,
    });
    await expect(source.readBinary("linked/private.pdf")).rejects.toMatchObject({
      kind: "unsafe-path",
    });
  });

  it("applies capability entry, depth, and byte limits that calls cannot widen", async () => {
    const root = await makeTempDir();
    await mkdir(join(root, "nested"));
    await writeFile(join(root, "a.pdf"), "1234");
    await writeFile(join(root, "nested", "b.pdf"), "5678");
    const source = await openScopedLibrarySource(root, {
      maxEntries: 1,
      maxDepth: 0,
      maxReadBytes: 3,
    });

    const inventory = await source.inventory({
      maxEntries: 100,
      maxDepth: 100,
    });
    expect(inventory.entries).toHaveLength(1);
    expect(inventory.truncated).toBe(true);
    expect(inventory.entries.some((entry) => entry.path === "nested/b.pdf")).toBe(false);
    await expect(source.readBinary("a.pdf", { maxBytes: 100 }))
      .rejects.toMatchObject({ kind: "limit-exceeded" });
  });

  it("marks inventory truncated when the depth limit omits descendants", async () => {
    const root = await makeTempDir();
    await mkdir(join(root, "nested"));
    await writeFile(join(root, "nested", "paper.pdf"), "paper");
    const source = await openScopedLibrarySource(root, { maxDepth: 0 });

    await expect(source.inventory()).resolves.toEqual({
      entries: [{ path: "nested", type: "folder" }],
      truncated: true,
    });
  });

  it("allows calls to tighten capability limits", async () => {
    const root = await makeTempDir();
    await writeFile(join(root, "a.pdf"), "1234");
    await writeFile(join(root, "b.pdf"), "5678");
    const source = await openScopedLibrarySource(root, {
      maxEntries: 10,
      maxReadBytes: 10,
    });

    await expect(source.inventory({ maxEntries: 1 })).resolves.toMatchObject({
      entries: [expect.any(Object)],
      truncated: true,
    });
    await expect(source.readBinary("a.pdf", { maxBytes: 3 }))
      .rejects.toMatchObject({ kind: "limit-exceeded" });
  });

  it("honors cancellation before inventory and reads", async () => {
    const root = await makeTempDir();
    await writeFile(join(root, "paper.pdf"), "paper");
    const source = await openScopedLibrarySource(root);
    const controller = new AbortController();
    controller.abort("stop inventory");

    await expect(source.inventory({ signal: controller.signal }))
      .rejects.toSatisfy(isCancellationError);
    await expect(source.readBinary("paper.pdf", { signal: controller.signal }))
      .rejects.toSatisfy(isCancellationError);
  });

  it("rejects missing roots and directory reads without exposing paths", async () => {
    const parent = await makeTempDir();
    await expect(openScopedLibrarySource(join(parent, "missing"))).rejects.toMatchObject({
      name: "LibrarySourceError",
      kind: "not-found",
      message: "Unable to open the selected library root",
    });
    const source = await openScopedLibrarySource(parent);
    await mkdir(join(parent, "folder"));
    await expect(source.readBinary("folder")).rejects.toMatchObject({
      kind: "not-file",
      message: "The requested library entry is not a file",
    });
  });
});
