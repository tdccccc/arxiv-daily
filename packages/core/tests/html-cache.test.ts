import { describe, it, expect } from "vitest";
import { HtmlCache } from "../src/pipeline/html-cache";
import type { StorageAdapter } from "../src/core/adapters";

function makeStorage() {
  const files: Record<string, string> = {};
  const dirs = new Set<string>();
  const storage = {
    normalizePath(path: string) {
      return path.replace(/\\/g, "/").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
    },
    async readText(path: string) {
      if (!(path in files)) throw new Error(`missing ${path}`);
      return files[path];
    },
    async writeText(path: string, content: string) {
      files[path] = content;
    },
    async exists(path: string) {
      return path in files || dirs.has(path);
    },
    async mkdir(path: string) {
      dirs.add(path);
    },
    async remove(path: string) {
      delete files[path];
      dirs.delete(path);
    },
    async rename(from: string, to: string) {
      files[to] = files[from];
      delete files[from];
    },
    async list(dir: string) {
      const prefix = `${dir.replace(/\/+$/g, "")}/`;
      const out: Array<{ path: string; type: "file" | "folder" }> = [];
      for (const path of Object.keys(files)) {
        if (!path.startsWith(prefix)) continue;
        const rest = path.slice(prefix.length);
        if (rest && !rest.includes("/")) out.push({ path, type: "file" });
      }
      for (const path of dirs) {
        if (!path.startsWith(prefix)) continue;
        const rest = path.slice(prefix.length);
        if (rest && !rest.includes("/")) out.push({ path, type: "folder" });
      }
      return out;
    },
  } satisfies StorageAdapter;
  return { files, dirs, storage };
}

describe("HtmlCache", () => {
  it("returns null when key not present", async () => {
    const cache = new HtmlCache({ rootDir: "cache", expiryDays: 7, storage: makeStorage().storage });
    expect(await cache.get("missing", "html")).toBeNull();
  });

  it("round-trips set then get", async () => {
    const cache = new HtmlCache({ rootDir: "cache", expiryDays: 7, storage: makeStorage().storage });
    await cache.set("2605.08080", "html", "<html>hi</html>");
    expect(await cache.get("2605.08080", "html")).toBe("<html>hi</html>");
  });

  it("keeps the legacy SHA-1 cache key", async () => {
    const { files, storage } = makeStorage();
    const cache = new HtmlCache({ rootDir: "cache", expiryDays: 7, storage });
    await cache.set("2605.08080", "html", "HTML");
    expect(Object.keys(files)).toEqual([
      "cache/html/d18a24abe03dd46c244455f5.json",
    ]);
  });

  it("html and abs are separate namespaces", async () => {
    const cache = new HtmlCache({ rootDir: "cache", expiryDays: 7, storage: makeStorage().storage });
    await cache.set("k", "html", "HTML");
    await cache.set("k", "abs", "ABS");
    expect(await cache.get("k", "html")).toBe("HTML");
    expect(await cache.get("k", "abs")).toBe("ABS");
  });

  it("expires entries older than expiryDays", async () => {
    const { files, storage } = makeStorage();
    const cache = new HtmlCache({ rootDir: "cache", expiryDays: 7, storage });
    await cache.set("k", "html", "old");
    const path = Object.keys(files)[0]!;
    files[path] = JSON.stringify({ schemaVersion: 1, cachedAt: new Date(Date.now() - 8 * 86_400_000).toISOString(), content: "old" });
    expect(await cache.get("k", "html")).toBeNull();
    expect(files[path]).toBeUndefined();
  });

  it("creates nested directories on first write", async () => {
    const cache = new HtmlCache({ rootDir: "deep/nest", expiryDays: 7, storage: makeStorage().storage });
    await cache.set("k", "html", "x");
    expect(await cache.get("k", "html")).toBe("x");
  });

  it("stores plugin cache entries through the storage adapter", async () => {
    const { files, storage } = makeStorage();
    const cache = new HtmlCache({
      rootDir: ".obsidian/plugins/arxiv-daily/.cache",
      expiryDays: 7,
      storage,
    });

    await cache.set("2605.08080", "html", "<html>hi</html>");

    expect(Object.keys(files)[0]).toMatch(
      /^\.obsidian\/plugins\/arxiv-daily\/\.cache\/html\/[a-f0-9]+\.json$/,
    );
    expect(await cache.get("2605.08080", "html")).toBe("<html>hi</html>");
  });

  it("cleans expired storage cache entries by cachedAt metadata", async () => {
    const { files, storage } = makeStorage();
    const cache = new HtmlCache({
      rootDir: ".obsidian/plugins/arxiv-daily/.cache",
      expiryDays: 7,
      storage,
    });
    await cache.set("old", "abs", "old abs");
    const path = Object.keys(files)[0];
    files[path] = JSON.stringify({
      schemaVersion: 1,
      cachedAt: new Date(Date.now() - 8 * 86_400_000).toISOString(),
      content: "old abs",
    });

    expect(await cache.cleanupExpired()).toBe(1);
    expect(files[path]).toBeUndefined();
  });
});
