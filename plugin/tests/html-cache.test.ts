import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { HtmlCache } from "../src/pipeline/html-cache";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as os from "node:os";

let tmpDir: string;

beforeEach(async () => {
  tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-cache-test-"));
});

afterEach(async () => {
  await fs.rm(tmpDir, { recursive: true, force: true });
});

describe("HtmlCache", () => {
  it("returns null when key not present", async () => {
    const cache = new HtmlCache({ rootDir: tmpDir, expiryDays: 7 });
    expect(await cache.get("missing", "html")).toBeNull();
  });

  it("round-trips set then get", async () => {
    const cache = new HtmlCache({ rootDir: tmpDir, expiryDays: 7 });
    await cache.set("2605.08080", "html", "<html>hi</html>");
    expect(await cache.get("2605.08080", "html")).toBe("<html>hi</html>");
  });

  it("html and abs are separate namespaces", async () => {
    const cache = new HtmlCache({ rootDir: tmpDir, expiryDays: 7 });
    await cache.set("k", "html", "HTML");
    await cache.set("k", "abs", "ABS");
    expect(await cache.get("k", "html")).toBe("HTML");
    expect(await cache.get("k", "abs")).toBe("ABS");
  });

  it("expires entries older than expiryDays", async () => {
    const cache = new HtmlCache({ rootDir: tmpDir, expiryDays: 7 });
    await cache.set("k", "html", "old");
    // Backdate mtime by 8 days
    const p = path.join(tmpDir, "html");
    const files = await fs.readdir(p);
    const oldTime = new Date(Date.now() - 8 * 86_400_000);
    await fs.utimes(path.join(p, files[0]), oldTime, oldTime);
    expect(await cache.get("k", "html")).toBeNull();
    // File should be removed after expired read
    expect(await fs.readdir(p)).toHaveLength(0);
  });

  it("creates nested directories on first write", async () => {
    const cache = new HtmlCache({ rootDir: path.join(tmpDir, "deep", "nest"), expiryDays: 7 });
    await cache.set("k", "html", "x");
    expect(await cache.get("k", "html")).toBe("x");
  });
});
