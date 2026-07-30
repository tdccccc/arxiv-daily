import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const mainSource = readFileSync(resolve(process.cwd(), "main.ts"), "utf-8");

describe("cache cleanup startup gate", () => {
  it("derives the cleanup day from the configured timezone", () => {
    expect(mainSource).toContain(
      "return formatDate(todayInTz(now, timezone));",
    );
  });

  it("wires the persistent Atom metadata cache into fetchers and cleanup", () => {
    expect(mainSource).toContain("metadataCache: new AtomMetadataCache({");
    expect(mainSource).toContain("rootDir: this.pluginCacheDir(),");
    expect(mainSource).toContain("const metadataRemoved = await new AtomMetadataCache({");
    expect(mainSource).toContain("}).cleanupExpired();");
  });

  it("gates startup cleanup before invoking the expensive scan", () => {
    expect(mainSource).toContain("let lastCacheCleanupDate: string | null = null;");
    expect(mainSource).toContain("this.cleanupCachesIfDue();");
    expect(mainSource).toContain("!shouldRunCacheCleanup(");
    expect(mainSource).toContain("lastCacheCleanupDate = cacheCleanupDateKey(");
    expect(mainSource).toContain("this.cleanupCaches().catch((e) =>");
  });
});
