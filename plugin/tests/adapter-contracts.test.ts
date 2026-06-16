import { describe, expect, it } from "vitest";
import type { HostAdapters, ProgressReporter } from "../src/core/adapters";
import { NoopProgressReporter } from "../src/services/progress";

describe("host adapter contracts", () => {
  it("can be implemented without Obsidian runtime types", async () => {
    const opened: string[] = [];
    const host: HostAdapters = {
      http: {
        async request(req) {
          return {
            status: req.url.startsWith("https://") ? 200 : 400,
            headers: { "content-type": "text/plain" },
            bodyText: "ok",
          };
        },
      },
      storage: {
        normalizePath: (path) => path.replace(/\\/g, "/"),
        readText: async () => "content",
        writeText: async () => {},
        exists: async () => false,
        mkdir: async () => {},
        remove: async () => {},
        rename: async () => {},
      },
      secrets: {
        getSecret: async (key) => (key === "llm" ? "secret" : null),
      },
      progress: new NoopProgressReporter(),
      opener: {
        openNote: async (path) => {
          opened.push(`note:${path}`);
        },
        openDailyReport: async (path) => {
          opened.push(`daily:${path}`);
        },
        openUrl: async (url) => {
          opened.push(`url:${url}`);
        },
      },
    };

    const res = await host.http.request({ url: "https://example.test" });
    await host.opener.openNote("papers/2606.12345.md");

    expect(res.status).toBe(200);
    expect(await host.secrets.getSecret("llm")).toBe("secret");
    expect(host.storage.normalizePath("a\\b")).toBe("a/b");
    expect(opened).toEqual(["note:papers/2606.12345.md"]);
  });

  it("keeps the existing noop progress reporter aligned with core", () => {
    const progress: ProgressReporter = new NoopProgressReporter();
    expect(() => progress.setTask("Daily", "2026-06-13")).not.toThrow();
    expect(() => progress.setStage("filter", 1, 2)).not.toThrow();
    expect(() => progress.setComplete("done")).not.toThrow();
    expect(() => progress.setError("failed")).not.toThrow();
    expect(() => progress.setIdle("2026-06-13", "weekend")).not.toThrow();
  });
});
