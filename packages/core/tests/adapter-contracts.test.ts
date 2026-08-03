import { describe, expect, it } from "vitest";
import {
  HttpTransportError,
  isHttpTransportError,
  NoopProgressReporter,
  type HostAdapters,
  type ProgressReporter,
  type ScopedLibrarySource,
} from "../src/index";

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
      markupParser: {
        parseFromString: () => ({}) as Document,
      },
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

  it("defines personal-library access as a read-only scoped capability", async () => {
    const source: ScopedLibrarySource = {
      inventory: async () => ({
        entries: [{ path: "papers/example.pdf", type: "file", size: 123 }],
        truncated: false,
      }),
      readBinary: async () => new Uint8Array([1, 2, 3]).buffer,
    };

    expect(await source.inventory()).toEqual({
      entries: [{ path: "papers/example.pdf", type: "file", size: 123 }],
      truncated: false,
    });
    expect(Array.from(new Uint8Array(await source.readBinary("papers/example.pdf"))))
      .toEqual([1, 2, 3]);
    expect("writeText" in source).toBe(false);
    expect("remove" in source).toBe(false);
    expect("rename" in source).toBe(false);
  });

  it("recognizes transport errors structurally across host boundaries", () => {
    expect(isHttpTransportError(new HttpTransportError("network", "offline"))).toBe(true);
    expect(isHttpTransportError({
      name: "HttpTransportError",
      message: "deadline",
      kind: "timeout",
    })).toBe(true);
    expect(isHttpTransportError({
      name: "HttpTransportError",
      message: "unknown",
      kind: "local",
    })).toBe(false);
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
