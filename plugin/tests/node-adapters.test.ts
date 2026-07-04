import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  buildNodeHostAdapters,
  EnvSecretProvider,
  NodeHttpClient,
  NodeStorageAdapter,
  StreamProgressReporter,
  StreamResourceOpener,
  type FetchLike,
} from "../src/hosts/node";

const tempDirs: string[] = [];

afterEach(async () => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();
    if (dir) await rm(dir, { recursive: true, force: true });
  }
});

async function makeTempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "arxiv-daily-node-"));
  tempDirs.push(dir);
  return dir;
}

function captureStream() {
  const chunks: string[] = [];
  return {
    chunks,
    stream: {
      write(chunk: string) {
        chunks.push(String(chunk));
      },
    },
  };
}

describe("NodeHttpClient", () => {
  it("wraps fetch responses in the core HTTP contract", async () => {
    const calls: Array<{ input: string; init?: RequestInit }> = [];
    const fetchImpl: FetchLike = async (input, init) => {
      calls.push({ input, init });
      return new Response("ok", {
        status: 201,
        headers: { "x-test": "yes" },
      });
    };
    const client = new NodeHttpClient(fetchImpl);

    const res = await client.request({
      url: "https://example.test",
      method: "POST",
      headers: { "x-input": "1" },
      body: "payload",
    });

    expect(res).toEqual({
      status: 201,
      headers: expect.objectContaining({ "x-test": "yes" }),
      bodyText: "ok",
    });
    expect(calls).toHaveLength(1);
    expect(calls[0].input).toBe("https://example.test");
    expect(calls[0].init?.method).toBe("POST");
    expect(calls[0].init?.body).toBe("payload");
  });
});

describe("NodeStorageAdapter", () => {
  it("reads and writes text and binary files under a root directory", async () => {
    const root = await makeTempDir();
    const storage = new NodeStorageAdapter(root);

    expect(storage.normalizePath("\\daily//2026-06-13.md")).toBe(
      "daily/2026-06-13.md",
    );
    await storage.mkdir("daily");
    await storage.writeText("daily/2026-06-13.md", "content");
    expect(await storage.exists("daily/2026-06-13.md")).toBe(true);
    expect(await storage.readText("daily/2026-06-13.md")).toBe("content");

    await storage.rename(
      "daily/2026-06-13.md",
      "archive/2026-06-13.md",
    );
    expect(await storage.exists("daily/2026-06-13.md")).toBe(false);
    expect(await storage.readText("archive/2026-06-13.md")).toBe("content");

    await storage.mkdir("binary");
    await storage.writeBinary("binary/data.bin", new Uint8Array([1, 2]).buffer);
    expect(Array.from(new Uint8Array(await storage.readBinary("binary/data.bin"))))
      .toEqual([1, 2]);
  });

  it("writes text atomically through a temporary file", async () => {
    const root = await makeTempDir();
    const storage = new NodeStorageAdapter(root);

    await storage.writeTextAtomic("daily/2026-06-13.md", "content");

    expect(await storage.readText("daily/2026-06-13.md")).toBe("content");
    expect(await storage.exists("daily/2026-06-13.md.tmp")).toBe(false);
  });

  it("lists entries and rejects paths outside the root", async () => {
    const root = await makeTempDir();
    const storage = new NodeStorageAdapter(root);
    await storage.mkdir("papers");
    await storage.writeText("papers/a.md", "a");
    await storage.mkdir("papers/nested");

    expect(await storage.list("papers")).toEqual([
      { path: "papers/a.md", type: "file" },
      { path: "papers/nested", type: "folder" },
    ]);
    await expect(storage.exists("../outside")).rejects.toThrow(/escapes root/);
  });
});

describe("Node env/progress/resource adapters", () => {
  it("reads secrets from env-style keys", async () => {
    const provider = new EnvSecretProvider({
      ARXIV_DAILY_LLM_API_KEY: "secret",
    });
    expect(await provider.getSecret("llm.apiKey")).toBe("secret");
    expect(await provider.getSecret("missing")).toBeNull();
  });

  it("stores CLI-set secrets in memory for the current Node session", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const provider = new EnvSecretProvider({});

    await provider.setSecret("llm.apiKey", "session-secret");
    expect(await provider.getSecret("llm.apiKey")).toBe("session-secret");
    expect(warn).toHaveBeenCalledWith(
      "Node EnvSecretProvider stores secrets in memory only; they will not persist across restarts",
    );

    await provider.deleteSecret("llm.apiKey");
    expect(await provider.getSecret("llm.apiKey")).toBeNull();
  });

  it("writes progress and resource targets to streams", async () => {
    const progress = captureStream();
    const opener = captureStream();

    new StreamProgressReporter(progress.stream).setStage("filter", 1, 2);
    await new StreamResourceOpener(opener.stream).openUrl("https://arxiv.org");

    expect(progress.chunks.join("")).toContain("stage filter 1/2");
    expect(opener.chunks.join("")).toContain("url https://arxiv.org");
  });

  it("assembles host adapters for the Node runtime", async () => {
    const root = await makeTempDir();
    const progress = captureStream();
    const opener = captureStream();
    const host = buildNodeHostAdapters({
      rootDir: root,
      env: { ARXIV_DAILY_API_KEY: "key" },
      fetch: async () => new Response("ok", { status: 200 }),
      progressStream: progress.stream,
      openerStream: opener.stream,
    });

    await host.storage.mkdir("daily");
    await host.storage.writeText("daily/test.md", "ok");
    host.progress.setDisabled();
    await host.opener.openNote("daily/test.md");

    expect(await host.http.request({ url: "https://example.test" }))
      .toMatchObject({ status: 200, bodyText: "ok" });
    expect(await host.secrets.getSecret("apiKey")).toBe("key");
    expect(await host.storage.readText("daily/test.md")).toBe("ok");
    expect(progress.chunks.join("")).toContain("disabled");
    expect(opener.chunks.join("")).toContain("note daily/test.md");
  });
});
