import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  DailyFilterCheckpointStore,
  DailySummaryCheckpointStore,
  DEFAULT_SETTINGS,
  HttpTransportError,
  isCancellationError,
  prepareDailyFilterCheckpoint,
  type DailyPaperResult,
  type DailySummaryCheckpointCompatibilityInput,
} from "@arxiv-daily/core";
import {
  buildNodeHostAdapters,
  EnvSecretProvider,
  NodeHttpClient,
  NodeStorageAdapter,
  StreamProgressReporter,
  StreamResourceOpener,
  type FetchLike,
} from "../src/index";

const tempDirs: string[] = [];

afterEach(async () => {
  vi.useRealTimers();
  vi.restoreAllMocks();
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

async function fileMode(path: string): Promise<number> {
  return (await stat(path)).mode & 0o777;
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

  it("enforces the deadline through body consumption when fetch ignores abort", async () => {
    vi.useFakeTimers();
    let rejectBody!: (error: Error) => void;
    const body = new Promise<string>((_resolve, reject) => { rejectBody = reject; });
    const fetchImpl: FetchLike = vi.fn(async () => ({
      status: 200,
      headers: new Headers(),
      text: () => body,
    }) as Response);
    const client = new NodeHttpClient(fetchImpl);

    const result = client.request({ url: "https://example.test", timeoutMs: 25 });
    const assertion = expect(result).rejects.toMatchObject({
      name: "HttpTransportError",
      kind: "timeout",
    });
    await vi.advanceTimersByTimeAsync(25);
    await assertion;

    rejectBody(new Error("late body failure"));
    await Promise.resolve();
  });

  it("distinguishes caller cancellation from timeout without waiting for fetch", async () => {
    const controller = new AbortController();
    const client = new NodeHttpClient(() => new Promise<Response>(() => {}));
    const result = client.request({
      url: "https://example.test",
      timeoutMs: 60_000,
      signal: controller.signal,
    });

    controller.abort("stop now");

    await expect(result).rejects.toSatisfy(isCancellationError);
  });

  it("normalizes fetch rejection as a network transport error", async () => {
    const cause = new Error("socket closed");
    const client = new NodeHttpClient(async () => { throw cause; });

    await expect(client.request({ url: "https://example.test" })).rejects.toEqual(
      expect.objectContaining<HttpTransportError>({
        name: "HttpTransportError",
        kind: "network",
      }),
    );
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

  it("preserves a recoverable checkpoint backup across consecutive real upserts", async () => {
    const root = await makeTempDir();
    const storage = new NodeStorageAdapter(root);
    const privateWrites: Array<{ path: string; mode: number; actualMode: number }> = [];
    const writePrivate = storage.writeTextWithMode.bind(storage);
    storage.writeTextWithMode = async (path, content, mode) => {
      await writePrivate(path, content, mode);
      privateWrites.push({ path, mode, actualMode: await fileMode(join(root, path)) });
    };
    const store = new DailySummaryCheckpointStore(storage, DEFAULT_SETTINGS.output);
    const first: DailySummaryCheckpointCompatibilityInput = {
      paper: {
        id: "2608.00001",
        title: "First",
        authors: "Author",
        abstract: "First abstract.",
        abstractConclusion: "First content.",
      },
      llm: {
        provider: "custom",
        baseUrl: "https://example.test/v1?token=private",
        model: "model-a",
        thinkingMode: false,
        reasoningEffort: "medium",
      },
    };
    const second = {
      ...first,
      paper: { ...first.paper, id: "2608.00002", title: "Second", abstract: "Second abstract." },
    };
    const result = (id: string): DailyPaperResult => ({
      kind: "structured",
      summary: {
        id,
        coreProblem: "Problem",
        keyMethod: "Method",
        mainResult: "Result",
        whyRelevant: "Relevant",
        limitations: "Limits",
      },
    });

    await store.upsert("2026-08-01", first, result("2608.00001"));
    await store.upsert("2026-08-01", second, result("2608.00002"));
    const paths = store.pathsFor("2026-08-01");
    await writeFile(join(root, paths.documentPath), "corrupt", "utf8");

    await expect(new DailySummaryCheckpointStore(storage, DEFAULT_SETTINGS.output)
      .lookupReusable("2026-08-01", first)).resolves.toEqual(result("2608.00001"));
    expect(JSON.parse(await readFile(join(root, paths.backupPath), "utf8")))
      .toMatchObject({ entries: { "arxiv:2608.00001": {} } });
    expect(await fileMode(join(root, paths.documentPath))).toBe(0o600);
    expect(await fileMode(join(root, paths.backupPath))).toBe(0o600);
    expect(privateWrites).toEqual(expect.arrayContaining([
      { path: `${paths.documentPath}.tmp`, mode: 0o600, actualMode: 0o600 },
      { path: `${paths.backupPath}.tmp`, mode: 0o600, actualMode: 0o600 },
    ]));
  });

  it("reconstructs filter checkpoints from backup and cleans every real file", async () => {
    const root = await makeTempDir();
    const storage = new NodeStorageAdapter(root);
    const privateWrites: Array<{ path: string; mode: number; actualMode: number }> = [];
    const writePrivate = storage.writeTextWithMode.bind(storage);
    storage.writeTextWithMode = async (path, content, mode) => {
      await writePrivate(path, content, mode);
      privateWrites.push({ path, mode, actualMode: await fileMode(join(root, path)) });
    };
    const store = new DailyFilterCheckpointStore(storage, DEFAULT_SETTINGS.output);
    const compatibility = {
      papers: [
        { id: "2608.00001", title: "First", authors: "Author", abstract: "First abstract." },
      ],
      arxivSettings: {
        ...DEFAULT_SETTINGS.arxiv,
        categories: ["astro-ph"],
        topics: [
          { id: "topic-id", name: "Topic", tag: "topic", description: "Topic", detail: false },
        ],
      },
      llm: {
        provider: "custom" as const,
        baseUrl: "https://example.test/v1?token=private",
        model: "model-a",
        thinkingMode: false,
        reasoningEffort: "medium" as const,
      },
    };
    const prepared = prepareDailyFilterCheckpoint(compatibility);
    const first = [{ id: "2608.00001", category: "topic" }];
    const second = [{ id: "2608.00001", category: "skip" }];

    await store.save("2026-08-01", prepared, first);
    await store.save("2026-08-01", prepared, second);
    const paths = store.pathsFor("2026-08-01");
    await writeFile(join(root, paths.documentPath), "corrupt", "utf8");

    await expect(new DailyFilterCheckpointStore(storage, DEFAULT_SETTINGS.output)
      .lookupReusable("2026-08-01", prepared)).resolves.toEqual(first);
    expect(JSON.parse(await readFile(join(root, paths.backupPath), "utf8")))
      .toMatchObject({ result: first });
    expect(await fileMode(join(root, paths.documentPath))).toBe(0o600);
    expect(await fileMode(join(root, paths.backupPath))).toBe(0o600);
    expect(privateWrites).toEqual(expect.arrayContaining([
      { path: `${paths.documentPath}.tmp`, mode: 0o600, actualMode: 0o600 },
      { path: `${paths.backupPath}.tmp`, mode: 0o600, actualMode: 0o600 },
    ]));

    await new DailyFilterCheckpointStore(storage, DEFAULT_SETTINGS.output)
      .removeAll("2026-08-01");
    for (const path of [
      paths.documentPath,
      paths.backupPath,
      `${paths.documentPath}.tmp`,
      `${paths.backupPath}.tmp`,
    ]) {
      expect(await storage.exists(path)).toBe(false);
    }
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
