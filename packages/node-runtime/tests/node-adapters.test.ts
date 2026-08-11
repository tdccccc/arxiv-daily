import { afterEach, describe, expect, it, vi } from "vitest";
import {
  chmod,
  mkdir,
  mkdtemp,
  readFile,
  readdir as fsReaddir,
  rename,
  rm,
  stat,
  symlink,
  writeFile,
} from "node:fs/promises";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import {
  DailyFilterCheckpointStore,
  DailySummaryCheckpointStore,
  DEFAULT_SETTINGS,
  deliveryStatePath,
  deliverDailyEmailIfEnabled,
  emptyDeliveryState,
  HttpTransportError,
  markDelivered,
  readDeliveryState,
  isCancellationError,
  prepareDailyFilterCheckpoint,
  sampleDailyDigest,
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
        retryableAttempt: false,
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

  it("fails closed and preserves the target when atomic rename reports EXDEV", async () => {
    const root = await makeTempDir();
    await mkdir(join(root, "private"), { recursive: true });
    await writeFile(join(root, "private/state.json"), "old", "utf8");
    const storage = new NodeStorageAdapter(root, {
      renameAtomic: async () => {
        const error = new Error("cross-device rename") as NodeJS.ErrnoException;
        error.code = "EXDEV";
        throw error;
      },
    });

    await expect(
      storage.writeTextAtomic("private/state.json", "new", 0o600),
    ).rejects.toMatchObject({ code: "EXDEV" });
    expect(await readFile(join(root, "private/state.json"), "utf8")).toBe("old");
    expect((await fsReaddir(join(root, "private"))).filter((entry) =>
      entry.includes("state.json.tmp-") || entry.includes("state.json.bak-"),
    )).toEqual([]);
  });

  it("exclusively creates one file across independent adapter instances", async () => {
    const root = await makeTempDir();
    const first = new NodeStorageAdapter(root) as NodeStorageAdapter & {
      createTextExclusive(path: string, content: string): Promise<boolean>;
    };
    const second = new NodeStorageAdapter(root) as NodeStorageAdapter & {
      createTextExclusive(path: string, content: string): Promise<boolean>;
    };

    const results = await Promise.all([
      first.createTextExclusive("claims/daily.lock", "first"),
      second.createTextExclusive("claims/daily.lock", "second"),
    ]);

    expect(results.sort()).toEqual([false, true]);
    expect(["first", "second"]).toContain(
      await first.readText("claims/daily.lock"),
    );
  });

  it("rejects parent traversal and existing parent symlink escape for exclusive create", async () => {
    const root = await makeTempDir();
    const outside = await makeTempDir();
    const storage = new NodeStorageAdapter(root);
    await symlink(outside, join(root, "escape"), "dir");

    await expect(
      storage.createTextExclusive?.("../outside.claim", "secret"),
    ).rejects.toThrow(/escapes root/);
    await expect(
      storage.createTextExclusive?.("escape/outside.claim", "secret"),
    ).rejects.toThrow(/symlink|escapes root/);
    await expect(readFile(join(outside, "outside.claim"), "utf8"))
      .rejects.toMatchObject({ code: "ENOENT" });
  });

  it("anchors exclusive create to opened parent descriptors across a symlink swap", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const outside = await makeTempDir();
    const parent = join(root, "claims", "daily");
    const movedParent = join(root, "claims", "daily-opened");
    let resume!: () => void;
    let validated!: () => void;
    const reachedBarrier = new Promise<void>((resolve) => { validated = resolve; });
    const barrier = new Promise<void>((resolve) => { resume = resolve; });
    const storage = new NodeStorageAdapter(root, {
      afterFinalParentOpened: async () => {
        validated();
        await barrier;
      },
    });

    const creating = storage.createTextExclusive!("claims/daily/send.claim", "claimed");
    await reachedBarrier;
    await rename(parent, movedParent);
    await symlink(outside, parent, "dir");
    resume();
    const result = await creating.then(
      (created) => ({ created }),
      (error) => ({ error }),
    );

    expect(await readFile(join(outside, "send.claim"), "utf8").catch(() => null))
      .toBeNull();
    expect(result).toHaveProperty("error");
    expect(await readFile(join(movedParent, "send.claim"), "utf8").catch(() => null))
      .toBeNull();
  });

  it("fails closed when an opened parent tree is moved outside root and linked back", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const outside = await makeTempDir();
    const claims = join(root, "claims");
    const movedClaims = join(outside, "claims-moved");
    const storage = new NodeStorageAdapter(root, {
      afterFinalParentOpened: async () => {
        await rename(claims, movedClaims);
        await symlink(movedClaims, claims, "dir");
      },
    });

    const result = await storage
      .createTextExclusive!("claims/daily/send.claim", "claimed")
      .then(
        (created) => ({ created }),
        (error) => ({ error }),
      );

    expect(result).toHaveProperty("error");
    expect(
      await readFile(join(movedClaims, "daily/send.claim"), "utf8").catch(
        () => null,
      ),
    ).toBeNull();
  });

  it("fails closed when an opened parent tree moves within root and is linked back", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const claims = join(root, "claims");
    const movedClaims = join(root, "claims-moved-within-root");
    const storage = new NodeStorageAdapter(root, {
      afterFinalParentOpened: async () => {
        await rename(claims, movedClaims);
        await symlink(movedClaims, claims, "dir");
      },
    });

    const result = await storage
      .createTextExclusive!("claims/daily/send.claim", "claimed")
      .then(
        (created) => ({ created }),
        (error) => ({ error }),
      );

    expect(result).toHaveProperty("error");
    expect(
      await readFile(join(movedClaims, "daily/send.claim"), "utf8").catch(
        () => null,
      ),
    ).toBeNull();
  });

  it("removes a claim created just before its parent tree leaves root", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const outside = await makeTempDir();
    const claims = join(root, "claims");
    const movedClaims = join(outside, "claims-post-create");
    const storage = new NodeStorageAdapter(root, {
      afterTargetCreated: async () => {
        await rename(claims, movedClaims);
        await symlink(movedClaims, claims, "dir");
      },
    });

    await expect(
      storage.createTextExclusive!("claims/daily/send.claim", "claimed"),
    ).rejects.toThrow(/escapes root|replaced/);
    expect(
      await readFile(join(movedClaims, "daily/send.claim"), "utf8").catch(
        () => null,
      ),
    ).toBeNull();
  });

  it("never calls the provider twice when a claim parent is renamed after opening", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const claimParent = join(
      root,
      `${deliveryStatePath(DEFAULT_SETTINGS.output)}.claims`,
    );
    const movedParent = `${claimParent}-hidden`;
    let swapped = false;
    const firstStorage = new NodeStorageAdapter(root, {
      afterFinalParentOpened: async () => {
        if (swapped) return;
        swapped = true;
        await rename(claimParent, movedParent);
        await mkdir(claimParent, { recursive: true });
      },
    });
    const secondStorage = new NodeStorageAdapter(root);
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ id: "msg_once" }),
    }));
    const digest = sampleDailyDigest({ date: "2026-08-10" });
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "rename@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_test",
    };

    const first = await deliverDailyEmailIfEnabled(digest, {
      storage: firstStorage,
      http: { request },
      output: DEFAULT_SETTINGS.output,
      email,
      sleep: async () => {},
    });
    const second = await deliverDailyEmailIfEnabled(digest, {
      storage: secondStorage,
      http: { request },
      output: DEFAULT_SETTINGS.output,
      email,
      sleep: async () => {},
    });

    expect(first).toMatchObject({ kind: "failed", attempts: 0 });
    expect(second.kind).toBe("delivered");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("never sends from a claim world hidden immediately after exclusive create", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const claimParent = join(
      root,
      `${deliveryStatePath(DEFAULT_SETTINGS.output)}.claims`,
    );
    const movedParent = `${claimParent}-post-create-hidden`;
    const firstStorage = new NodeStorageAdapter(root);
    const originalCreate = firstStorage.createTextExclusive!;
    let swapped = false;
    Object.defineProperty(firstStorage, "createTextExclusive", {
      configurable: true,
      value: async (storagePath: string, content: string) => {
        const created = await originalCreate(storagePath, content);
        if (created && storagePath.endsWith(".claim.json") && !swapped) {
          swapped = true;
          await rename(claimParent, movedParent);
          await mkdir(claimParent, { recursive: true });
        }
        return created;
      },
    });
    const secondStorage = new NodeStorageAdapter(root);
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ id: "msg_once" }),
    }));
    const digest = sampleDailyDigest({ date: "2026-08-10" });
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "post-create-rename@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_test",
    };

    const first = await deliverDailyEmailIfEnabled(digest, {
      storage: firstStorage,
      http: { request },
      output: DEFAULT_SETTINGS.output,
      email,
      sleep: async () => {},
    });
    const second = await deliverDailyEmailIfEnabled(digest, {
      storage: secondStorage,
      http: { request },
      output: DEFAULT_SETTINGS.output,
      email,
      sleep: async () => {},
    });

    expect(first).toMatchObject({ kind: "failed", attempts: 0 });
    expect(second).toEqual({
      kind: "delivered",
      attempts: 1,
    });
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("fails closed if the claim namespace is replaced after the final result absence read", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const statePath = deliveryStatePath(DEFAULT_SETTINGS.output);
    const indexDir = join(root, dirname(statePath));
    const movedIndex = `${indexDir}-old-world`;
    const firstStorage = new NodeStorageAdapter(root);
    const originalExists = firstStorage.exists.bind(firstStorage);
    let resultAbsenceReads = 0;
    firstStorage.exists = async (storagePath) => {
      const exists = await originalExists(storagePath);
      if (!exists && storagePath.endsWith(".result.json")) {
        resultAbsenceReads += 1;
        if (resultAbsenceReads === 4) {
          await rename(indexDir, movedIndex);
          await mkdir(indexDir, { recursive: true });
        }
      }
      return exists;
    };
    const secondStorage = new NodeStorageAdapter(root);
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ id: "msg_same_provider_key" }),
    }));
    const digest = sampleDailyDigest({ date: "2026-08-10" });
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "last-read-swap@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_test",
    };

    const first = await deliverDailyEmailIfEnabled(digest, {
      storage: firstStorage,
      http: { request },
      output: DEFAULT_SETTINGS.output,
      email,
      sleep: async () => {},
    });
    const second = await deliverDailyEmailIfEnabled(digest, {
      storage: secondStorage,
      http: { request },
      output: DEFAULT_SETTINGS.output,
      email,
      sleep: async () => {},
    });

    expect({ first, second, providerCalls: request.mock.calls.length, resultAbsenceReads })
      .toEqual({
        first: {
          kind: "ambiguous",
          reason: "delivery_claim_storage_failed",
          attempts: 0,
        },
        second: { kind: "delivered", attempts: 1 },
        providerCalls: 1,
        resultAbsenceReads: expect.any(Number),
      });
  });

  it("tightens an existing legacy delivery-state primary to 0600 before reading it", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const storage = new NodeStorageAdapter(root);
    const statePath = deliveryStatePath(DEFAULT_SETTINGS.output);
    const target = join(root, statePath);
    await mkdir(dirname(target), { recursive: true });
    const rawRecipient = "legacy-private@example.com";
    const state = markDelivered(emptyDeliveryState(), {
      date: "2026-08-10",
      recipient: rawRecipient,
      attempts: 1,
    });
    await writeFile(target, `${JSON.stringify(state, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o644,
    });
    await chmod(target, 0o644);

    const read = await readDeliveryState(storage, DEFAULT_SETTINGS.output);

    expect(read.kind).toBe("valid");
    expect(await fileMode(target)).toBe(0o600);
    expect(await readFile(target, "utf8")).toContain(rawRecipient);
  });

  it("coordinates concurrent automatic delivery across Node adapter instances", async () => {
    const root = await makeTempDir();
    const firstStorage = new NodeStorageAdapter(root);
    const secondStorage = new NodeStorageAdapter(root);
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      bodyText: JSON.stringify({ id: "msg_once" }),
    }));
    const digest = sampleDailyDigest({ date: "2026-08-10" });
    const email = {
      enabled: true,
      mode: "self" as const,
      to: "shared@example.com",
      fromEmail: "from@example.com",
      apiKey: "re_test",
    };

    const results = await Promise.all([
      deliverDailyEmailIfEnabled(digest, {
        storage: firstStorage,
        http: { request },
        output: DEFAULT_SETTINGS.output,
        email,
        sleep: async () => {},
      }),
      deliverDailyEmailIfEnabled(digest, {
        storage: secondStorage,
        http: { request },
        output: DEFAULT_SETTINGS.output,
        email,
        sleep: async () => {},
      }),
    ]);

    expect(request).toHaveBeenCalledTimes(1);
    expect(results.map((result) => result.kind).sort()).toEqual([
      "delivered",
      "skipped",
    ]);

    const statePath = deliveryStatePath(DEFAULT_SETTINGS.output);
    expect(await fileMode(join(root, statePath))).toBe(0o600);
    const claimDir = join(root, `${statePath}.claims`);
    for (const entry of await fsReaddir(claimDir)) {
      expect(await fileMode(join(claimDir, entry))).toBe(0o600);
    }
    const indexDir = join(root, dirname(statePath));
    expect((await fsReaddir(indexDir)).filter((entry) =>
      entry.includes("delivery-state.json.tmp-") ||
      entry.includes("delivery-state.json.bak-")
    )).toEqual([]);
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
