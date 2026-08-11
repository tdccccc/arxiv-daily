import { readFileSync } from "node:fs";
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
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildObsidianHostAdapters,
  ObsidianHttpClient,
  ObsidianStorageAdapter,
} from "../src/hosts/obsidian";
import { createDesktopTextExclusive } from "../node-fs-exclusive";
import { ObsidianMarkupParser } from "../src/hosts/obsidian/markup-parser";
import { SettingsChangeService } from "../src/settings/change-service";
import {
  ArxivFetcher,
  DailyFilterCheckpointStore,
  DailySummaryCheckpointStore,
  DEFAULT_SETTINGS,
  deliveryStatePath,
  emptyDeliveryState,
  isCancellationError,
  Logger,
  markDelivered,
  prepareDailyFilterCheckpoint,
  readDeliveryState,
  saveDeliveryState,
  shouldSendEmail,
  type DailyPaperResult,
  type DailySummaryCheckpointCompatibilityInput,
  type PluginSettings,
  type StorageAdapter,
} from "@arxiv-daily/core";

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
  const dir = await mkdtemp(join(tmpdir(), "arxiv-daily-obsidian-"));
  tempDirs.push(dir);
  return dir;
}

async function fileMode(path: string): Promise<number> {
  return (await stat(path)).mode & 0o777;
}

function realFilesystemAdapter(root: string) {
  return {
    getBasePath: () => root,
    async read(path: string) {
      return readFile(join(root, path), "utf8");
    },
    async write(path: string, content: string) {
      await mkdir(join(root, path, ".."), { recursive: true });
      await writeFile(join(root, path), content, "utf8");
    },
    async exists(path: string) {
      return stat(join(root, path)).then(() => true, () => false);
    },
    async mkdir(path: string) {
      await mkdir(join(root, path), { recursive: true });
    },
    async rename(from: string, to: string) {
      await rename(join(root, from), join(root, to));
    },
    async remove(path: string) {
      await rm(join(root, path), { recursive: true, force: true });
    },
    async list(path: string) {
      return { files: [], folders: [], path };
    },
  };
}

const resourceOpenerSource = readFileSync(
  resolve(process.cwd(), "src/hosts/obsidian/resource-opener.ts"),
  "utf-8",
);

function testSettings(): PluginSettings {
  return {
    ...DEFAULT_SETTINGS,
    llm: {
      ...DEFAULT_SETTINGS.llm,
      apiKey: "stored-key",
    },
    arxiv: { ...DEFAULT_SETTINGS.arxiv },
    output: { ...DEFAULT_SETTINGS.output },
    schedule: { ...DEFAULT_SETTINGS.schedule },
    advanced: { ...DEFAULT_SETTINGS.advanced },
  };
}

function testApp() {
  const files: Record<string, string> = {};
  const dirs = new Set<string>();
  const workspace = {
    openLinkText: vi.fn().mockResolvedValue(undefined),
  };
  const app = {
    vault: {
      adapter: {
        read: vi.fn(async (path: string) => files[path]),
        write: vi.fn(async (path: string, content: string) => {
          files[path] = content;
        }),
        exists: vi.fn(async (path: string) => path in files || dirs.has(path)),
        mkdir: vi.fn(async (path: string) => {
          dirs.add(path);
        }),
        rename: vi.fn(async (from: string, to: string) => {
          if (!(from in files)) throw new Error(`missing ${from}`);
          if (to in files || dirs.has(to)) throw new Error(`destination exists: ${to}`);
          files[to] = files[from];
          delete files[from];
        }),
        copy: vi.fn(async (from: string, to: string) => {
          if (!(from in files)) throw new Error(`missing ${from}`);
          if (to in files || dirs.has(to)) throw new Error(`destination exists: ${to}`);
          files[to] = files[from];
        }),
        remove: vi.fn(async (path: string) => {
          delete files[path];
          dirs.delete(path);
        }),
        list: vi.fn(async (path: string) => ({
          files: Object.keys(files).filter((entry) => entry.startsWith(`${path}/`)),
          folders: Array.from(dirs).filter((entry) => entry.startsWith(`${path}/`)),
        })),
        readBinary: vi.fn(async () => new Uint8Array([1, 2]).buffer),
        writeBinary: vi.fn(async () => {}),
      },
    },
    workspace,
  };
  return { app, files, workspace };
}

describe("Obsidian host adapters", () => {
  it("uses Obsidian's active window without globalThis", () => {
    expect(resourceOpenerSource).toContain("window.activeWindow.open");
    expect(resourceOpenerSource).not.toContain("globalThis");
  });

  it("uses the host DOMParser for markup parsing", () => {
    const parseFromString = vi.fn(() => ({}) as Document);
    const DomParser = vi.fn(function (this: { parseFromString: typeof parseFromString }) {
      this.parseFromString = parseFromString;
    });
    vi.stubGlobal("DOMParser", DomParser);

    new ObsidianMarkupParser().parseFromString("<feed />", "text/xml");

    expect(DomParser).toHaveBeenCalledTimes(1);
    expect(parseFromString).toHaveBeenCalledWith("<feed />", "text/xml");
    vi.unstubAllGlobals();
  });

  it("assembles vault storage, settings secrets, and resource opening", async () => {
    const { app, files, workspace } = testApp();
    const settings = testSettings();
    const persistSettings = vi.fn();
    const settingsChanges = new SettingsChangeService({
      settings,
      persistSettings,
    });

    const host = buildObsidianHostAdapters({
      app: app as any,
      getSettings: () => settings,
      changeSettingValue: (key, value) => settingsChanges.changeValue(key, value),
    });

    await host.storage.mkdir("arxiv-daily/daily");
    await host.storage.writeText("arxiv-daily/daily/today.md", "daily");
    await host.storage.writeTextAtomic?.("arxiv-daily/daily/atomic.md", "atomic");
    expect(files["arxiv-daily/daily/today.md"]).toBe("daily");
    expect(files["arxiv-daily/daily/atomic.md"]).toBe("atomic");
    expect(files["arxiv-daily/daily/atomic.md.tmp"]).toBeUndefined();
    expect(await host.storage.exists("arxiv-daily/daily")).toBe(true);

    expect(await host.secrets.getSecret("llm.apiKey")).toBe("stored-key");
    await host.secrets.setSecret?.("apiKey", "next-key");
    expect(settings.llm.apiKey).toBe("next-key");
    expect(persistSettings).toHaveBeenCalledTimes(1);

    await host.opener.openNote("arxiv-daily/papers/2606.12345.md", {
      newLeaf: true,
    });
    expect(workspace.openLinkText).toHaveBeenCalledWith(
      "arxiv-daily/papers/2606.12345.md",
      "",
      true,
    );
  });

  it("keeps raw-recipient delivery state and every atomic artifact at 0600 under umask 0002", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const adapter = realFilesystemAdapter(root);
    const observedArtifacts: Array<{ path: string; mode: number; content: string }> = [];
    const inspectArtifact = async (path: string) => {
      observedArtifacts.push({
        path,
        mode: await fileMode(path),
        content: await readFile(path, "utf8"),
      });
    };
    const storage = new ObsidianStorageAdapter(
      { adapter } as any,
      {
        privateAtomicWrite: {
          afterTemporaryFileReady: inspectArtifact,
          afterBackupFileReady: inspectArtifact,
        },
      },
    ) as StorageAdapter;
    const rawRecipient = "Private.Recipient@Example.COM";
    const previousUmask = process.umask(0o002);
    try {
      let state = emptyDeliveryState(new Date("2026-08-10T00:00:00.000Z"));
      state = markDelivered(state, {
        date: "2026-08-10",
        recipient: rawRecipient,
        attempts: 1,
        now: new Date("2026-08-10T00:00:01.000Z"),
      });
      await saveDeliveryState(
        storage,
        DEFAULT_SETTINGS.output,
        state,
        new Date("2026-08-10T00:00:02.000Z"),
      );
      await saveDeliveryState(
        storage,
        DEFAULT_SETTINGS.output,
        state,
        new Date("2026-08-10T00:00:03.000Z"),
      );

      const target = join(root, deliveryStatePath(DEFAULT_SETTINGS.output));
      expect(await fileMode(target)).toBe(0o600);
      expect(await readFile(target, "utf8")).toContain(rawRecipient);
      expect(observedArtifacts.some(({ path }) => path.includes(".tmp-"))).toBe(true);
      expect(observedArtifacts.some(({ path }) => path.includes(".bak-"))).toBe(true);
      expect(observedArtifacts.every(({ mode }) => mode === 0o600)).toBe(true);
      expect(observedArtifacts.every(({ content }) => content.includes(rawRecipient)))
        .toBe(true);
      expect((await fsReaddir(join(root, "arxiv-daily/.index"))).filter(
        (entry) =>
          entry.includes("delivery-state.json.tmp-") ||
          entry.includes("delivery-state.json.bak-"),
      )).toEqual([]);
    } finally {
      process.umask(previousUmask);
    }
  });

  it("keeps the primary present until the single tmp-to-target rename", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const adapter = realFilesystemAdapter(root);
    const targetPath = deliveryStatePath(DEFAULT_SETTINGS.output);
    const target = join(root, targetPath);
    await mkdir(join(target, ".."), { recursive: true });
    await writeFile(target, "legacy-v1", { encoding: "utf8", mode: 0o600 });
    let inspected = false;
    const storage = new ObsidianStorageAdapter(
      { adapter } as any,
      {
        privateAtomicWrite: {
          afterBackupFileReady: async (backup) => {
            inspected = true;
            const primaryExists = await stat(target).then(() => true, () => false);
            if (!primaryExists) throw new Error("primary was absent before install rename");
            expect(await readFile(target, "utf8")).toBe("legacy-v1");
            expect(await readFile(backup, "utf8")).toBe("legacy-v1");
            expect(await fileMode(target)).toBe(0o600);
            expect(await fileMode(backup)).toBe(0o600);
          },
        },
      },
    );

    await storage.writeTextAtomic(targetPath, "v2", 0o600);

    expect(inspected).toBe(true);
    expect(await readFile(target, "utf8")).toBe("v2");
    expect(await fileMode(target)).toBe(0o600);
  });

  it("tightens an existing legacy v1 primary to 0600 before reading it", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const adapter = realFilesystemAdapter(root);
    const storage = new ObsidianStorageAdapter({ adapter } as any);
    const targetPath = deliveryStatePath(DEFAULT_SETTINGS.output);
    const target = join(root, targetPath);
    await mkdir(join(target, ".."), { recursive: true });
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

  it("recovers a legacy v1 delivered record from backup-only state and cleans random artifacts", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const adapter = realFilesystemAdapter(root);
    const storage = new ObsidianStorageAdapter({ adapter } as any);
    const targetPath = deliveryStatePath(DEFAULT_SETTINGS.output);
    const target = join(root, targetPath);
    await mkdir(join(target, ".."), { recursive: true });
    const state = markDelivered(emptyDeliveryState(), {
      date: "2026-08-10",
      recipient: "legacy@example.com",
      attempts: 1,
    });
    const serialized = `${JSON.stringify(state, null, 2)}\n`;
    const backup = `${target}.bak-deadbeef`;
    const temporary = `${target}.tmp-cafebabe`;
    await writeFile(backup, serialized, { encoding: "utf8", mode: 0o644 });
    await writeFile(temporary, "future", { encoding: "utf8", mode: 0o644 });

    const read = await readDeliveryState(storage, DEFAULT_SETTINGS.output);

    expect(read.kind).toBe("valid");
    if (read.kind === "valid") {
      expect(shouldSendEmail(read.state, "2026-08-10", "legacy@example.com"))
        .toBe(false);
    }
    expect(await readFile(target, "utf8")).toBe(serialized);
    expect(await fileMode(target)).toBe(0o600);
    expect(await stat(backup).then(() => true, () => false)).toBe(false);
    expect(await stat(temporary).then(() => true, () => false)).toBe(false);
  });

  it("does not expose exclusive create for a non-filesystem Obsidian adapter", () => {
    const { app } = testApp();
    const storage = new ObsidianStorageAdapter(app.vault as any);

    expect(storage.createTextExclusive).toBeUndefined();
  });

  it("creates desktop claims without process.getBuiltinModule", async () => {
    const root = await makeTempDir();
    const filesystemAdapter = {
      ...testApp().app.vault.adapter,
      getBasePath: () => root,
    };
    const original = Object.getOwnPropertyDescriptor(process, "getBuiltinModule");
    Object.defineProperty(process, "getBuiltinModule", {
      configurable: true,
      value: undefined,
    });
    try {
      const storage = new ObsidianStorageAdapter({ adapter: filesystemAdapter } as any);

      await expect(
        storage.createTextExclusive!("claims/compatible.claim", "claimed"),
      ).resolves.toBe(true);
      await expect(readFile(join(root, "claims/compatible.claim"), "utf8"))
        .resolves.toBe("claimed");
    } finally {
      if (original) Object.defineProperty(process, "getBuiltinModule", original);
      else delete (process as { getBuiltinModule?: unknown }).getBuiltinModule;
    }
  });

  it("exclusively creates one file across desktop filesystem adapter instances", async () => {
    const root = await makeTempDir();
    const filesystemAdapter = {
      ...testApp().app.vault.adapter,
      getBasePath: () => root,
    };
    const first = new ObsidianStorageAdapter({ adapter: filesystemAdapter } as any);
    const second = new ObsidianStorageAdapter({ adapter: filesystemAdapter } as any);

    const results = await Promise.all([
      first.createTextExclusive!("claims/daily.lock", "first"),
      second.createTextExclusive!("claims/daily.lock", "second"),
    ]);

    expect(results.sort()).toEqual([false, true]);
    expect(["first", "second"]).toContain(
      await readFile(join(root, "claims/daily.lock"), "utf8"),
    );
  });

  it("rejects traversal and existing parent symlink escape on desktop", async () => {
    const root = await makeTempDir();
    const outside = await makeTempDir();
    await symlink(outside, join(root, "escape"), "dir");
    const filesystemAdapter = {
      ...testApp().app.vault.adapter,
      getBasePath: () => root,
    };
    const storage = new ObsidianStorageAdapter({ adapter: filesystemAdapter } as any);

    await expect(
      storage.createTextExclusive!("../outside.claim", "secret"),
    ).rejects.toThrow(/escapes vault|invalid vault-relative path/);
    await expect(
      storage.createTextExclusive!("escape/outside.claim", "secret"),
    ).rejects.toThrow(/symlink|escapes vault/);
    await expect(readFile(join(outside, "outside.claim"), "utf8"))
      .rejects.toMatchObject({ code: "ENOENT" });
  });

  it("anchors desktop exclusive create across a final-parent symlink swap", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const outside = await makeTempDir();
    const parent = join(root, "claims", "daily");
    const movedParent = join(root, "claims", "daily-opened");
    const filesystemAdapter = {
      ...testApp().app.vault.adapter,
      getBasePath: () => root,
    };
    let resume!: () => void;
    let validated!: () => void;
    const reachedBarrier = new Promise<void>((resolve) => { validated = resolve; });
    const barrier = new Promise<void>((resolve) => { resume = resolve; });

    const creating = createDesktopTextExclusive(
      filesystemAdapter,
      "claims/daily/send.claim",
      "claimed",
      {
        afterFinalParentOpened: async () => {
          validated();
          await barrier;
        },
      },
    );
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

  it("fails closed when an opened vault parent is moved outside and linked back", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const outside = await makeTempDir();
    const claims = join(root, "claims");
    const movedClaims = join(outside, "claims-moved");
    const filesystemAdapter = {
      ...testApp().app.vault.adapter,
      getBasePath: () => root,
    };

    const result = await createDesktopTextExclusive(
      filesystemAdapter,
      "claims/daily/send.claim",
      "claimed",
      {
        afterFinalParentOpened: async () => {
          await rename(claims, movedClaims);
          await symlink(movedClaims, claims, "dir");
        },
      },
    ).then(
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

  it("fails closed when an opened vault parent moves within the vault and is linked back", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const claims = join(root, "claims");
    const movedClaims = join(root, "claims-moved-within-vault");
    const filesystemAdapter = {
      ...testApp().app.vault.adapter,
      getBasePath: () => root,
    };

    const result = await createDesktopTextExclusive(
      filesystemAdapter,
      "claims/daily/send.claim",
      "claimed",
      {
        afterFinalParentOpened: async () => {
          await rename(claims, movedClaims);
          await symlink(movedClaims, claims, "dir");
        },
      },
    ).then(
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

  it("removes a desktop claim created just before its parent leaves the vault", async () => {
    if (process.platform !== "linux") return;
    const root = await makeTempDir();
    const outside = await makeTempDir();
    const claims = join(root, "claims");
    const movedClaims = join(outside, "claims-post-create");
    const filesystemAdapter = {
      ...testApp().app.vault.adapter,
      getBasePath: () => root,
    };

    await expect(
      createDesktopTextExclusive(
        filesystemAdapter,
        "claims/daily/send.claim",
        "claimed",
        {
          afterTargetCreated: async () => {
            await rename(claims, movedClaims);
            await symlink(movedClaims, claims, "dir");
          },
        },
      ),
    ).rejects.toThrow(/escapes vault|replaced/);
    expect(
      await readFile(join(movedClaims, "daily/send.claim"), "utf8").catch(
        () => null,
      ),
    ).toBeNull();
  });

  it("keeps the live host secret unchanged until candidate persistence succeeds", async () => {
    const { app } = testApp();
    const settings = testSettings();
    let finishSave: (() => void) | undefined;
    const persistSettings = vi.fn(async (candidate: PluginSettings) => {
      expect(candidate.llm.apiKey).toBe("next-key");
      await new Promise<void>((resolve) => { finishSave = resolve; });
    });
    const settingsChanges = new SettingsChangeService({ settings, persistSettings });
    const host = buildObsidianHostAdapters({
      app: app as any,
      getSettings: () => settings,
      changeSettingValue: (key, value) => settingsChanges.changeValue(key, value),
    });

    const changing = host.secrets.setSecret?.("llm.apiKey", "next-key");
    await vi.waitFor(() => expect(persistSettings).toHaveBeenCalledOnce());
    expect(settings.llm.apiKey).toBe("stored-key");
    finishSave?.();
    await changing;

    expect(settings.llm.apiKey).toBe("next-key");
  });

  it("restores the live host secret when candidate persistence fails", async () => {
    const { app } = testApp();
    const settings = testSettings();
    const settingsChanges = new SettingsChangeService({
      settings,
      persistSettings: vi.fn().mockRejectedValue(new Error("disk full")),
    });
    const host = buildObsidianHostAdapters({
      app: app as any,
      getSettings: () => settings,
      changeSettingValue: (key, value) => settingsChanges.changeValue(key, value),
    });

    await expect(host.secrets.deleteSecret?.("apiKey")).rejects.toThrow("disk full");
    expect(settings.llm.apiKey).toBe("stored-key");
  });

  it("rotates checkpoint backups with Obsidian rename semantics", async () => {
    const { app, files } = testApp();
    const host = buildObsidianHostAdapters({
      app: app as any,
      getSettings: testSettings,
    });
    const store = new DailySummaryCheckpointStore(host.storage, DEFAULT_SETTINGS.output);
    const input = (id: string): DailySummaryCheckpointCompatibilityInput => ({
      paper: {
        id,
        title: `Paper ${id}`,
        authors: "Author",
        abstract: `Abstract ${id}`,
        abstractConclusion: `Content ${id}`,
      },
      llm: {
        provider: "custom",
        baseUrl: "https://example.test/v1",
        model: "model-a",
        thinkingMode: false,
        reasoningEffort: "medium",
      },
    });
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

    for (const id of ["2608.00001", "2608.00002", "2608.00003"]) {
      await store.upsert("2026-08-01", input(id), result(id));
    }
    const paths = store.pathsFor("2026-08-01");
    files[paths.documentPath] = "corrupt";

    await expect(new DailySummaryCheckpointStore(host.storage, DEFAULT_SETTINGS.output)
      .lookupReusable("2026-08-01", input("2608.00001")))
      .resolves.toEqual(result("2608.00001"));
    await expect(new DailySummaryCheckpointStore(host.storage, DEFAULT_SETTINGS.output)
      .lookupReusable("2026-08-01", input("2608.00002")))
      .resolves.toEqual(result("2608.00002"));
    expect(files[`${paths.documentPath}.tmp`]).toBeUndefined();
    expect(files[`${paths.backupPath}.tmp`]).toBeUndefined();
  });

  it("reconstructs filter checkpoints from backup and cleans every Obsidian file", async () => {
    const { app, files } = testApp();
    const host = buildObsidianHostAdapters({
      app: app as any,
      getSettings: testSettings,
    });
    const store = new DailyFilterCheckpointStore(host.storage, DEFAULT_SETTINGS.output);
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
    files[paths.documentPath] = "corrupt";

    await expect(new DailyFilterCheckpointStore(host.storage, DEFAULT_SETTINGS.output)
      .lookupReusable("2026-08-01", prepared)).resolves.toEqual(first);
    expect(JSON.parse(files[paths.backupPath]!)).toMatchObject({ result: first });

    await new DailyFilterCheckpointStore(host.storage, DEFAULT_SETTINGS.output)
      .removeAll("2026-08-01");
    for (const path of [
      paths.documentPath,
      paths.backupPath,
      `${paths.documentPath}.tmp`,
      `${paths.backupPath}.tmp`,
    ]) {
      expect(files[path]).toBeUndefined();
    }
  });

  it("atomically preserves scientific Markdown bytes without touching plugin data", async () => {
    const { app, files } = testApp();
    const host = buildObsidianHostAdapters({
      app: app as any,
      getSettings: testSettings,
    });
    const body = String.raw`- **Core results**: $\mathrm{NMAD}$ and $\eta$ hold for \(r_{\rm cut}/R_{\rm vir}\), z<0.1, z>3.5, and A & B.`;

    await host.storage.writeTextAtomic?.("arxiv-daily/daily/science.md", body);

    expect(files["arxiv-daily/daily/science.md"]).toBe(body);
    expect(files["arxiv-daily/daily/science.md.tmp"]).toBeUndefined();
    expect(files["arxiv-daily/daily/science.md.bak"]).toBeUndefined();
    expect(Object.keys(files)).toEqual(["arxiv-daily/daily/science.md"]);
  });

  it("injects the Obsidian request implementation through the host builder", async () => {
    const { app } = testApp();
    const request = vi.fn(async () => ({
      status: 200,
      headers: {},
      text: "injected",
    }));
    const host = buildObsidianHostAdapters({
      app: app as any,
      getSettings: testSettings,
      request,
    });

    await expect(host.http.request({ url: "https://example.test" }))
      .resolves.toMatchObject({ bodyText: "injected" });
    expect(request).toHaveBeenCalledOnce();
  });

  it("settles logically on timeout and consumes a late rejection", async () => {
    vi.useFakeTimers();
    let rejectRequest!: (error: Error) => void;
    const request = vi.fn(() => new Promise<any>((_resolve, reject) => {
      rejectRequest = reject;
    }));
    const client = new ObsidianHttpClient(request);

    const result = client.request({ url: "https://example.test", timeoutMs: 25 });
    const assertion = expect(result).rejects.toMatchObject({
      name: "HttpTransportError",
      kind: "timeout",
    });
    await vi.advanceTimersByTimeAsync(25);
    await assertion;

    rejectRequest(new Error("late request failure"));
    await Promise.resolve();
  });

  it("does not immediately retry an Obsidian timeout while physical requestUrl may run", async () => {
    vi.useFakeTimers();
    let active = 0;
    const request = vi.fn(() => {
      active += 1;
      if (request.mock.calls.length === 1) return new Promise<any>(() => {});
      active -= 1;
      return Promise.resolve({ status: 200, headers: {}, text: "later success" });
    });
    const fetcher = new ArxivFetcher({
      categories: ["astro-ph"],
      http: new ObsidianHttpClient(request),
      markupParser: new ObsidianMarkupParser(),
      logger: new Logger("error"),
      requestDelayMs: 0,
      textTimeoutMs: 25,
    });

    const first = fetcher.fetchRecent();
    const firstAssertion = expect(first).rejects.toMatchObject({
      kind: "timeout",
      retryableAttempt: false,
    });
    await vi.advanceTimersByTimeAsync(25);
    await firstAssertion;
    await vi.advanceTimersByTimeAsync(60_000);
    expect(request).toHaveBeenCalledOnce();
    expect(active).toBe(1);

    const later = fetcher.fetchRecent();
    await vi.advanceTimersByTimeAsync(3_000);
    await expect(later).resolves.toBe("later success");
    expect(request).toHaveBeenCalledTimes(2);
    // The first physical request is unavoidably still live because requestUrl
    // exposes no cancellation primitive; only a later invocation may overlap it.
    expect(active).toBe(1);
  });

  it("settles logically on cancellation without classifying it as transport", async () => {
    const controller = new AbortController();
    const client = new ObsidianHttpClient(() => new Promise<any>(() => {}));
    const result = client.request({
      url: "https://example.test",
      timeoutMs: 60_000,
      signal: controller.signal,
    });

    controller.abort("stop now");

    await expect(result).rejects.toSatisfy(isCancellationError);
  });

  it("normalizes request rejection as a network transport error", async () => {
    const client = new ObsidianHttpClient(async () => {
      throw new Error("requestUrl failed");
    });

    await expect(client.request({ url: "https://example.test" })).rejects.toMatchObject({
      name: "HttpTransportError",
      kind: "network",
      retryableAttempt: false,
    });
  });

  it("opens URLs through Obsidian's active window", async () => {
    const { app } = testApp();
    const activeWindow = { open: vi.fn(() => null) } as unknown as Window;
    Object.defineProperty(window, "activeWindow", {
      configurable: true,
      value: activeWindow,
    });
    const open = activeWindow.open as ReturnType<typeof vi.fn>;
    const host = buildObsidianHostAdapters({
      app: app as any,
      getSettings: testSettings,
    });

    await host.opener.openUrl("https://arxiv.org/abs/2606.12345");

    expect(open).toHaveBeenCalledWith(
      "https://arxiv.org/abs/2606.12345",
      "_blank",
      "noopener",
    );
  });
});
