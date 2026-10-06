import { afterEach, describe, expect, it } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { parse, stringify } from "smol-toml";
import {
  authorizeLibraryConnection,
  createLibraryConnection,
  sha256Hex,
} from "@arxiv-daily/core";
import { loadCliConfig } from "../src/config";
import {
  authorizeCliLibrary,
  connectCliLibrary,
  inspectCliLibraryConnection,
  revokeCliLibrary,
} from "../src/library-connection-cmd";

const roots: string[] = [];
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => fs.rm(root, { recursive: true, force: true })));
});

describe("CLI library connection commands", () => {
  it("connects a read-only source without granting model authorization and preserves every other setting", async () => {
    const { config, configPath, selectedRoot, document } = await fixture();
    const before = await fs.readFile(path.join(selectedRoot, "source.pdf"));
    const next = await connectCliLibrary(config, selectedRoot);
    expect(next.libraryConnection).toMatchObject({
      schemaVersion: 1,
      selectedRoot: await fs.realpath(selectedRoot),
      eligibleExtensions: [".pdf"],
      processingDepth: "metadata-and-abstracts",
    });
    expect(next.libraryConnection?.rootIdentity).toMatch(/^\d+:\d+$/);
    expect(next.libraryConnection?.authorization).toBeUndefined();
    const persisted = parse(await fs.readFile(configPath, "utf8"));
    expect(persisted.library).toEqual(next.libraryConnection);
    delete persisted.library;
    expect(persisted).toEqual(document);
    expect(next.settings.llm.apiKey).toBe(config.settings.llm.apiKey);
    expect(next.settings.embedding.apiKey).toBe(config.settings.embedding.apiKey);
    expect(next.settings.email.apiKey).toBe(config.settings.email.apiKey);
    expect(await fs.readFile(path.join(selectedRoot, "source.pdf"))).toEqual(before);
    expect(await fs.readdir(selectedRoot)).toEqual(["source.pdf"]);
    if (process.platform !== "win32") expect((await fs.stat(configPath)).mode & 0o777).toBe(0o600);
  });

  it("requires the displayed scope fingerprint before granting and can revoke the saved grant", async () => {
    const { config, configPath, selectedRoot } = await fixture();
    const connected = await connectCliLibrary(config, selectedRoot);
    const inspection = inspectCliLibraryConnection(connected);
    expect(inspection.status.kind).toBe("authorization-required");
    expect(inspection.disclosure?.processingDepth).toBe("full-text");
    const inspectionText = JSON.stringify(inspection);
    for (const secret of ["sk-llm-private", "sk-embed-private", "email-private", "password", "llm-secret", "embed-secret"]) {
      expect(inspectionText).not.toContain(secret);
    }
    const before = await fs.readFile(configPath, "utf8");
    await expect(authorizeCliLibrary(connected, `sha256:${"0".repeat(64)}`)).rejects.toThrow(/fingerprint|disclosure/i);
    expect(await fs.readFile(configPath, "utf8")).toBe(before);

    const authorized = await authorizeCliLibrary(connected, inspection.disclosure!.authorizationFingerprint);
    expect(inspectCliLibraryConnection(authorized).status.kind).toBe("authorized");
    expect(authorized.libraryConnection?.processingDepth).toBe("full-text");
    const revoked = await revokeCliLibrary(authorized);
    expect(revoked.libraryConnection?.authorization).toBeUndefined();
    expect(inspectCliLibraryConnection(revoked).status.kind).toBe("authorization-required");
    expect(revoked.settings).toEqual(config.settings);
    expect(await fs.readFile(path.join(selectedRoot, "source.pdf"), "utf8")).toBe("original paper bytes");
  });

  it.each(["llm", "embedding"])("invalidates authorization after the %s endpoint changes", async (table) => {
    const { config, configPath, selectedRoot } = await fixture();
    const connected = await connectCliLibrary(config, selectedRoot);
    const oldFingerprint = inspectCliLibraryConnection(connected).disclosure!.authorizationFingerprint;
    await authorizeCliLibrary(connected, oldFingerprint);
    const raw = parse(await fs.readFile(configPath, "utf8"));
    (raw[table] as Record<string, unknown>).base_url = "https://changed.example/v1";
    await fs.writeFile(configPath, stringify(raw));
    const changed = await loadCliConfig({ configPath });
    expect(inspectCliLibraryConnection(changed).status.kind).toBe("authorization-invalidated");
    const before = await fs.readFile(configPath, "utf8");
    await expect(authorizeCliLibrary(changed, oldFingerprint)).rejects.toThrow(/fingerprint|disclosure/i);
    expect(await fs.readFile(configPath, "utf8")).toBe(before);
  });

  it("rejects stale config revisions without losing external settings changes", async () => {
    const { config, configPath, selectedRoot, raw } = await fixture();
    const edited = `${raw}\n# editor changed the file\n`;
    await fs.writeFile(configPath, edited);
    await expect(connectCliLibrary(config, selectedRoot)).rejects.toThrow(/changed|stale|reload/i);
    expect(await fs.readFile(configPath, "utf8")).toBe(edited);
    await expect(connectCliLibrary({ ...config, configRevision: undefined }, selectedRoot)).rejects.toThrow(/reload|revision/i);
  });

  it("serializes concurrent writers and rejects the losing stale request", async () => {
    const { config, configPath, selectedRoot } = await fixture();
    const results = await Promise.allSettled([
      connectCliLibrary(config, selectedRoot),
      connectCliLibrary(config, selectedRoot),
    ]);
    expect(results.filter((result) => result.status === "fulfilled")).toHaveLength(1);
    const rejected = results.find((result) => result.status === "rejected") as PromiseRejectedResult;
    expect(rejected.reason).toBeInstanceOf(Error);
    expect(rejected.reason.message).toMatch(/changed|stale|reload/i);
    expect((await loadCliConfig({ configPath })).libraryConnection).toBeDefined();
  });

  it("refuses to authorize a replacement directory at the same path", async () => {
    const { config, configPath, selectedRoot } = await fixture();
    const connected = await connectCliLibrary(config, selectedRoot);
    const fingerprint = inspectCliLibraryConnection(connected).disclosure!.authorizationFingerprint;
    await fs.rename(selectedRoot, `${selectedRoot}-original`);
    await fs.mkdir(selectedRoot);
    const before = await fs.readFile(configPath, "utf8");
    await expect(authorizeCliLibrary(connected, fingerprint)).rejects.toThrow(/identity|reconnect|folder.*changed/i);
    expect(await fs.readFile(configPath, "utf8")).toBe(before);
  });

  it("reconnecting after an authorization clears the prior grant", async () => {
    const { config, selectedRoot, root } = await fixture();
    const connected = await connectCliLibrary(config, selectedRoot);
    const authorized = await authorizeCliLibrary(connected, inspectCliLibraryConnection(connected).disclosure!.authorizationFingerprint);
    const otherRoot = path.join(root, "other-papers");
    await fs.mkdir(otherRoot);
    const replaced = await connectCliLibrary(authorized, otherRoot);
    expect(replaced.libraryConnection?.selectedRoot).toBe(await fs.realpath(otherRoot));
    expect(replaced.libraryConnection?.authorization).toBeUndefined();
  });

  it("uses current persisted endpoints instead of caller-mutated settings when authorizing", async () => {
    const { config, selectedRoot } = await fixture();
    const connected = await connectCliLibrary(config, selectedRoot);
    connected.settings.llm.baseUrl = "https://caller-changed.example/v1";
    const fingerprint = inspectCliLibraryConnection(connected).disclosure!.authorizationFingerprint;
    await expect(authorizeCliLibrary(connected, fingerprint)).rejects.toThrow(/fingerprint|disclosure/i);
  });

  it("keeps inspection safe for absent, corrupt, and invalid-endpoint library settings", async () => {
    const { config, selectedRoot, configPath, document } = await fixture();
    expect(inspectCliLibraryConnection(config)).toEqual({ status: { kind: "disconnected" }, disclosure: null });
    await fs.writeFile(configPath, stringify({ ...document, library: "private-broken-library" }));
    const corrupted = await loadCliConfig({ configPath });
    const inspection = inspectCliLibraryConnection(corrupted);
    expect(inspection.status.kind).toBe("disconnected");
    expect(inspection.diagnostic).toMatch(/library/i);
    expect(JSON.stringify(inspection)).not.toContain("private-broken-library");
    const connected = await connectCliLibrary(corrupted, selectedRoot);
    connected.settings.llm.baseUrl = "invalid-private-endpoint";
    const invalid = inspectCliLibraryConnection(connected);
    expect(invalid.disclosure).toBeNull();
    expect(invalid.diagnostic).toMatch(/endpoint/i);
    expect(JSON.stringify(invalid)).not.toContain("invalid-private-endpoint");
  });
});

async function fixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "arxiv-cli-library-"));
  roots.push(root);
  const selectedRoot = path.join(root, "papers");
  await fs.mkdir(selectedRoot);
  await fs.writeFile(path.join(selectedRoot, "source.pdf"), "original paper bytes");
  const configPath = path.join(root, "config.toml");
  const document = {
    schema_version: 1,
    vault_root: path.join(root, "vault"),
    llm: { api_key: "sk-llm-private", base_url: "https://user:password@model.example/v1?token=llm-secret", model: "test-model" },
    embedding: { mode: "remote", api_key: "sk-embed-private", base_url: "https://embedding.example/v1?token=embed-secret", model: "test-embedding", dimension: 4 },
    arxiv: { categories: ["cs.AI"], topics: [{ name: "Agents", tag: "agents", description: "Reliable agents", detail: true }] },
    output: { daily_dir: "research/daily", papers_dir: "research/papers", summary_language: "en" },
    email: { enabled: false, api_key: "email-private", to: "researcher@example.org" },
    schedule: { enabled: true, on: "10:00", interval_hours: 0, until: "18:00", weekdays_only: true },
    future_setting: { enabled: true, values: ["one", "two"] },
  };
  const raw = stringify(document);
  await fs.writeFile(configPath, raw, { mode: 0o644 });
  return { root, selectedRoot, configPath, raw, document, config: await loadCliConfig({ configPath }) };
}

describe("CLI optional library config", () => {
  it("keeps the manual-topic path without a library and fingerprints the exact config bytes", async () => {
    const { raw, config } = await fixture();
    expect(config.libraryConnection).toBeUndefined();
    expect(config.libraryConnectionError).toBeUndefined();
    expect(config.configRevision).toBe(`sha256:${sha256Hex(raw)}`);
    expect(config.settings.arxiv.topics[0]?.tag).toBe("agents");
  });

  it("loads the existing camelCase connection schema and authorization unchanged", async () => {
    const { configPath, document } = await fixture();
    const connection = authorizeLibraryConnection(createLibraryConnection("/papers", "1:2"), {
      llmBaseUrl: document.llm.base_url,
      embeddingEndpoint: { baseUrl: document.embedding.base_url },
    }, new Date("2026-10-01T00:00:00Z"));
    await fs.writeFile(configPath, stringify({ ...document, library: connection }));
    const loaded = await loadCliConfig({ configPath });
    expect(loaded.libraryConnection).toEqual(connection);
    expect(loaded.libraryConnectionError).toBeUndefined();
  });

  it.each(["not-a-table", { schemaVersion: 2 }, { schemaVersion: 1, selectedRoot: "sk-bad-private" }])(
    "keeps manual settings usable when the optional library is invalid (%j)", async (library) => {
      const { configPath, document } = await fixture();
      await fs.writeFile(configPath, stringify({ ...document, library }));
      const config = await loadCliConfig({ configPath });
      expect(config.libraryConnection).toBeUndefined();
      expect(config.libraryConnectionError).toMatch(/library/i);
      expect(config.libraryConnectionError).not.toContain("sk-bad-private");
      expect(config.settings.arxiv.topics[0]?.tag).toBe("agents");
    },
  );

  it("surfaces a malformed authorization while retaining its disconnected grant", async () => {
    const { configPath, document } = await fixture();
    await fs.writeFile(configPath, stringify({ ...document, library: {
      ...createLibraryConnection("/papers", "1:2"),
      authorization: { fingerprint: "private-invalid-grant", grantedAt: "not-a-date" },
    } }));
    const config = await loadCliConfig({ configPath });
    expect(config.libraryConnection?.authorization).toBeUndefined();
    expect(config.libraryConnectionError).toMatch(/authorization/i);
    expect(config.libraryConnectionError).not.toContain("private-invalid-grant");
  });
});
