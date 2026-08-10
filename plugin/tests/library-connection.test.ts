import { describe, expect, it } from "vitest";
import type { LibraryInventory } from "@arxiv-daily/core";
import {
  authorizeLibraryConnection,
  buildLibraryInventoryPreview,
  createLibraryConnection,
  decodeLibraryConnection,
  libraryAuthorizationDisclosure,
  libraryAuthorizationFingerprint,
  libraryConnectionStatus,
  revokeLibraryConnection,
} from "../src/library/connection";

const endpoint = "https://user:secret@example.com/v1?token=hidden#fragment";
const scope = (llmBaseUrl: string, embeddingBaseUrl?: string) => ({
  llmBaseUrl,
  ...(embeddingBaseUrl ? { embeddingEndpoint: { baseUrl: embeddingBaseUrl } } : {}),
});

describe("personal library connection", () => {
  it("decodes the supported persisted shape and rejects malformed values", () => {
    const decoded = decodeLibraryConnection({
      schemaVersion: 1,
      selectedRoot: "/papers",
      rootIdentity: "1:2",
      eligibleExtensions: [".PDF", ".pdf"],
      processingDepth: "metadata-and-abstracts",
      authorization: {
        fingerprint: `sha256:${"a".repeat(64)}`,
        grantedAt: "2026-08-02T12:00:00.000Z",
      },
    });

    expect(decoded).toEqual({
      schemaVersion: 1,
      selectedRoot: "/papers",
      rootIdentity: "1:2",
      eligibleExtensions: [".pdf"],
      processingDepth: "metadata-and-abstracts",
      authorization: {
        fingerprint: `sha256:${"a".repeat(64)}`,
        grantedAt: "2026-08-02T12:00:00.000Z",
      },
    });
    expect(decodeLibraryConnection(null)).toBeUndefined();
    expect(decodeLibraryConnection({ schemaVersion: 2 })).toBeUndefined();
    expect(decodeLibraryConnection({
      schemaVersion: 1,
      selectedRoot: "",
      eligibleExtensions: [".pdf"],
      processingDepth: "metadata-and-abstracts",
    })).toBeUndefined();
    expect(decodeLibraryConnection({
      schemaVersion: 1,
      selectedRoot: "/papers",
      eligibleExtensions: ["pdf"],
      processingDepth: "metadata-and-abstracts",
    })).toBeUndefined();
  });

  it("grants, invalidates, and revokes endpoint-bound authorization", () => {
    const connection = createLibraryConnection("/papers", "1:2");
    expect(libraryConnectionStatus(undefined, scope(endpoint))).toEqual({ kind: "disconnected" });
    expect(libraryConnectionStatus(connection, scope(endpoint))).toEqual({
      kind: "authorization-required",
      rootLabel: "papers",
    });

    const authorized = authorizeLibraryConnection(
      connection,
      scope(endpoint),
      new Date("2026-08-02T12:00:00.000Z"),
    );
    expect(libraryConnectionStatus(authorized, scope(endpoint))).toEqual({
      kind: "authorized",
      rootLabel: "papers",
      grantedAt: "2026-08-02T12:00:00.000Z",
    });
    expect(libraryConnectionStatus(authorized, scope("https://other.example/v1")).kind)
      .toBe("authorization-invalidated");
    expect(libraryConnectionStatus(revokeLibraryConnection(authorized), scope(endpoint)).kind)
      .toBe("authorization-required");
  });

  it("keeps status evaluation total when the endpoint can no longer be digested", () => {
    const connection = createLibraryConnection("/papers", "1:2");
    const authorized = authorizeLibraryConnection(connection, scope(endpoint));

    // An invalid or non-http(s) endpoint shape invalidates the grant instead
    // of throwing into the settings tab or daily-run snapshot path.
    for (const broken of ["not a url", "ftp://example.com/v1", ""]) {
      expect(() => libraryConnectionStatus(authorized, scope(broken))).not.toThrow();
      expect(libraryConnectionStatus(authorized, scope(broken)).kind)
        .toBe("authorization-invalidated");
    }
    // Pre-grant states never touch the endpoint digest at all.
    expect(libraryConnectionStatus(undefined, scope("not a url")))
      .toEqual({ kind: "disconnected" });
    expect(libraryConnectionStatus(connection, scope("not a url")).kind)
      .toBe("authorization-required");
  });

  it("normalizes file types and binds the fingerprint to scope and processing terms", () => {
    const base = createLibraryConnection("/papers", "1:2");
    const reordered = { ...base, eligibleExtensions: [".PDF", ".pdf"] };
    expect(libraryAuthorizationFingerprint(reordered, scope(endpoint)))
      .toBe(libraryAuthorizationFingerprint(base, scope(endpoint)));
    expect(libraryAuthorizationFingerprint(
      { ...base, selectedRoot: "/other" },
      scope(endpoint),
    )).not.toBe(libraryAuthorizationFingerprint(base, scope(endpoint)));
    expect(libraryAuthorizationFingerprint(
      { ...base, rootIdentity: "1:3" },
      scope(endpoint),
    )).not.toBe(libraryAuthorizationFingerprint(base, scope(endpoint)));
    expect(libraryAuthorizationFingerprint(
      { ...base, eligibleExtensions: [".md"] },
      scope(endpoint),
    )).not.toBe(libraryAuthorizationFingerprint(base, scope(endpoint)));
  });

  it("discloses the exact scope while stripping endpoint credentials and decorations", () => {
    const connection = createLibraryConnection("/private/research/papers", "1:2");
    const disclosure = libraryAuthorizationDisclosure(connection, scope(endpoint));

    expect(disclosure.selectedRoot).toBe("/private/research/papers");
    expect(disclosure.eligibleExtensions).toEqual([".pdf"]);
    expect(disclosure.endpoint).toBe(
      "https://example.com/v1?token=%5Bredacted%5D",
    );
    expect(disclosure.endpoint).not.toContain("secret");
    expect(disclosure.endpoint).not.toContain("hidden");
    expect(disclosure.authorizationFingerprint).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(JSON.stringify(connection)).not.toContain("secret");
  });

  it("binds remote-embedding grants to the embedding endpoint and full-text depth", () => {
    const connection = createLibraryConnection("/papers", "1:2");
    const local = authorizeLibraryConnection(connection, scope(endpoint));
    const remote = authorizeLibraryConnection(
      connection,
      scope(endpoint, "https://embed.example.com/v1"),
      new Date("2026-08-02T12:00:00.000Z"),
    );

    // Remote grants carry the full-text depth; local stays metadata-level.
    expect(local.processingDepth).toBe("metadata-and-abstracts");
    expect(remote.processingDepth).toBe("full-text");
    // The fingerprints differ (embedding endpoint + depth are part of the grant).
    expect(remote.authorization?.fingerprint).not.toBe(local.authorization?.fingerprint);

    // Status stays authorized only while the scope matches the grant.
    expect(libraryConnectionStatus(remote, scope(endpoint, "https://embed.example.com/v1")).kind)
      .toBe("authorized");
    // Missing the embedding endpoint invalidates a remote grant.
    expect(libraryConnectionStatus(remote, scope(endpoint)).kind)
      .toBe("authorization-invalidated");
    // A different embedding endpoint invalidates it too.
    expect(libraryConnectionStatus(remote, scope(endpoint, "https://embed-other.example.com/v1")).kind)
      .toBe("authorization-invalidated");

    // Disclosure surfaces the embedding endpoint and full-text depth.
    const disclosure = libraryAuthorizationDisclosure(
      remote,
      scope(endpoint, "https://embed.example.com/v1"),
    );
    expect(disclosure.processingDepth).toBe("full-text");
    expect(disclosure.embeddingEndpoint).toContain("embed.example.com");
    // Local disclosure has no embedding endpoint line.
    const localDisclosure = libraryAuthorizationDisclosure(local, scope(endpoint));
    expect(localDisclosure.embeddingEndpoint).toBeUndefined();
    expect(localDisclosure.processingDepth).toBe("metadata-and-abstracts");
  });

  it("decodes persisted full-text connections", () => {
    const connection = createLibraryConnection("/papers", "1:2");
    const remote = authorizeLibraryConnection(connection, scope(endpoint, "https://embed.example.com/v1"));
    expect(decodeLibraryConnection(JSON.parse(JSON.stringify(remote)))?.processingDepth)
      .toBe("full-text");
  });
});

describe("personal library inventory preview", () => {
  it("classifies PDFs locally and accounts for ignored entries and folders", () => {
    const inventory: LibraryInventory = {
      truncated: true,
      entries: [
        { path: "paper.PDF", type: "file", size: 12 },
        { path: "notes.md", type: "file", size: 8 },
        { path: "nested", type: "folder" },
        { path: "outside", type: "ignored", ignoredReason: "symbolic-link" },
        { path: "socket", type: "ignored", ignoredReason: "unsupported-entry" },
      ],
    };

    expect(buildLibraryInventoryPreview(inventory, [".pdf"])).toEqual({
      eligible: [{ path: "paper.PDF", size: 12 }],
      ignored: [
        { path: "notes.md", reason: "Unsupported file type" },
        { path: "outside", reason: "Symbolic link" },
        { path: "socket", reason: "Unsupported filesystem entry" },
      ],
      folders: 1,
      truncated: true,
    });
  });
});
