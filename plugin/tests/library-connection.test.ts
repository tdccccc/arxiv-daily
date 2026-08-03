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
    expect(libraryConnectionStatus(undefined, endpoint)).toEqual({ kind: "disconnected" });
    expect(libraryConnectionStatus(connection, endpoint)).toEqual({
      kind: "authorization-required",
      rootLabel: "papers",
    });

    const authorized = authorizeLibraryConnection(
      connection,
      endpoint,
      new Date("2026-08-02T12:00:00.000Z"),
    );
    expect(libraryConnectionStatus(authorized, endpoint)).toEqual({
      kind: "authorized",
      rootLabel: "papers",
      grantedAt: "2026-08-02T12:00:00.000Z",
    });
    expect(libraryConnectionStatus(authorized, "https://other.example/v1").kind)
      .toBe("authorization-invalidated");
    expect(libraryConnectionStatus(revokeLibraryConnection(authorized), endpoint).kind)
      .toBe("authorization-required");
  });

  it("normalizes file types and binds the fingerprint to scope and processing terms", () => {
    const base = createLibraryConnection("/papers", "1:2");
    const reordered = { ...base, eligibleExtensions: [".PDF", ".pdf"] };
    expect(libraryAuthorizationFingerprint(reordered, endpoint))
      .toBe(libraryAuthorizationFingerprint(base, endpoint));
    expect(libraryAuthorizationFingerprint(
      { ...base, selectedRoot: "/other" },
      endpoint,
    )).not.toBe(libraryAuthorizationFingerprint(base, endpoint));
    expect(libraryAuthorizationFingerprint(
      { ...base, rootIdentity: "1:3" },
      endpoint,
    )).not.toBe(libraryAuthorizationFingerprint(base, endpoint));
    expect(libraryAuthorizationFingerprint(
      { ...base, eligibleExtensions: [".md"] },
      endpoint,
    )).not.toBe(libraryAuthorizationFingerprint(base, endpoint));
  });

  it("discloses the exact scope while stripping endpoint credentials and decorations", () => {
    const connection = createLibraryConnection("/private/research/papers", "1:2");
    const disclosure = libraryAuthorizationDisclosure(connection, endpoint);

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
