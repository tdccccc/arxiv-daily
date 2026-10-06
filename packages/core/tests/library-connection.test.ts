import { describe, expect, it } from "vitest";
import {
  authorizeLibraryConnection,
  createLibraryConnection,
  libraryAuthorizationDisclosure,
  libraryConnectionStatus,
} from "../src/library/library-connection";

describe("library connection endpoint disclosure", () => {
  it("redacts repeated query parameters and the parameters after them", () => {
    const connection = createLibraryConnection("/papers", "1:2");
    const disclosure = libraryAuthorizationDisclosure(connection, {
      llmBaseUrl: "https://user:secret@example.com/v1?token=first&token=second&after=hidden#private",
    });

    expect(disclosure.endpoint).toBe(
      "https://example.com/v1?token=%5Bredacted%5D&after=%5Bredacted%5D",
    );
  });
});

describe("library processing depth authorization", () => {
  const localScope = { llmBaseUrl: "https://model.example/v1" };
  const remoteScope = { ...localScope, embeddingEndpoint: { baseUrl: "https://embedding.example/v1" } };

  it("discloses and grants the same metadata scope after switching from remote to local embedding", () => {
    const remote = authorizeLibraryConnection(createLibraryConnection("/papers", "1:2"), remoteScope);
    expect(libraryConnectionStatus(remote, remoteScope).kind).toBe("authorized");
    expect(libraryConnectionStatus(remote, localScope).kind).toBe("authorization-invalidated");

    const disclosure = libraryAuthorizationDisclosure(remote, localScope);
    const local = authorizeLibraryConnection(remote, localScope);
    expect.soft(disclosure.processingDepth).toBe("metadata-and-abstracts");
    expect(disclosure.embeddingEndpoint).toBeUndefined();
    expect(local.processingDepth).toBe("metadata-and-abstracts");
    expect(local.authorization?.fingerprint).toBe(disclosure.authorizationFingerprint);
    expect.soft(libraryConnectionStatus(local, localScope).kind).toBe("authorized");
    expect(remote.processingDepth).toBe("full-text");
    expect(libraryConnectionStatus(remote, remoteScope).kind).toBe("authorized");
  });

  it.each(["local", "remote"])("invalidates an existing %s grant if its persisted processing depth is changed", (mode) => {
    const scope = mode === "remote" ? remoteScope : localScope;
    const authorized = authorizeLibraryConnection(createLibraryConnection("/papers", "1:2"), scope);
    const tampered = {
      ...authorized,
      processingDepth: authorized.processingDepth === "full-text" ? "metadata-and-abstracts" as const : "full-text" as const,
    };
    expect(libraryConnectionStatus(tampered, scope).kind).toBe("authorization-invalidated");
  });
});
