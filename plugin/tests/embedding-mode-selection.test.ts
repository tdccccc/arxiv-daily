import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS, type EmbeddingSettings } from "@arxiv-daily/core";
import ArxivDailyPlugin from "../main.ts";

function pluginWith(embedding: Partial<EmbeddingSettings>): any {
  const plugin = Object.create(ArxivDailyPlugin.prototype);
  Object.assign(plugin, {
    settings: {
      ...DEFAULT_SETTINGS,
      embedding: { ...DEFAULT_SETTINGS.embedding, ...embedding },
    },
    host: { http: {} },
  });
  return plugin;
}

describe("embedding backend selection (ADR 0008)", () => {
  it("builds the local transformers model by default", () => {
    const plugin = pluginWith({});
    const model = plugin.buildEmbeddingModel();
    expect(model.modelId).toBe("multilingual-e5-small-q8");
    expect(model.prefixPolicy).toBe("e5");
    expect(model.dimension).toBe(384);
  });

  it("builds the remote model in remote mode with an endpoint-agnostic identity", () => {
    const plugin = pluginWith({
      mode: "remote",
      baseUrl: "https://api.openai.com/v1",
      apiKey: "sk-embed",
      model: "text-embedding-3-small",
      dimension: 1536,
    });
    const model = plugin.buildEmbeddingModel();
    expect(model.modelId).toBe("remote:text-embedding-3-small:1536");
    expect(model.prefixPolicy).toBe("none");
    expect(model.dimension).toBe(1536);
  });

  it("gates remote embedding on a complete configuration", () => {
    const plugin = pluginWith({ mode: "remote", baseUrl: "", apiKey: "", model: "" });
    expect(() => plugin.assertRemoteEmbeddingReady()).toThrow(
      "Remote embedding configuration incomplete",
    );
  });

  it("gates remote embedding on full-text processing authorization", () => {
    const plugin = pluginWith({
      mode: "remote",
      baseUrl: "https://api.openai.com/v1",
      apiKey: "sk-embed",
      model: "text-embedding-3-small",
      dimension: 1536,
    });
    plugin.libraryConnection = undefined;
    expect(() => plugin.assertRemoteEmbeddingReady()).toThrow(
      "authorizing full-text processing",
    );
  });

  it("never gates the local mode", () => {
    const plugin = pluginWith({ mode: "local" });
    expect(() => plugin.assertRemoteEmbeddingReady()).not.toThrow();
  });
});
