import { describe, expect, it } from "vitest";
import { renderInitToml, runInit } from "../src/init";

describe("CLI init template", () => {
  it("writes English comments with schema_version at the end", () => {
    const body = renderInitToml({
      vaultRoot: "/vault",
      cacheDir: "/vault/.cache/arxiv-daily",
      apiKey: "sk-x",
      baseUrl: "https://api.example.com/v1",
      model: "m1",
      provider: "openai",
      thinkingMode: false,
      categories: ["cs.LG", "cs.AI"],
      timezone: "UTC",
      summaryLanguage: "en",
      topic: {
        name: "ML",
        tag: "ml",
        description: "machine learning papers",
        detail: true,
      },
      email: {
        enabled: false,
        mode: "self",
        to: "a@b.com",
        apiKey: "re_x",
        hostedToken: "",
      },
      linkStyle: "wikilink",
      logLevel: "info",
      schedule: {
        enabled: false,
        on: "09:30",
        intervalHours: 0,
        until: "18:00",
        weekdaysOnly: true,
      },
    });
    expect(body).toContain("arXiv Daily — CLI config");
    expect(body).toContain("Do not change unless a release note tells you to");
    expect(body.trimEnd().endsWith("schema_version = 1")).toBe(true);
    expect(body).not.toContain("可含密钥");
    expect(body).toContain('categories = ["cs.LG", "cs.AI"]');
    expect(body).toContain('provider = "openai"');
  });

  it("runs non-TUI wizard: provider → url → key → models → rest", async () => {
    const answers = [
      "/tmp/vault-test",
      "1", // provider
      "https://api.example.com/v1",
      "sk-test-key",
      "y", // fetch models
      "1", // first remote model
      "1", // email skip
      "1", // first category
      "UTC",
      "2", // en
      "Photo-z",
      "photo-z",
      "photo-z methods",
      "n", // auto paper notes off
      "n", // schedule off
    ];
    let i = 0;
    const written: { path: string; body: string }[] = [];
    const code = await runInit({
      isTTY: true,
      configPath: "/tmp/arxiv-daily-init-test.toml",
      ask: async () => {
        const v = answers[i] ?? "";
        i += 1;
        return v;
      },
      fetchModels: async () => ["model-a", "model-b"],
      writeFile: async (filePath, body) => {
        written.push({ path: filePath, body });
      },
      readFile: async () => {
        const err = new Error("missing") as NodeJS.ErrnoException;
        err.code = "ENOENT";
        throw err;
      },
      mkdir: async () => undefined,
      stdout: { write: () => undefined },
      stderr: { write: () => undefined },
    });
    expect(code).toBe(0);
    expect(written).toHaveLength(1);
    expect(written[0]!.body).toContain("sk-test-key");
    expect(written[0]!.body).toContain("model-a");
    expect(written[0]!.body).toContain("photo-z");
    expect(written[0]!.body).toContain("https://api.example.com/v1");
    expect(written[0]!.body.trimEnd().endsWith("schema_version = 1")).toBe(
      true,
    );
    expect(written[0]!.body).toContain("detail = false");
  });
});
