import { describe, expect, it } from "vitest";
import { renderInitToml, runInit } from "../src/init";

describe("CLI init template", () => {
  it("writes English comments and selected fields", () => {
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
      },
      email: {
        enabled: false,
        mode: "self",
        to: "a@b.com",
        apiKey: "re_x",
        hostedToken: "",
      },
    });
    expect(body).toContain("arXiv Daily — CLI config");
    expect(body).toContain("May contain secrets");
    expect(body).not.toContain("可含密钥");
    expect(body).toContain('categories = ["cs.LG", "cs.AI"]');
    expect(body).toContain('provider = "openai"');
    expect(body).toContain('name = "ML"');
    expect(body).toContain("machine learning papers");
  });

  it("runs wizard with injected answers", async () => {
    const answers = [
      "/tmp/vault-test",
      "1", // deepseek often first - use number; provider list order
      "sk-test-key",
      "1", // model
      "", // base url default
      "1", // skip email
      "1", // physics group
      "1", // first category
      "11", // UTC often 11th - or type UTC
      "2", // en
      "Photo-z",
      "photo-z",
      "photo-z methods",
    ];
    // Fix timezone to explicit UTC string to avoid index drift
    answers[8] = "UTC";
    let i = 0;
    const written: { path: string; body: string }[] = [];
    const stdout: string[] = [];
    const code = await runInit({
      isTTY: true,
      configPath: "/tmp/arxiv-daily-init-test.toml",
      ask: async () => {
        const v = answers[i] ?? "";
        i += 1;
        return v;
      },
      writeFile: async (p, body) => {
        written.push({ path: p, body });
      },
      readFile: async () => {
        const err = new Error("missing") as NodeJS.ErrnoException;
        err.code = "ENOENT";
        throw err;
      },
      mkdir: async () => undefined,
      stdout: { write: (c) => stdout.push(String(c)) },
      stderr: { write: () => undefined },
    });
    expect(code).toBe(0);
    expect(written).toHaveLength(1);
    expect(written[0]!.body).toContain("sk-test-key");
    expect(written[0]!.body).toContain("photo-z");
    expect(written[0]!.body).toContain("CLI config");
    expect(stdout.join("")).toContain("Next steps");
  });
});
