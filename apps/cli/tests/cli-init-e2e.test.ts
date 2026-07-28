import { describe, expect, it } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { runInit } from "../src/init";
import { loadCliConfig } from "../src/config";

describe("init e2e", () => {
  it("writes config that loadCliConfig accepts", async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "ad-init-"));
    const cfgPath = path.join(dir, "config.toml");
    const vault = path.join(dir, "vault");
    const answers = [
      vault,
      "1",
      "https://api.example.com/v1",
      "sk-test-key",
      "n",
      "1",
      "1",
      "1",
      "UTC",
      "2",
      "Photo-z",
      "photo-z",
      "photo-z methods",
      "n",
      "n",
    ];
    let i = 0;
    const code = await runInit({
      isTTY: true,
      configPath: cfgPath,
      ask: async () => answers[i++] ?? "",
      writeFile: (p, b) => fs.writeFile(p, b),
      readFile: async () => {
        const e = new Error("missing") as NodeJS.ErrnoException;
        e.code = "ENOENT";
        throw e;
      },
      mkdir: async (p) => {
        await fs.mkdir(p, { recursive: true });
      },
      stdout: { write: () => undefined },
      stderr: { write: () => undefined },
    });
    expect(code).toBe(0);
    const body = await fs.readFile(cfgPath, "utf8");
    expect(body).toContain("sk-test-key");
    expect(body).toContain("detail = false");
    expect(body.trimEnd().endsWith("schema_version = 1")).toBe(true);
    expect(body).toContain('link_style = "wikilink"');
    expect(body).toContain('log_level = "info"');

    const cfg = await loadCliConfig({ configPath: cfgPath });
    expect(cfg.vaultRoot).toBe(path.resolve(vault));
    expect(cfg.settings.llm.apiKey).toBe("sk-test-key");
    expect(cfg.settings.llm.baseUrl).toBe("https://api.example.com/v1");
    expect(cfg.settings.arxiv.topics[0]?.detail).toBe(false);
    expect(cfg.settings.arxiv.topics[0]?.tag).toBe("photo-z");
    expect(cfg.settings.output.summaryLanguage).toBe("en");
    expect(cfg.settings.output.linkStyle).toBe("wikilink");
    expect(cfg.settings.advanced.logLevel).toBe("info");
    expect(cfg.scheduleIntent.enabled).toBe(false);
  });
});
