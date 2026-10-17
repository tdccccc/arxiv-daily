import { describe, expect, it } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { runInit } from "../src/init";
import { loadCliConfig } from "../src/config";

describe("init e2e", () => {
  it.skipIf(process.platform === "win32").each(["new", "overwrite", "keep"])(
    "keeps credentials private when initializing a %s config",
    async (mode) => {
      const dir = await fs.mkdtemp(path.join(os.tmpdir(), "ad-init-private-"));
      const configDir = path.join(dir, "config");
      const configPath = path.join(configDir, "config.toml");
      const existing = 'vault_root = "/previous-vault"\n[llm]\napi_key = "old-test-key"\n';
      try {
        if (mode !== "new") {
          await fs.mkdir(configDir);
          await fs.writeFile(configPath, existing);
          await fs.chmod(configPath, 0o644);
        }
        const answers = [
          ...(mode === "new" ? [] : [mode === "keep" ? "k" : "o"]),
          path.join(dir, "vault"), "1", "https://api.example.com/v1",
          "sk-private-test-key", "n", "1", "y", "3", "1", "1",
          "UTC", "2", "Photo-z", "photo-z", "photo-z methods", "n", "n",
        ];
        let i = 0;
        expect(await runInit({
          configPath,
          isTTY: true,
          ask: async () => {
            if (i >= answers.length) throw new Error("unexpected wizard prompt");
            return answers[i++]!;
          },
          stdout: { write: () => undefined },
          stderr: { write: () => undefined },
        })).toBe(0);
        expect((await fs.stat(configPath)).mode & 0o777).toBe(0o600);
        if (mode === "new") {
          expect((await fs.stat(configDir)).mode & 0o777).toBe(0o700);
        }
        const config = await loadCliConfig({ configPath });
        expect(config.settings.llm.apiKey).toBe(
          mode === "keep" ? "old-test-key" : "sk-private-test-key",
        );
        if (mode === "keep") {
          expect(config.vaultRoot).toBe("/previous-vault");
          expect(await fs.readFile(configPath, "utf8")).toBe(existing);
          // Keeping the file ends init before any wizard question.
          expect(i).toBe(1);
        }
        const entries = await fs.readdir(configDir);
        if (mode === "overwrite") {
          const backups = entries.filter(name => name.startsWith("config.toml.bak-"));
          expect(backups).toHaveLength(1);
          const backup = path.join(configDir, backups[0]!);
          expect(await fs.readFile(backup, "utf8")).toBe(existing);
          expect((await fs.stat(backup)).mode & 0o777).toBe(0o600);
          expect(entries.filter(name => !backups.includes(name)).sort()).toEqual([".arxiv-daily-config-locks", "config.toml"].sort().filter(name => entries.includes(name)));
        } else {
          expect(entries.filter(name => name !== ".arxiv-daily-config-locks")).toEqual(["config.toml"]);
        }
      } finally {
        await fs.rm(dir, { recursive: true, force: true });
      }
    },
  );

  it("writes config that loadCliConfig accepts", async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "ad-init-"));
    const cfgPath = path.join(dir, "config.toml");
    const vault = path.join(dir, "vault");
    const answers = [
      vault,
      "1", // provider
      "https://api.example.com/v1",
      "sk-test-key",
      "n", // skip fetch
      "1", // model preset
      "y", // thinking on
      "3", // high
      "1", // email skip
      "1", // category
      "UTC",
      "2", // en
      "Photo-z",
      "photo-z",
      "photo-z methods",
      "n", // paper notes off
      "n", // schedule off
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
    expect(cfg.settings.llm.thinkingMode).toBe(true);
    expect(cfg.settings.llm.reasoningEffort).toBe("high");
  });

  it("keeps the existing config on the default answer and asks no wizard question", async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "ad-init-default-"));
    const configPath = path.join(dir, "config.toml");
    const existing = 'vault_root = "/previous-vault"\n[library]\nselected_root = "/papers"\n';
    try {
      await fs.writeFile(configPath, existing);
      const prompts: string[] = [];
      const out: string[] = [];
      expect(await runInit({
        configPath, isTTY: true,
        ask: async prompt => { prompts.push(prompt); return ""; },
        stdout: { write: text => { out.push(text); } }, stderr: { write: () => undefined },
      })).toBe(0);
      expect(prompts).toHaveLength(1);
      expect(prompts[0]).toContain("Config already exists");
      expect(await fs.readFile(configPath, "utf8")).toBe(existing);
      expect(await fs.readdir(dir)).toEqual(["config.toml"]);
      expect(out.join("")).toContain("nothing changed");
    } finally {
      await fs.rm(dir, { recursive: true, force: true });
    }
  });

  it("leaves the file alone when cancelled and does not append anything on repeated keeps", async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "ad-init-cancel-"));
    const configPath = path.join(dir, "config.toml");
    const existing = 'vault_root = "/previous-vault"\n';
    try {
      await fs.writeFile(configPath, existing);
      for (const answer of ["c", "k", "k"]) {
        expect(await runInit({
          configPath, isTTY: true, ask: async () => answer,
          stdout: { write: () => undefined }, stderr: { write: () => undefined },
        })).toBe(0);
      }
      expect(await fs.readFile(configPath, "utf8")).toBe(existing);
    } finally {
      await fs.rm(dir, { recursive: true, force: true });
    }
  });

  it("does not back up or touch the file when the overwrite wizard is abandoned", async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "ad-init-abandon-"));
    const configPath = path.join(dir, "config.toml");
    const existing = 'vault_root = "/previous-vault"\n';
    try {
      await fs.writeFile(configPath, existing);
      const answers = ["o", path.join(dir, "vault"), "back"];
      let i = 0;
      // "back" on the first wizard step is not available, so the wizard keeps asking; throw to abandon it.
      await expect(runInit({
        configPath, isTTY: true,
        ask: async () => { if (i >= answers.length) throw new Error("abandoned"); return answers[i++]!; },
        stdout: { write: () => undefined }, stderr: { write: () => undefined },
      })).rejects.toThrow("abandoned");
      expect(await fs.readFile(configPath, "utf8")).toBe(existing);
      expect(await fs.readdir(dir)).toEqual(["config.toml"]);
    } finally {
      await fs.rm(dir, { recursive: true, force: true });
    }
  });
});
