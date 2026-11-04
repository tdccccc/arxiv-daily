import { describe, expect, it } from "vitest";
import { getBusinessSetting, TIMEZONE_OPTIONS } from "@arxiv-daily/core";
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
      thinkingMode: true,
      reasoningEffort: "high",
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
    expect(body).toContain("\nmax_daily_papers = 20\n");
  });

  it("runs non-TUI wizard: provider → url → key → models → rest", async () => {
    const answers = [
      "/tmp/vault-test",
      "1", // provider
      "https://api.example.com/v1",
      "sk-test-key",
      "y", // fetch models
      "1", // first remote model
      "y", // thinking on
      "3", // high effort (1 low 2 medium 3 high)
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
    const display: string[] = [];
    const code = await runInit({
      isTTY: true,
      configPath: "/tmp/arxiv-daily-init-test.toml",
      ask: async () => {
        const v = answers[i] ?? "";
        i += 1;
        return v;
      },
      fetchModels: async () => ["model-a", "model-b"],
      testChat: async () => ({ success: true }),
      writeFile: async (filePath, body) => {
        written.push({ path: filePath, body });
      },
      readFile: async () => {
        const err = new Error("missing") as NodeJS.ErrnoException;
        err.code = "ENOENT";
        throw err;
      },
      mkdir: async () => undefined,
      stdout: { write: value => { display.push(value); } },
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
    expect(written[0]!.body).toContain("\nmax_daily_papers = 20\n");
    expect(written[0]!.body).toContain("thinking_mode = true");
    expect(written[0]!.body).toContain('reasoning_effort = "high"');
    for (const id of ["reasoningEffort", "emailMode", "summaryLanguage"] as const) {
      for (const [value,label] of Object.entries(getBusinessSetting(id).options!)) {
        if (id !== "reasoningEffort" || value !== "none") expect(display.join(" ")).toContain(label);
      }
    }
    for (const option of TIMEZONE_OPTIONS) expect(display.join(" ")).toContain(option.label);

  });
  it("re-prompts a topic tag or description the daily report cannot carry", async () => {
    const tags = ["t".repeat(121), "photo-z"];
    const descriptions = ["x".repeat(1001), "photo-z methods"];
    let body = ""; const errors: string[] = [];
    const code = await runInit({
      configPath:"/tmp/init-topic-limits.toml", isTTY:true,
      ask: async prompt => {
        if (prompt.startsWith("API base URL")) return "https://model.test/v1";
        if (prompt.startsWith("LLM API key")) return "test-key";
        if (prompt.startsWith("Topic tag slug")) return tags.shift() ?? "photo-z";
        if (prompt.startsWith("Topic description")) return descriptions.shift() ?? "photo-z methods";
        if (/fetch|connect|thinking|schedule/i.test(prompt)) return "n";
        return "";
      },
      fetchModels: async () => [],
      readFile: async () => {throw Object.assign(new Error("missing"),{code:"ENOENT"});},
      mkdir:async()=>{}, writeFile:async(_path,text)=>{body=text;}, stdout:{write:()=>{}},stderr:{write:text=>errors.push(text)},
    });
    expect(code).toBe(0); expect(tags).toEqual([]); expect(descriptions).toEqual([]);
    expect(errors.join(" ")).toContain("A tag can have at most 120 characters");
    expect(errors.join(" ")).toContain("A topic description can have at most 1000 characters");
    expect(body).toContain('tag = "photo-z"');
    expect(body).toContain("photo-z methods");
  });
  it("requires a real research direction instead of the former placeholder", async () => {
    const descriptions = ["", "Describe in natural language what papers belong in this topic", "photo-z methods"];
    let body = ""; const errors: string[] = []; const out: string[] = [];
    const code = await runInit({
      configPath:"/tmp/init-direction-required.toml", isTTY:true,
      ask: async prompt => {
        if (prompt.startsWith("API base URL")) return "https://model.test/v1";
        if (prompt.startsWith("LLM API key")) return "test-key";
        if (prompt.startsWith("Topic description")) return descriptions.shift() ?? "photo-z methods";
        if (/fetch|connect|thinking|schedule/i.test(prompt)) return "n";
        return "";
      },
      fetchModels: async () => [],
      readFile: async () => {throw Object.assign(new Error("missing"),{code:"ENOENT"});},
      mkdir:async()=>{}, writeFile:async(_path,text)=>{body=text;},
      stdout:{write:text=>out.push(String(text))}, stderr:{write:text=>errors.push(String(text))},
    });
    expect(code).toBe(0); expect(descriptions).toEqual([]);
    expect(errors.join(" ")).toContain("Required — one sentence");
    expect(errors.join(" ")).toContain("Replace the example with your own research direction");
    expect(body).toContain('name = "My research"');
    expect(body).toContain('description = "photo-z methods"');
    expect(body).not.toContain("Describe in natural language");
    expect(out.join("")).not.toContain("Optional: refine");
  });
  it("re-prompts custom endpoints and timezones using shared settings validation", async () => {
    const urls = ["file:///tmp/model", "https://user:secret@model.test/v1", "https://model.test/v1?api_key=secret", "https://model.test/v1"];
    const zones = ["invalid/timezone", "Pacific/Auckland"];
    let body = ""; const errors: string[] = [];
    const code = await runInit({
      configPath:"/tmp/init-shared-validation.toml", isTTY:true,
      ask: async prompt => {
        if (prompt.startsWith("API base URL")) return urls.shift() ?? "https://model.test/v1";
        if (prompt.startsWith("LLM API key")) return "test-key";
        if (prompt.startsWith("Timezone for")) return "__other__";
        if (prompt.startsWith("IANA timezone")) return zones.shift() ?? "Pacific/Auckland";
        if (prompt.startsWith("Topic description")) return "photo-z methods";
        if (/fetch|connect|thinking|schedule/i.test(prompt)) return "n";
        return "";
      },
      fetchModels: async () => [],
      readFile: async () => {throw Object.assign(new Error("missing"),{code:"ENOENT"});},
      mkdir:async()=>{}, writeFile:async(_path,text)=>{body=text;}, stdout:{write:()=>{}},stderr:{write:text=>errors.push(text)},
    });
    expect(code).toBe(0); expect(urls).toEqual([]); expect(zones).toEqual([]);
    expect(body).toContain('base_url = "https://model.test/v1"');
    expect(body).toContain('timezone = "Pacific/Auckland"');
    expect(errors.join(" ")).toContain("llm.baseUrl");
    expect(errors.join(" ")).toContain("arxiv.timezone");
  });

  it("defaults thinking mode off for a custom endpoint, following the preset", async () => {
    let body = ""; let thinkingPrompt = "";
    const code = await runInit({
      configPath: "/tmp/init-custom-thinking.toml", isTTY: true,
      ask: async prompt => {
        if (prompt.startsWith("Vault root")) return "/tmp/vault-test";
        if (prompt.startsWith("LLM provider")) return "5"; // custom
        if (prompt.startsWith("API base URL")) return "https://gateway.example/v1";
        if (prompt.startsWith("LLM API key")) return "test-key";
        if (prompt.startsWith("Model id")) return "my-model";
        if (prompt.startsWith("Enable thinking mode?")) { thinkingPrompt = prompt; return ""; }
        if (prompt.startsWith("Topic name")) return "My research";
        if (prompt.startsWith("Topic tag")) return "my-research";
        if (prompt.startsWith("Topic description")) return "photo-z methods";
        if (/fetch|connect|test chat|schedule/i.test(prompt)) return "n";
        return "";
      },
      readFile: async () => { throw Object.assign(new Error("missing"), { code: "ENOENT" }); },
      mkdir: async () => {}, writeFile: async (_path, text) => { body = text; },
      stdout: { write: () => {} }, stderr: { write: () => {} },
    });
    expect(code).toBe(0);
    expect(thinkingPrompt).toContain("(default: No)");
    expect(body).toContain("thinking_mode = false");
    expect(body).toContain('base_url = "https://gateway.example/v1"');
  });

  it("re-runs the chat test after going back and turning thinking off when the provider rejects the parameters", async () => {
    let body = ""; const errors: string[] = []; const out: string[] = []; const chats: boolean[] = [];
    let thinkingPrompts = 0;
    const code = await runInit({
      configPath: "/tmp/init-chat-test.toml", isTTY: true,
      ask: async prompt => {
        if (prompt.startsWith("Vault root")) return "/tmp/vault-test";
        if (prompt.startsWith("API base URL")) return "https://model.test/v1";
        if (prompt.startsWith("LLM API key")) return "test-key";
        if (prompt.startsWith("Test connection and load model list?")) return "y";
        if (prompt.startsWith("Model id")) return "gpt-x";
        if (prompt.startsWith("Enable thinking mode?")) { thinkingPrompts += 1; return thinkingPrompts === 1 ? "y" : "n"; }
        if (prompt.startsWith("Reasoning effort")) return "3";
        if (prompt.startsWith("Keep these settings anyway?")) return "n";
        if (prompt.startsWith("Topic name")) return "My research";
        if (prompt.startsWith("Topic tag")) return "my-research";
        if (prompt.startsWith("Topic description")) return "photo-z methods";
        return "";
      },
      fetchModels: async () => [],
      testChat: async (settings) => {
        chats.push(settings.thinkingMode);
        return settings.thinkingMode
          ? { success: false, error: "Unsupported parameter: reasoning_effort" }
          : { success: true };
      },
      readFile: async () => { throw Object.assign(new Error("missing"), { code: "ENOENT" }); },
      mkdir: async () => {}, writeFile: async (_path, text) => { body = text; },
      stdout: { write: t => out.push(String(t)) }, stderr: { write: t => errors.push(String(t)) },
    });
    expect(code).toBe(0);
    expect(chats).toEqual([true, false]);
    expect(errors.join(" ")).toContain("Chat test failed: Unsupported parameter: reasoning_effort");
    expect(errors.join(" ")).toContain("turn thinking mode off");
    expect(out.join(" ")).toContain("Chat OK");
    expect(body).toContain("thinking_mode = false");
  });

});
