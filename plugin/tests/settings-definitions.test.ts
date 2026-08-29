import { describe, expect, it } from "vitest";
import {
  allSettingKeys,
  buildSettingDefinitions,
  readSettingValue,
  SETTING_KEYS,
  writeSettingValue,
} from "../src/settings/definitions";
import { DEFAULT_SETTINGS } from "@arxiv-daily/core";

describe("setting key path mapping", () => {
  it("registers flat keys for every settings section", () => {
    expect(SETTING_KEYS.llm.baseUrl).toBe("llm.baseUrl");
    expect(SETTING_KEYS.email.hostedToken).toBe("email.hostedToken");
    expect(allSettingKeys().length).toBeGreaterThanOrEqual(24);
    expect(new Set(allSettingKeys()).size).toBe(allSettingKeys().length);
  });

  it("reads nested values through dotted keys", () => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    settings.llm.baseUrl = "https://example.com/v1";
    settings.email.to = "me@example.com";
    settings.schedule.tickIntervalMin = 7;

    expect(readSettingValue(settings, "llm.baseUrl")).toBe("https://example.com/v1");
    expect(readSettingValue(settings, "email.to")).toBe("me@example.com");
    expect(readSettingValue(settings, "schedule.tickIntervalMin")).toBe(7);
    expect(readSettingValue(settings, "llm.apiKey")).toBe(DEFAULT_SETTINGS.llm.apiKey);
  });

  it("writes nested values through dotted keys", () => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    writeSettingValue(settings, "llm.model", "deepseek-chat");
    writeSettingValue(settings, "output.linkStyle", "relative");
    writeSettingValue(settings, "email.fromName", "arXiv Daily");

    expect(settings.llm.model).toBe("deepseek-chat");
    expect(settings.output.linkStyle).toBe("relative");
    expect(settings.email.fromName).toBe("arXiv Daily");
  });

  it("round-trips values through read+write", () => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    for (const key of allSettingKeys()) {
      const value = readSettingValue(settings, key);
      writeSettingValue(settings, key, value);
      expect(readSettingValue(settings, key)).toEqual(value);
    }
  });

  it("returns undefined for unknown or missing paths", () => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    expect(readSettingValue(settings, "nope.missing")).toBeUndefined();
    expect(readSettingValue(settings, "llm")).toBe(settings.llm);
  });

  it("ignores writes through a missing intermediate path", () => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    writeSettingValue(settings, "missing.deep.value", 1);
    expect(settings).toEqual(structuredClone(DEFAULT_SETTINGS));
  });
});

describe("buildSettingDefinitions structure", () => {
  function makeHost() {
    return {
      plugin: {
        settings: structuredClone(DEFAULT_SETTINGS),
        manifest: { version: "0.0.0-test" },
        app: {},
      },
    };
  }

  /** Host with every render/action callback supplied, as the tab will wire it. */
  function makeFullHost() {
    return {
      ...makeHost(),
      showSetupGuide: true,
      renderSetupGuideRow: () => {},
      renderLlmBaseUrlRow: () => {},
      renderApiKeyRow: () => {},
      renderModelRow: () => {},
      renderReasoningEffortRow: () => {},
      renderLibraryConnectionRow: () => {},
      renderCategoryRow: () => {},
      renderTopicRow: () => {},
      renderTimezoneRow: () => {},
      addCategory: () => {},
      deleteCategory: () => {},
      addTopic: () => {},
      renderScheduleEnabledRow: () => {},
      renderRunWindowRow: () => {},
      renderTickIntervalRow: () => {},
      renderEmailGuideRow: () => {},
      renderEmailModeRow: () => {},
      renderEmailToRow: () => {},
      renderEmailApiKeyRow: () => {},
      renderHostedTokenRow: () => {},
      renderEmbeddingModeRow: () => {},
      renderEmbeddingBaseUrlRow: () => {},
      renderEmbeddingApiKeyRow: () => {},
      renderEmbeddingModelRow: () => {},
      renderEmbeddingDimensionRow: () => {},
    };
  }

  it("returns top-level items with an Enable toggle and section groups", () => {
    const items = buildSettingDefinitions(makeHost());
    expect(items.length).toBeGreaterThanOrEqual(4);
    const groups = items.filter((item) => item.type === "group");
    expect(groups.map((g) => g.heading)).toEqual(
      expect.arrayContaining(["LLM", "Embedding", "Output & schedule", "Advanced", "Help & feedback"]),
    );
  });

  it("orders the compact LLM rows and removes obsolete controls", () => {
    const items = buildSettingDefinitions(makeFullHost());
    const llm = items.find(
      (item): item is Extract<(typeof items)[number], { type: "group" }> =>
        item.type === "group" && item.heading === "LLM",
    );
    expect(llm?.items.map((item) => item.name)).toEqual([
      "API base URL",
      "API key",
      "Model",
      "Reasoning effort",
    ]);
    expect(llm?.items.map((item) => item.name)).not.toContain("Thinking mode");
    expect(items.some((item) => item.name === "Quick start")).toBe(false);
  });

  it("orders the embedding rows for remote configuration", () => {
    const items = buildSettingDefinitions(makeFullHost());
    const embedding = items.find(
      (item): item is Extract<(typeof items)[number], { type: "group" }> =>
        item.type === "group" && item.heading === "Embedding",
    );
    expect(embedding?.items.map((item) => item.name)).toEqual([
      "Embedding mode",
      "Embedding API base URL",
      "Embedding API key",
      "Embedding model",
      "Embedding dimension",
    ]);
    // Without the render callbacks the group carries no rows.
    const bare = buildSettingDefinitions(makeHost()).find(
      (item) => item.type === "group" && item.heading === "Embedding",
    );
    expect(bare?.items ?? []).toHaveLength(0);
  });

  it("only includes Getting started while setup is incomplete", () => {
    const host = makeFullHost();
    expect(buildSettingDefinitions(host).some((item) => item.name === "Getting started"))
      .toBe(true);
    host.showSetupGuide = false;
    expect(buildSettingDefinitions(host).some((item) => item.name === "Getting started"))
      .toBe(false);
  });

  it("resolves every declarative control key through readSettingValue", () => {
    const keys = new Set(allSettingKeys());
    const walk = (items: readonly unknown[]): void => {
      for (const item of items as ReadonlyArray<Record<string, unknown>>) {
        if (Array.isArray(item.items)) walk(item.items as unknown[]);
        const control = item.control as { key?: string } | undefined;
        if (control?.key) {
          expect(keys.has(control.key)).toBe(true);
          expect(readSettingValue(makeHost().plugin.settings, control.key)).toBeDefined();
        }
      }
    };
    walk(buildSettingDefinitions(makeHost()));
  });

  it("renders categories and topics without drag-to-reorder affordances", () => {
    const host = makeHost();
    const lists = buildSettingDefinitions(host).filter(
      (item) => item.type === "list",
    );
    const categoriesList = lists.find((list) => list.heading === "arXiv categories");
    const topicsList = lists.find((list) => list.heading === "Research topics");
    expect(categoriesList).toBeDefined();
    expect(topicsList).toBeDefined();
    expect(categoriesList?.addItem?.name).toBe("Add category");
    expect(categoriesList?.onDelete).toEqual(expect.any(Function));
    expect(categoriesList?.onReorder).toBeUndefined();
    expect(topicsList?.addItem?.name).toBe("Add topic");
    expect(topicsList?.onReorder).toBeUndefined();
  });

  it("maps one list item per category and topic with searchable names", () => {
    const host = makeHost();
    host.plugin.settings.arxiv.categories = ["cs.AI", "cs.LG"];
    host.plugin.settings.arxiv.topics = [
      {
        id: "t1",
        name: "Photometric redshift",
        tag: "photometric-redshift",
        description: "",
        detail: false,
      },
      {
        id: "t2",
        name: "",
        tag: "",
        description: "",
        detail: false,
      },
    ];
    const items = buildSettingDefinitions(host);
    const categoryNames = items
      .filter((item) => item.type === "list")
      .find((list) => list.heading === "arXiv categories")?.items
      .map((item) => item.name);
    expect(categoryNames).toEqual(["1", "2"]);
    const topicNames = items
      .filter((item) => item.type === "list")
      .find((list) => list.heading === "Research topics")?.items
      .map((item) => item.name);
    expect(topicNames).toContain("Photometric redshift");
    expect(topicNames).toContain("(unnamed)");
  });

  it("keeps the detail-notes profile dropdown on the balanced preset", () => {
    const host = makeHost();
    const items = buildSettingDefinitions(host);
    const detailNotes = items.find((item) => item.name === "Automatic detail notes");
    expect(detailNotes).toBeDefined();
    if (detailNotes && "control" in detailNotes && detailNotes.control) {
      expect(detailNotes.control).toMatchObject({
        type: "dropdown",
        key: SETTING_KEYS.detailSelection.profile,
        defaultValue: "balanced",
      });
    }
  });

  it("renders the scheduler enable row with a Running/Paused name, no control", () => {
    const host = makeFullHost();
    const items = buildSettingDefinitions(host);
    const enableRow = items.find(
      (item) => "name" in item && item.name.startsWith("Enable ·"),
    );
    expect(enableRow).toBeDefined();
    expect(enableRow).toHaveProperty("render");
    expect(enableRow).not.toHaveProperty("control");
    expect(
      items.some((item) => "name" in item && item.name === "Enable · Paused"),
    ).toBe(true);
    host.plugin.settings.schedule.enabled = true;
    expect(
      buildSettingDefinitions(host).some(
        (item) => "name" in item && item.name === "Enable · Running",
      ),
    ).toBe(true);
  });

  it("adds run window and interval rows to the Output & schedule group", () => {
    const host = makeFullHost();
    const items = buildSettingDefinitions(host);
    const scheduleGroup = items.find(
      (item): item is Extract<(typeof items)[number], { type: "group" }> =>
        item.type === "group" && item.heading === "Output & schedule",
    );
    const names = scheduleGroup?.items.map((item) => item.name) ?? [];
    expect(names).toContain("Run window");
    expect(names).toContain("Check every (minutes)");
  });

  it("swaps email rows by mode: api key + from (self) vs verify + code (hosted)", () => {
    const host = makeFullHost();
    const items = buildSettingDefinitions(host);
    const emailGroup = items.find(
      (item): item is Extract<(typeof items)[number], { type: "group" }> =>
        item.type === "group" && item.heading === "Email delivery",
    );
    expect(emailGroup).toBeDefined();
    const names = emailGroup?.items.map((item) => item.name) ?? [];
    expect(names).toContain("Your email");
    expect(names).toContain("Resend API key");
    expect(names).toContain("From email");
    expect(names).toContain("From name");
    expect(names).not.toContain("Send test email");
    expect(names).not.toContain("Send verification email");
    expect(names).toContain("Daily auto-send");
    expect(names).not.toContain("Verification code");

    host.plugin.settings.email.mode = "hosted";
    const hostedNames =
      buildSettingDefinitions(host)
        .find(
          (item): item is Extract<(typeof items)[number], { type: "group" }> =>
            item.type === "group" && item.heading === "Email delivery",
        )
        ?.items.map((item) => item.name) ?? [];
    expect(hostedNames).not.toContain("Send verification email");
    expect(hostedNames).not.toContain("Send test email");
    expect(hostedNames).toContain("Verification code");
    expect(hostedNames).not.toContain("Resend API key");
    expect(hostedNames).not.toContain("From email");
  });
});
