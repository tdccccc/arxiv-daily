import { describe, expect, it, vi } from "vitest";
import {
  allSettingKeys,
  buildSettingDefinitions,
  dailyAutoSendDesc,
  readSettingValue,
  SETTING_KEYS,
  writeSettingValue,
} from "../src/settings/definitions";
import { AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE, DEFAULT_SETTINGS, getBusinessSettingsSections, businessSettingsContext } from "@arxiv-daily/core";
import type { SettingDefinitionItem } from "obsidian";

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

  it("does not pollute Object.prototype through a settings path", () => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    const sentinel = "__arxivSettingsPrototypeSentinel__";
    const before = Object.getOwnPropertyDescriptor(Object.prototype, sentinel);
    try {
      writeSettingValue(settings, `__proto__.${sentinel}`, "polluted");
      expect(Object.getOwnPropertyDescriptor(Object.prototype, sentinel)).toEqual(before);
      expect(settings).toEqual(DEFAULT_SETTINGS);
    } finally {
      if (before) Object.defineProperty(Object.prototype, sentinel, before);
      else Reflect.deleteProperty(Object.prototype, sentinel);
    }
  });

  it.each(["__proto__", "constructor", "prototype"])("rejects %s at every path position without changing settings", (segment) => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    const originalPrototype = Object.getPrototypeOf(settings.llm);
    const originalSettingsPrototype = Object.getPrototypeOf(settings);
    writeSettingValue(settings, segment, { sentinel: true });
    writeSettingValue(settings, `llm.${segment}`, { sentinel: true });
    expect(Object.getPrototypeOf(settings)).toBe(originalSettingsPrototype);
    expect(Object.getPrototypeOf(settings.llm)).toBe(originalPrototype);
    expect(settings).toEqual(DEFAULT_SETTINGS);

    // Own object-valued reserved keys must not become a traversal bypass.
    Object.defineProperty(settings.llm, segment, {
      value: { model: "original" }, enumerable: true, configurable: true, writable: true,
    });
    const before = structuredClone(settings);
    writeSettingValue(settings, `llm.${segment}.model`, "polluted");
    expect(settings).toEqual(before);
  });

  it("rejects a dangerous later segment before reading any intermediate property", () => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    let reads = 0;
    Object.defineProperty(settings, "draft", {
      get: () => { reads += 1; return { value: "original" }; },
    });
    writeSettingValue(settings, "draft.prototype.value", "polluted");
    expect(reads).toBe(0);
  });

  it("does not modify an inherited intermediate object", () => {
    const shared = { model: "original" };
    const settings = structuredClone(DEFAULT_SETTINGS);
    Object.setPrototypeOf(settings, { inherited: shared });
    writeSettingValue(settings, "inherited.model", "polluted");
    expect(shared.model).toBe("original");
    expect(Object.hasOwn(settings, "inherited")).toBe(false);
  });

  it("does not invoke an inherited setter for the final segment", () => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    let value = "original";
    const inherited = Object.create(null);
    Object.defineProperty(inherited, "draft", { set: (next: string) => { value = next; } });
    Object.setPrototypeOf(settings.llm, inherited);
    writeSettingValue(settings, "llm.draft", "polluted");
    expect(value).toBe("original");
    expect(Object.hasOwn(settings.llm, "draft")).toBe(false);
  });

  it("continues allowing new ordinary leaf keys without creating missing parents", () => {
    const settings = structuredClone(DEFAULT_SETTINGS);
    writeSettingValue(settings, "llm.futureOption", "supported");
    expect(readSettingValue(settings, "llm.futureOption")).toBe("supported");
    writeSettingValue(settings, "futureSection.option", "ignored");
    expect(Object.hasOwn(settings, "futureSection")).toBe(false);
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
      renderOutputDirectoryRow: () => {},
      renderEmailSenderRow: () => {},
      addCategory: () => {},
      deleteCategory: () => {},
      addTopic: () => {},
      renderScheduleEnabledRow: () => {},
      renderRunWindowRow: () => {},
      renderTickIntervalRow: () => {},
      renderDailyPaperLimitRow: () => {},
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
      renderPdfParserSidecarEnabledRow: () => {},
      renderPdfParserSidecarCapabilitiesUrlRow: () => {},
      renderPdfParserSidecarParseUrlRow: () => {},
    };
  }

  it("returns top-level items with an Enable toggle and section groups", () => {
    const items = buildSettingDefinitions(makeHost());
    expect(items.length).toBeGreaterThanOrEqual(4);
    const groups = items.filter((item) => item.type === "group");
    expect(groups.map((g) => g.heading)).toEqual(
      expect.arrayContaining(["LLM", "Output & schedule", "Advanced", "Help & feedback"]),
    );
    expect(groups.map((g) => g.heading)).not.toContain("Embedding");
    expect(groups.map((g) => g.heading)).not.toContain("PDF parsing");
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

  it("folds embedding and PDF parser into Personal library and hides extras by default", () => {
    const host = makeFullHost();
    const items = buildSettingDefinitions(host);
    expect(items.map((item) => ("heading" in item ? item.heading : item.name))).toEqual(
      expect.arrayContaining(["LLM", "Personal library"]),
    );
    const llmIndex = items.findIndex((item) => item.type === "group" && item.heading === "LLM");
    const libraryIndex = items.findIndex(
      (item) => item.type === "group" && item.heading === "Personal library",
    );
    expect(llmIndex).toBeGreaterThanOrEqual(0);
    expect(libraryIndex).toBeGreaterThan(llmIndex);
    expect(items.some((item) => item.type === "group" && item.heading === "Embedding")).toBe(false);
    expect(items.some((item) => item.type === "group" && item.heading === "PDF parsing")).toBe(false);

    const library = items.find(
      (item): item is Extract<(typeof items)[number], { type: "group" }> =>
        item.type === "group" && item.heading === "Personal library",
    );
    expect(library?.items.map((item) => item.name)).toEqual([
      "Library",
      "Embedding",
    ]);

    host.plugin.settings.embedding.mode = "remote";
    // Enabling the sidecar must not surface any row: it takes no part in
    // indexing, so a visible control would promise a change it cannot deliver.
    host.plugin.settings.pdfParserSidecar.enabled = true;
    const expanded = buildSettingDefinitions(host).find(
      (item): item is Extract<(typeof items)[number], { type: "group" }> =>
        item.type === "group" && item.heading === "Personal library",
    );
    expect(expanded?.items.map((item) => item.name)).toEqual([
      "Library",
      "Embedding",
      "Embedding API base URL",
      "Embedding API key",
      "Embedding model",
      "Embedding dimension",
    ]);

    const bare = buildSettingDefinitions(makeHost()).find(
      (item) => item.type === "group" && item.heading === "Personal library",
    );
    expect(bare).toBeUndefined();
  });

  it("only includes Getting started while setup is incomplete", () => {
    const host = makeFullHost();
    const isGuideRow = (item: SettingDefinitionItem) =>
      item.name === "" && "render" in item;
    expect(buildSettingDefinitions(host).some(isGuideRow)).toBe(true);
    host.showSetupGuide = false;
    expect(buildSettingDefinitions(host).some(isGuideRow)).toBe(false);
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
    host.plugin.settings.arxiv.categories = ["astro-ph", "gr-qc"];
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
      { directions: [],
        id: "t1",
        name: "Photometric redshift",
        tag: "photometric-redshift",
        description: "",
        detail: false,
      },
      { directions: [],
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

  it("warns on the auto-send row when this system cannot send automatic email", () => {
    const autoSendDesc = (host: ReturnType<typeof makeFullHost>) => {
      const items = buildSettingDefinitions(host);
      const group = items.find(
        (item): item is Extract<(typeof items)[number], { type: "group" }> =>
          item.type === "group" && item.heading === "Email delivery",
      );
      return group?.items.find((item) => item.name === "Daily auto-send")?.desc;
    };
    const host = makeFullHost();
    expect(autoSendDesc(host)).not.toContain(AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE);

    const unsupported = { ...host, automaticEmailSupported: false };
    expect(autoSendDesc(unsupported)).toContain(AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE);
    expect(dailyAutoSendDesc(false, false)).toContain(AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE);
    expect(dailyAutoSendDesc(true, true)).not.toContain(AUTOMATIC_EMAIL_UNSUPPORTED_MESSAGE);
  });
  it.each([false,true])("uses shared names, descriptions, options and visibility (expanded=%s)", expanded => {
    const host=makeFullHost();
    host.plugin.settings.embedding.mode=expanded?'remote':'local';
    host.plugin.settings.pdfParserSidecar.enabled=expanded;
    host.plugin.settings.email.mode=expanded?'hosted':'self';
    host.plugin.settings.detailSelection.profile=expanded?'custom':'balanced';
    host.plugin.settings.schedule.enabled=expanded;
    const items=buildSettingDefinitions(host);
    for(const section of getBusinessSettingsSections(businessSettingsContext(host.plugin.settings))){
      if(section.type==='list'){
        const list=items.find(item=>item.type==='list'&&item.heading===section.heading);
        expect(list).toMatchObject({emptyState:section.emptyState,addItem:{name:section.addItemName}});
        continue;
      }
      // The native title/abstract index deliberately hides the retired structured-PDF controls.
      const expected=(section.type==='field'?[section.field]:section.items).filter(field=>!field.id.startsWith('sidecar'));
      const actual=section.type==='field'?items.filter(item=>item.name===section.field.name):items.find(item=>item.type==='group'&&item.heading===section.heading)?.items.filter(item=>item.name);
      expect(actual?.map(item=>({name:item.name,description:item.desc}))).toEqual(expected.map(item=>({name:item.name,description:item.description})));
      for(const item of actual??[])if(item.control){
        const metadata=expected.find(field=>field.name===item.name)!;
        expect(item.control.key).toBe(metadata.key);
        if(item.control.type==='dropdown')expect(item.control.options).toEqual(metadata.options);
      }
    }
  });

  it("adds native library guide and direction actions without exposing sidecar controls", () => {
    const base = makeFullHost();
    const host = {
      ...base,
      plugin: {
        ...base.plugin,
        getLibraryConnectionStatus: () => ({ kind: "connected" }),
        libraryIndexStatus: { snapshot: () => ({ lastRun: { papers: 6 } }) },
      },
      renderLibraryGuideRow: vi.fn(),
      renderLibraryDirectionsRow: vi.fn(),
    };
    const items = buildSettingDefinitions(host);
    const library = items.find(item => item.type === "group" && item.heading === "Personal library");
    if (!library || library.type !== "group") throw new Error("missing library section");
    expect(library.items[0]).toMatchObject({ name: "", render: expect.any(Function) });
    const directions = library.items.find(item => item.name === "Topics from library");
    expect(directions?.desc).toContain("Only directions added to Research topics steer daily reports");
    expect(library.items.some(item => item.name.includes("Sidecar") || item.name.includes("Structured PDF"))).toBe(false);
    const setting = {} as import("obsidian").Setting;
    library.items[0]!.render?.(setting);
    directions?.render?.(setting);
    expect(host.renderLibraryGuideRow).toHaveBeenCalledWith(setting);
    expect(host.renderLibraryDirectionsRow).toHaveBeenCalledWith(setting);
  });

  it("keeps host-specific render callbacks and output field routing", () => {
    const host=makeFullHost();host.renderOutputDirectoryRow=vi.fn();host.renderApiKeyRow=vi.fn();
    const items=buildSettingDefinitions(host);
    const all=items.flatMap(item=>item.type==='group'?item.items:[item]);
    const setting={} as import("obsidian").Setting;
    all.find(item=>item.name==='Daily reports folder')?.render?.(setting);
    all.find(item=>item.name==='Paper notes folder')?.render?.(setting);
    all.find(item=>item.name==='API key')?.render?.(setting);
    expect(host.renderOutputDirectoryRow).toHaveBeenNthCalledWith(1,setting,'dailyDir');
    expect(host.renderOutputDirectoryRow).toHaveBeenNthCalledWith(2,setting,'papersDir');
    expect(host.renderApiKeyRow).toHaveBeenCalledWith(setting);
  });

});
