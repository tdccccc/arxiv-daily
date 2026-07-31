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

  it("returns top-level items with an Enable toggle and section groups", () => {
    const items = buildSettingDefinitions(makeHost());
    expect(items.length).toBeGreaterThanOrEqual(4);
    const groups = items.filter((item) => item.type === "group");
    expect(groups.map((g) => g.heading)).toEqual(
      expect.arrayContaining(["AI model", "Output & schedule", "Advanced", "Help & feedback"]),
    );
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
});
