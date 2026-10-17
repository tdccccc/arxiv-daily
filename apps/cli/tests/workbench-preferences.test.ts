import { afterEach, expect, it } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { DEFAULT_UI_APPEARANCE, normalizeUiAppearancePreferences, validateUiAppearancePreferences } from "@arxiv-daily/core";
import { readPreferences, savePreferences } from "../src/workbench/preferences";
const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => fs.rm(root, { recursive: true, force: true }))); });
async function fixture() { const root = await fs.mkdtemp(path.join(os.tmpdir(), "ui-preferences-")); roots.push(root); return path.join(root, "nested", "config.toml"); }
it("shares pure appearance defaults and strict validation", () => {
  expect(DEFAULT_UI_APPEARANCE).toEqual({ theme: "light", language: "zh" });
  expect(normalizeUiAppearancePreferences(undefined)).toEqual(DEFAULT_UI_APPEARANCE);
  expect(validateUiAppearancePreferences({ theme: "system", language: "en" })).toBe(true);
  expect(validateUiAppearancePreferences({ theme: "invalid", language: "en" })).toBe(false);
  expect(validateUiAppearancePreferences({ theme: "dark", language: "en", extra: true })).toBe(false);
  const normalized = normalizeUiAppearancePreferences({ theme: "dark", language: "en" });
  normalized.theme = "light";
  expect(DEFAULT_UI_APPEARANCE.theme).toBe("light");
});
it("reads missing and legacy layout without appearance using defaults", async () => {
  const config = await fixture();
  expect(await readPreferences(config)).toEqual({ sidebarWidth: null, sidebarCollapsed: false, appearance: DEFAULT_UI_APPEARANCE });
  await fs.mkdir(path.dirname(config));
  await fs.writeFile(path.join(path.dirname(config), "workbench-ui.json"), JSON.stringify({ sidebarWidth: 620, sidebarCollapsed: true }));
  expect(await readPreferences(config)).toEqual({ sidebarWidth: 620, sidebarCollapsed: true, appearance: DEFAULT_UI_APPEARANCE });
});
it("merges independent appearance and layout writes atomically, including concurrent first writes", async () => {
  const config = await fixture();
  await Promise.all([
    savePreferences(config, { appearance: { theme: "dark", language: "en" } }),
    savePreferences(config, { sidebarWidth: 640, sidebarCollapsed: true }),
  ]);
  expect(await readPreferences(config)).toEqual({ sidebarWidth: 640, sidebarCollapsed: true, appearance: { theme: "dark", language: "en" } });
  await savePreferences(config, { sidebarCollapsed: false });
  await savePreferences(config, { appearance: { theme: "system", language: "zh" } });
  expect(await readPreferences(config)).toEqual({ sidebarWidth: 640, sidebarCollapsed: false, appearance: { theme: "system", language: "zh" } });
  if (process.platform !== "win32") expect((await fs.stat(path.join(path.dirname(config), "workbench-ui.json"))).mode & 0o777).toBe(0o600);
});
it.each([{ appearance: { theme: "invalid", language: "en" } }, { appearance: { theme: "dark", language: "xx" } }, { appearance: { theme: "dark" } }, { extra: true }, { sidebarWidth: 999 }, { sidebarCollapsed: "yes" }, { appearance: { theme: "dark", language: "en", extra: true } }])("rejects invalid patch without changing storage %j", async body => {
  const config = await fixture();
  await savePreferences(config, { sidebarWidth: 560, sidebarCollapsed: false });
  const before = await fs.readFile(path.join(path.dirname(config), "workbench-ui.json"), "utf8");
  await expect(savePreferences(config, body)).rejects.toMatchObject({ status: 400 });
  expect(await fs.readFile(path.join(path.dirname(config), "workbench-ui.json"), "utf8")).toBe(before);
});
async function writeStored(config: string, text: string) {
  await fs.mkdir(path.dirname(config), { recursive: true });
  await fs.writeFile(path.join(path.dirname(config), "workbench-ui.json"), text);
}
it.each(["{", "", "   ", "not json", "null", "[]", "42", '"text"'])("reads a damaged preferences file %j as defaults and repairs it on save", async text => {
  const config = await fixture();
  await writeStored(config, text);
  const warnings: string[] = [];
  expect(await readPreferences(config, message => warnings.push(message))).toEqual({ sidebarWidth: null, sidebarCollapsed: false, appearance: DEFAULT_UI_APPEARANCE });
  expect(warnings).toHaveLength(1);
  expect(warnings[0]).toContain("workbench-ui.json");
  const saved = await savePreferences(config, { sidebarCollapsed: true });
  expect(saved).toEqual({ sidebarWidth: null, sidebarCollapsed: true, appearance: DEFAULT_UI_APPEARANCE });
  expect(JSON.parse(await fs.readFile(path.join(path.dirname(config), "workbench-ui.json"), "utf8"))).toEqual(saved);
  expect(await readPreferences(config, () => { throw new Error("no warning expected after repair"); })).toEqual(saved);
});
it("keeps the valid fields of a partly invalid file and warns about each bad one", async () => {
  const config = await fixture();
  await writeStored(config, JSON.stringify({ sidebarWidth: 620, sidebarCollapsed: "yes", appearance: { theme: "nope", language: "en" }, futureField: 1 }));
  const warnings: string[] = [];
  expect(await readPreferences(config, message => warnings.push(message))).toEqual({ sidebarWidth: 620, sidebarCollapsed: false, appearance: DEFAULT_UI_APPEARANCE });
  expect(warnings.join("\n")).toContain('"sidebarCollapsed"');
  expect(warnings.join("\n")).toContain('"appearance"');
  expect(warnings.join("\n")).toContain('"futureField"');
  expect(warnings).toHaveLength(3);
});
it("still surfaces genuine I/O errors instead of hiding them behind defaults", async () => {
  const config = await fixture();
  await fs.mkdir(path.join(path.dirname(config), "workbench-ui.json"), { recursive: true });
  await expect(readPreferences(config)).rejects.toMatchObject({ code: "EISDIR" });
});
