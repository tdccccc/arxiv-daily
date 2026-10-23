import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { getBusinessSettingsSections } from "@arxiv-daily/core";

/** Line endings normalized: a Windows checkout may turn them into CRLF. */
const read = (relative: string) =>
  readFileSync(new URL(relative, import.meta.url), "utf8").replace(/\r\n/g, "\n");

const DOCS = {
  "plugin/README.md": read("../README.md"),
  "plugin/README.zh-CN.md": read("../README.zh-CN.md"),
  "docs/getting-started.md": read("../../docs/getting-started.md"),
  "docs/getting-started.zh-CN.md": read("../../docs/getting-started.zh-CN.md"),
};

/** What the settings page shows, from the shared schema plus the one section the plugin adds. */
const sections = getBusinessSettingsSections({}, { includeHidden: true });
const NAMES_BY_HEADING = new Map<string, string[]>();
const ROW_NAMES: string[] = ["Help & feedback"];
for (const section of sections) {
  if (section.type === "field") {
    ROW_NAMES.push(section.field.name);
  } else if (section.type === "list") {
    ROW_NAMES.push(section.heading);
  } else {
    ROW_NAMES.push(section.heading);
    NAMES_BY_HEADING.set(section.heading, section.items.map((item) => item.name));
  }
}
/** Rows the plugin's settings page has beyond the shared schema. */
const PAGE_ONLY_ROWS: Record<string, string[]> = { "Email delivery": ["Send test email"] };

function overviewRows(markdown: string): Array<{ name: string; covers: string }> {
  const lines = markdown.split("\n");
  const header = lines.findIndex((line) => /^\| (Section|分区) \| /.test(line));
  expect(header).toBeGreaterThan(-1);
  const rows: Array<{ name: string; covers: string }> = [];
  for (const line of lines.slice(header + 2)) {
    const match = /^\| \*\*(.+?)\*\* \| (.+) \|$/.exec(line);
    if (!match) break;
    rows.push({ name: match[1]!, covers: match[2]! });
  }
  return rows;
}

/** A row named like the "Enable · Paused" / "Enable · Running" switch matches on what comes before the dot. */
const isPageRow = (name: string) =>
  ROW_NAMES.some((row) => row === name || row.split(" · ")[0] === name);

describe("the settings overview in the plugin README", () => {
  for (const file of ["plugin/README.md", "plugin/README.zh-CN.md"] as const) {
    it(`${file} only lists rows and sections the settings page has`, () => {
      const rows = overviewRows(DOCS[file]);

      expect(rows.length).toBeGreaterThan(5);
      expect(rows.filter((row) => !isPageRow(row.name)).map((row) => row.name)).toEqual([]);
    });
  }

  it("plugin/README.md lists the settings of each section by their names on the page", () => {
    const rows = overviewRows(DOCS["plugin/README.md"]);

    for (const heading of ["LLM", "Output & schedule", "Email delivery", "Advanced"]) {
      const row = rows.find((candidate) => candidate.name === heading);
      expect(row, heading).toBeDefined();
      const listed = row!.covers.split(", ").sort();
      const expected = [...NAMES_BY_HEADING.get(heading)!, ...(PAGE_ONLY_ROWS[heading] ?? [])].sort();
      expect({ heading, listed }).toEqual({ heading, listed: expected });
    }
  });
});

describe("the documentation of the Obsidian plugin", () => {
  for (const [file, text] of Object.entries(DOCS)) {
    describe(file, () => {
      it("does not promise tick boxes in the daily report that sync marks", () => {
        expect(text).not.toMatch(/arxiv-daily:\d{4}\.\d{4,5}:(watch|highlight)/);
        expect(text).not.toMatch(/Daily selections|每日勾选/);
      });

      it("does not say an API key only ever displays as Configured or is changed with Replace and Clear", () => {
        expect(text).not.toMatch(/[`*]Configured[`*]/);
        expect(text).not.toMatch(/\*\*(Replace|Clear)\*\*/);
      });

      it("does not list settings the page does not have", () => {
        expect(text).not.toMatch(/Request delay|cache retention|section\/char limits|lookback days \(/i);
        expect(text).not.toMatch(/请求延迟|缓存保留|分段\/字符上限|回溯天数/);
      });
    });
  }
});
