import { describe, it, expect } from "vitest";
import { parseAtomAbstracts } from "../src/pipeline/atom-parser";

const sample = `<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2605.08080v1</id>
    <title>Paper one</title>
    <summary>This is the first abstract with some  whitespace.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2605.08068v2</id>
    <title>Paper two</title>
    <summary>Second abstract with line
    breaks    and indentation.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2605.08051</id>
    <title>Paper three (no version suffix)</title>
    <summary>Third abstract.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2605.00001v1</id>
    <title>Paper with empty summary</title>
    <summary></summary>
  </entry>
</feed>`;

describe("parseAtomAbstracts", () => {
  it("extracts abstracts keyed by base id", () => {
    const m = parseAtomAbstracts(sample);
    expect(m.size).toBe(3);
    expect(m.get("2605.08080")).toContain("first abstract");
    expect(m.get("2605.08068")).toContain("Second abstract");
    expect(m.get("2605.08051")).toBe("Third abstract.");
  });

  it("strips version suffix from id", () => {
    const m = parseAtomAbstracts(sample);
    expect(m.has("2605.08080v1")).toBe(false);
    expect(m.has("2605.08080")).toBe(true);
  });

  it("collapses internal whitespace and trims", () => {
    const m = parseAtomAbstracts(sample);
    expect(m.get("2605.08068")).toBe("Second abstract with line breaks and indentation.");
  });

  it("skips entries with empty summary", () => {
    const m = parseAtomAbstracts(sample);
    expect(m.has("2605.00001")).toBe(false);
  });

  it("returns empty map for empty feed", () => {
    const m = parseAtomAbstracts(
      `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>`,
    );
    expect(m.size).toBe(0);
  });
});
