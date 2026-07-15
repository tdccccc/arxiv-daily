import { describe, it, expect } from "vitest";
import { renderPrompt } from "../src/prompts/render";

describe("renderPrompt", () => {
  it("substitutes {{var}} placeholders", () => {
    expect(renderPrompt("a {{x}} b {{y}}", { x: "1", y: "2" })).toBe("a 1 b 2");
  });

  it("leaves single braces (JSON examples) untouched", () => {
    expect(renderPrompt('{"papers": [{{tag}}]}', { tag: "T" })).toBe(
      '{"papers": [T]}',
    );
  });

  it("throws on an unfilled placeholder", () => {
    expect(() => renderPrompt("a {{missing}}", { x: "1" })).toThrow(/missing/);
  });
});
