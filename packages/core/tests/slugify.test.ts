import { describe, it, expect } from "vitest";
import { slugify } from "../src/utils/slugify";

describe("slugify", () => {
  it("lowercases ASCII letters", () => {
    expect(slugify("Photo-z")).toBe("photo-z");
  });

  it("converts spaces to dashes", () => {
    expect(slugify("Galaxy Cluster")).toBe("galaxy-cluster");
  });

  it("converts underscores to dashes", () => {
    expect(slugify("photo_z_methods")).toBe("photo-z-methods");
  });

  it("collapses repeated separators", () => {
    expect(slugify("a   b___c")).toBe("a-b-c");
  });

  it("trims leading and trailing dashes", () => {
    expect(slugify("--hello--")).toBe("hello");
  });

  it("preserves CJK letters", () => {
    expect(slugify("Photo-z 相关")).toBe("photo-z-相关");
  });

  it("returns non-empty slugs for CJK-only input", () => {
    expect(slugify("光度红移")).toBe("光度红移");
    expect(slugify("")).toBe("");
    expect(slugify("   ")).toBe("");
  });

  it("normalizes accents and removes emoji and special symbols", () => {
    expect(slugify("Café déjà vu 🚀 / test!")).toBe("cafe-deja-vu-test");
  });

  it("preserves digits and dots-to-dashes", () => {
    expect(slugify("v0.1.1 release")).toBe("v0-1-1-release");
  });
});
