import { describe, expect, it } from "vitest";
import {
  isPlausibleEmail,
  normalizeEmail,
  sha256Hex,
  utcDateKey,
} from "../src/crypto";

describe("crypto helpers", () => {
  it("normalizes email", () => {
    expect(normalizeEmail("  A@B.Com ")).toBe("a@b.com");
  });

  it("checks plausible emails", () => {
    expect(isPlausibleEmail("a@b.co")).toBe(true);
    expect(isPlausibleEmail("nope")).toBe(false);
  });

  it("hashes stably", async () => {
    const a = await sha256Hex("x");
    const b = await sha256Hex("x");
    expect(a).toBe(b);
    expect(a).toHaveLength(64);
  });

  it("utc date key shape", () => {
    expect(utcDateKey(new Date("2026-07-27T15:00:00.000Z"))).toBe(
      "2026-07-27",
    );
  });
});
