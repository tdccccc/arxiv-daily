import { describe, expect, it } from "vitest";
import { redactError, redactText, redactUrl } from "../src/utils/redaction";

describe("secret redaction", () => {
  const secret = "sk-complete-secret-value";

  it("removes exact secrets and common credential forms without preserving fragments", () => {
    const safe = redactText(
      `key=${secret} Authorization: Bearer ${secret} api_key=${secret}`,
      { secrets: [secret] },
    );
    expect(safe).not.toContain(secret);
    expect(safe).not.toContain("sk-complete");
    expect(safe).toContain("[REDACTED]");
  });

  it("redacts sensitive URL credentials and query parameters", () => {
    const safe = redactUrl(
      `https://user:${secret}@example.test/v1?api_key=${secret}&model=x`,
      { secrets: [secret] },
    );
    expect(safe).not.toContain(secret);
    expect(safe).toContain("model=x");
  });

  // Local model servers need a non-empty key but ignore its value, so people enter a word.
  it.each(["local", "none", "test", "ollama", "EMPTY", "dummy", "lm-studio", "not-needed"])(
    "leaves ordinary text alone when the key is the placeholder %s",
    (placeholder) => {
      const text = `A ${placeholder} approach to calibration: ${placeholder.toUpperCase()} methods work on non-${placeholder} data.`;
      expect(redactText(text, { secrets: [placeholder] })).toBe(text);
    },
  );

  it("still redacts a short key where it is assigned", () => {
    const secrets = ["local"];
    expect(redactText("api_key=local", { secrets })).toBe("api_key=[REDACTED]");
    expect(redactText("Authorization: Bearer local", { secrets })).toBe("Authorization: Bearer [REDACTED]");
    expect(redactText('{"api_key":"local","note":"a local run"}', { secrets }))
      .toBe('{"api_key":"[REDACTED]","note":"a local run"}');
  });

  it("redacts a real key anywhere regardless of its length", () => {
    expect(redactText("echoed abcd1234 here", { secrets: ["abcd1234"] })).toBe("echoed [REDACTED] here");
    expect(redactText("echoed abcd123 here", { secrets: ["abcd123"] })).toBe("echoed [REDACTED] here");
  });

  it.each(["7", "X9", "x9Z", "x9Z2", "x9Z2q", "x9Z2q8", "x9Z2q8!"])(
    "redacts the short real key %s outside credential assignments",
    (key) => expect(redactText(`echoed ${key} here`, { secrets: [key] })).toBe("echoed [REDACTED] here"),
  );

  it("does not redact inside an earlier replacement or a pre-existing marker", () => {
    const options = { secrets: ["long-secret", "A", "D"] };
    const safe = redactText("long-secret A D [REDACTED]", options);
    expect(safe).toBe("[REDACTED] [REDACTED] [REDACTED] [REDACTED]");
    expect(redactText(safe, options)).toBe(safe);
  });

  it("treats regular expression metacharacters in a short key literally", () => {
    expect(redactText("echoed x.*$ here", { secrets: ["x.*$"] })).toBe("echoed [REDACTED] here");
  });

  it("skips a placeholder without dropping the real key listed beside it", () => {
    expect(redactText("local xk29-real-key-77 local", { secrets: ["local", "xk29-real-key-77"] }))
      .toBe("local [REDACTED] local");
  });

  it("does not leave the tail of a configured key that also matches a generic key prefix", () => {
    const key = "sk-example-key.suffix";
    expect(redactText(`echoed ${key} here`, { secrets: [key] })).toBe("echoed [REDACTED] here");
  });

  it("returns a sanitized Error while retaining status", () => {
    const source = Object.assign(new Error(`provider echoed ${secret}`), { status: 401 });
    const safe = redactError(source, { secrets: [secret] }) as Error & { status?: number };
    expect(safe.message).toBe("provider echoed [REDACTED]");
    expect(safe.status).toBe(401);
  });
});
