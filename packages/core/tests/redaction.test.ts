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

  it("returns a sanitized Error while retaining status", () => {
    const source = Object.assign(new Error(`provider echoed ${secret}`), { status: 401 });
    const safe = redactError(source, { secrets: [secret] }) as Error & { status?: number };
    expect(safe.message).toBe("provider echoed [REDACTED]");
    expect(safe.status).toBe(401);
  });
});
