import { describe, expect, it, vi } from "vitest";
import { sendSettingsTestEmail, requestSettingsEmailVerification } from "../src/services/settings-operations";
import { DEFAULT_SETTINGS } from "../src/settings/defaults";
import type { HttpRequest, StorageAdapter } from "../src/core/adapters";
import { DEFAULT_HOSTED_DELIVERY_BASE_URL } from "../src/delivery/hosted";

function fixture() {
  const settings = structuredClone(DEFAULT_SETTINGS);
  settings.arxiv.timezone = "Asia/Shanghai";
  settings.arxiv.categories = ["cs.AI", "cs.LG"];
  settings.output.summaryLanguage = "en";
  settings.output.dailyDir = "research/daily";
  settings.email = { ...settings.email, enabled: false, mode: "self", to: "reader@example.test", apiKey: "private-mail-key" };
  const files = new Map<string, string>();
  const storage: StorageAdapter = {
    normalizePath: path => path,
    readText: async path => { if (!files.has(path)) throw new Error("missing"); return files.get(path)!; },
    writeText: async (path, text) => { files.set(path, text); },
    writeTextAtomic: async (path, text) => { files.set(path, text); },
    exists: async path => files.has(path),
    mkdir: async () => {}, remove: async path => { files.delete(path); }, rename: async () => {}, list: async () => [],
  };
  const request = vi.fn(async (_request: HttpRequest) => ({ status: 200, headers: {}, bodyText: JSON.stringify({ id: "test-mail" }) }));
  return { settings, storage, http: {request}, files };
}

describe("shared settings email operations", () => {
  it("sends the same dated sample twice explicitly while automatic delivery remains disabled", async () => {
    const fixtureValue = fixture();
    const now = () => new Date("2026-10-03T18:30:00Z");
    const before = structuredClone(fixtureValue.settings);
    for (let index = 0; index < 2; index++) expect((await sendSettingsTestEmail({...fixtureValue, now})).kind).toMatch(/^delivered/);
    expect(fixtureValue.settings).toEqual(before);
    const requests = fixtureValue.http.request.mock.calls.map(([request]) => request);
    expect(requests).toHaveLength(2);
    expect(requests[0]!.headers?.Authorization).toBe("Bearer private-mail-key");
    const payload = JSON.parse(requests[0]!.body!);
    expect(payload.to).toEqual(["reader@example.test"]);
    expect(payload.subject).toContain("2026-10-04");
    expect(payload.html).toContain("cs.AI, cs.LG");
    expect(payload.text).toContain("research/daily/2026-10-04.md");
    expect(requests[0]!.headers?.["Idempotency-Key"]).not.toBe(requests[1]!.headers?.["Idempotency-Key"]);
    expect([...fixtureValue.files.values()].join(" ")).not.toContain("private-mail-key");
  });
  it("uses an explicitly selected report date", async () => {
    const value = fixture(); await sendSettingsTestEmail({...value,date:"2026-05-11"});
    expect(JSON.parse(value.http.request.mock.calls[0]![0].body!).subject).toContain("2026-05-11");
  });
  it("requests verification with a trimmed address, preserving the host's endpoint policy", async () => {
    const value = fixture(); value.settings.email.to = " reader@example.test ";
    value.settings.email.hostedBaseUrl = "https://ignored-config.example.test";
    await requestSettingsEmailVerification(value);
    expect(value.http.request.mock.calls[0]![0]).toMatchObject({url:DEFAULT_HOSTED_DELIVERY_BASE_URL + "/v1/verify/start", method:"POST", body:JSON.stringify({email:"reader@example.test"})});
    await requestSettingsEmailVerification({...value,baseUrl:"https://custom.example.test/"});
    expect(value.http.request.mock.calls[1]![0].url).toBe("https://custom.example.test/v1/verify/start");
    value.settings.email.to = " ";
    await expect(requestSettingsEmailVerification(value)).rejects.toThrow(/email address/);
    expect(value.http.request).toHaveBeenCalledTimes(2);
  });
});
