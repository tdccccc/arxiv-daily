import { describe, expect, it, vi } from "vitest";
import { Logger } from "../src/services/logger";

describe("Logger", () => {
  it("uses an injected notice sink when present", () => {
    const sink = vi.fn();
    const logger = new Logger("info", sink);

    logger.notice("hello", 1000);

    expect(sink).toHaveBeenCalledWith("hello", 1000);
  });

  it("falls back to buffered info logging without a notice sink", () => {
    const logger = new Logger("info");
    const spy = vi.spyOn(console, "info").mockImplementation(() => {});

    logger.notice("hello");
    expect(logger.getBuffer().join("\n")).toContain("[INFO] hello");
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });

  it("keeps the default production console at warn/error while buffering info", () => {
    const logger = new Logger("info");
    const info = vi.spyOn(console, "info").mockImplementation(() => {});
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    logger.info("operational detail");
    logger.warn("actionable warning");

    expect(info).not.toHaveBeenCalled();
    expect(warn).toHaveBeenCalledWith("[arxiv-daily]", "actionable warning");
    expect(logger.getBuffer().join("\n")).toContain("[INFO] operational detail");
  });

  it("redacts sensitive values from console, notices, errors, and the buffer", () => {
    const sink = vi.fn();
    const logger = new Logger("debug", sink);
    const secret = "sk-complete-secret-value";
    logger.setSensitiveValues([secret]);
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});

    logger.error(`failed ${secret}`, new Error(`Bearer ${secret}`));
    logger.notice(`notice ${secret}`);

    expect(JSON.stringify(spy.mock.calls)).not.toContain(secret);
    expect(logger.getBuffer().join("\n")).not.toContain(secret);
    expect(JSON.stringify(sink.mock.calls)).not.toContain(secret);
  });

  it("re-sanitizes existing buffered entries when sensitive values change", () => {
    const logger = new Logger("info");
    vi.spyOn(console, "log").mockImplementation(() => {});
    logger.info("new-secret-value");
    logger.setSensitiveValues(["new-secret-value"]);
    expect(logger.getBuffer().join("\n")).not.toContain("new-secret-value");
  });

  it("keeps the latest 5000 buffered log entries", () => {
    const logger = new Logger("info");
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});

    for (let i = 0; i < 5001; i += 1) {
      logger.info(`entry-${i}`);
    }

    const buffer = logger.getBuffer();
    expect(buffer).toHaveLength(5000);
    expect(buffer[0]).toContain("entry-1");
    expect(buffer[4999]).toContain("entry-5000");
    spy.mockRestore();
  });
});
