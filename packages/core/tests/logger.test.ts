import { describe, expect, it, vi } from "vitest";
import { Logger } from "../src/services/logger";

describe("Logger", () => {
  it("uses an injected notice sink when present", () => {
    const sink = vi.fn();
    const logger = new Logger("info", sink);

    logger.notice("hello", 1000);

    expect(sink).toHaveBeenCalledWith("hello", 1000);
  });

  it("falls back to info logging without a notice sink", () => {
    const logger = new Logger("info");
    const spy = vi.spyOn(console, "log").mockImplementation(() => {});

    logger.notice("hello");
    expect(spy).toHaveBeenCalledWith("[arxiv-daily]", "hello");
    spy.mockRestore();
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
