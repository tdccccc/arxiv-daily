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
});
