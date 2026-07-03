import { describe, expect, it } from "vitest";
import { RunCancellationService } from "../src/services/cancellation";

describe("RunCancellationService", () => {
  it("scopes cancellation to dates active when cancellation was requested", () => {
    const cancellation = new RunCancellationService();
    const dateA = cancellation.begin("2026-05-11");

    expect(cancellation.cancelAll("stop A")).toEqual(["2026-05-11"]);
    const dateB = cancellation.begin("2026-05-12");
    cancellation.finish("2026-05-11");
    const dateC = cancellation.begin("2026-05-13");

    expect(dateA.aborted).toBe(true);
    expect(dateB.aborted).toBe(false);
    expect(dateC.aborted).toBe(false);
  });
});
