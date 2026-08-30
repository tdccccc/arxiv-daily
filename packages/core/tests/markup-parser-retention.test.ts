import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { setFlagsFromString } from "node:v8";
import { runInNewContext } from "node:vm";
import { markupParser } from "./markup-parser";

const here = dirname(fileURLToPath(import.meta.url));
const recentHtml = readFileSync(
  resolve(here, "./fixtures/arxiv-recent-astroph.html"),
  "utf8",
);

setFlagsFromString("--expose-gc");
const collectGarbage = runInNewContext("gc") as () => void;

/**
 * Parses in a nested frame so the parsed document is only ever reachable
 * through the returned WeakRef, never through a live stack slot here.
 */
function parseAndForget(markup: string, mimeType: "text/html" | "text/xml") {
  return new WeakRef(markupParser.parseFromString(markup, mimeType) as unknown as object);
}

describe("core test markup parser retention", () => {
  it("retains no parsed document once the caller drops it", async () => {
    // The shared test parser must hold no per-parse state. A parser that keeps
    // every parsed document alive (happy-dom's Window does, until it is closed
    // asynchronously) makes memory grow without bound across a suite that
    // parses the arXiv listing fixture once per pipeline construction, and
    // exhausts the vitest worker heap.
    const refs = [
      parseAndForget(recentHtml, "text/html"),
      parseAndForget(recentHtml, "text/html"),
      parseAndForget(recentHtml, "text/html"),
    ];

    // A WeakRef target stays alive for the rest of the job in which the WeakRef
    // was created, so yield to a fresh macrotask before collecting.
    await new Promise((done) => setTimeout(done, 0));
    collectGarbage();
    collectGarbage();

    expect(refs.filter((ref) => ref.deref() !== undefined)).toHaveLength(0);
  });

  it("keeps a parsed document usable while the caller holds it", () => {
    const document = markupParser.parseFromString(recentHtml, "text/html");
    collectGarbage();
    expect(document.querySelectorAll("dt").length).toBeGreaterThan(0);
  });
});
