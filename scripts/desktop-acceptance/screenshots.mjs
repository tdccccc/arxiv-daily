import fsPromises from "node:fs/promises";
import path from "node:path";

/**
 * Screenshots of the real renderer, written to a git-ignored directory so a
 * person can look at the states an assertion only describes in words.
 *
 * These are evidence for human judgement, not a comparison baseline: nothing
 * here diffs pixels, and no assertion depends on a stored image. Wording,
 * timing and visual polish still need a person, and this is what they look at.
 */

/** Relative to the repository root; `.acceptance-out` is already git-ignored. */
export const DEFAULT_SCREENSHOT_DIR = path.join(".acceptance-out", "desktop-acceptance");

export function resolveScreenshotDir({ repoRoot, env = process.env }) {
  const override = env.OBSIDIAN_ACCEPTANCE_SCREENSHOT_DIR;
  if (typeof override === "string" && override.trim().length > 0) {
    return path.isAbsolute(override) ? override : path.resolve(repoRoot, override);
  }
  return path.resolve(repoRoot, DEFAULT_SCREENSHOT_DIR);
}

/**
 * A file name has to say which state the image shows without opening it, and
 * must not be able to escape the output directory.
 */
export function screenshotFileName(name) {
  if (typeof name !== "string" || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(name)) {
    throw new TypeError(
      `screenshot name must be a lowercase, hyphen-separated slug, received ${JSON.stringify(name)}`,
    );
  }
  return `${name}.png`;
}

/**
 * Round a rectangle out to whole pixels and pad it, so a clip never shaves the
 * border off the element it is meant to show. Negative origins are clamped:
 * CDP refuses a clip that starts outside the page.
 */
export function padClip(rect, { padding = 8 } = {}) {
  if (!rect || !Number.isFinite(rect.width) || !Number.isFinite(rect.height)) {
    throw new TypeError(`screenshot clip needs a measured rectangle, received ${JSON.stringify(rect)}`);
  }
  if (rect.width <= 0 || rect.height <= 0) {
    throw new Error(
      `refusing to capture a ${rect.width}x${rect.height} region: the element is not laid out`,
    );
  }
  const x = Math.max(0, Math.floor(rect.x - padding));
  const y = Math.max(0, Math.floor(rect.y - padding));
  return {
    x,
    y,
    width: Math.ceil(rect.x + rect.width + padding) - x,
    height: Math.ceil(rect.y + rect.height + padding) - y,
    scale: 1,
  };
}

/**
 * Measure one element, or the span from one element to another, in page
 * coordinates. Returning `null` rather than throwing lets the caller report a
 * missing element as a failed assertion instead of a harness crash.
 */
export function elementRectExpression({ selector, throughSelector }) {
  return `(() => {
    const first = document.querySelector(${JSON.stringify(selector)});
    if (!first) return null;
    const last = ${throughSelector ? `document.querySelector(${JSON.stringify(throughSelector)})` : "first"};
    if (!last) return null;
    first.scrollIntoView({ block: "start" });
    const a = first.getBoundingClientRect();
    const b = last.getBoundingClientRect();
    const left = Math.min(a.left, b.left) + window.scrollX;
    const top = Math.min(a.top, b.top) + window.scrollY;
    const right = Math.max(a.right, b.right) + window.scrollX;
    const bottom = Math.max(a.bottom, b.bottom) + window.scrollY;
    return JSON.stringify({ x: left, y: top, width: right - left, height: bottom - top });
  })()`;
}

/**
 * Capture PNGs through the debugging protocol the session already owns, so the
 * harness still needs no browser-automation dependency.
 */
export async function createScreenshotWriter({
  client,
  evaluate,
  outputDir,
  fs = fsPromises,
}) {
  await fs.mkdir(outputDir, { recursive: true });
  // Screenshots are served by the Page domain; enabling it is idempotent.
  await client.send("Page.enable");
  const written = [];

  /**
   * @param name  slug that names the state, not the element
   * @param where `{ rect }`, or `{ selector, throughSelector }` measured in the
   *              renderer just before the capture
   */
  async function capture(name, where) {
    const file = path.join(outputDir, screenshotFileName(name));
    let rect = where.rect;
    if (!rect) {
      const raw = await evaluate(elementRectExpression(where));
      if (typeof raw !== "string") {
        throw new Error(
          `nothing matched ${JSON.stringify(where.selector)}, so ${name} would have captured the wrong thing`,
        );
      }
      rect = JSON.parse(raw);
    }
    const clip = padClip(rect, where);
    // The clip is deliberately taken from inside the current viewport rather
    // than with `captureBeyondViewport`: expanding the viewport for a capture
    // re-runs layout, which moves the very element that was just measured.
    // Callers make the region visible by sizing the viewport instead.
    const { data } = await client.send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: false,
      clip,
    });
    await fs.writeFile(file, Buffer.from(data, "base64"));
    const entry = { name, file, clip };
    written.push(entry);
    return entry;
  }

  return { capture, written: () => [...written], outputDir };
}
