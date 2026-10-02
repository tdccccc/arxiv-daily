/** Embedding is opt-in for one exact local application, never a broad browser origin. */
export function validateFrameOrigin(value: string): string {
  // Fixed, application-owned Desktop origin; never accept arbitrary custom schemes.
  if (value === "dsh-app://app") return value;
  let url: URL;
  try { url = new URL(value); } catch { throw new Error("ui --frame-origin requires an exact loopback HTTP(S) origin"); }
  if (!["http:", "https:"].includes(url.protocol) || !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname) || url.username || url.password || value !== url.origin) {
    throw new Error("ui --frame-origin requires an exact loopback HTTP(S) origin without a path, query or credentials");
  }
  return url.origin;
}
