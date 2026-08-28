import http from "node:http";

/**
 * A real loopback listener the sidecar settings can point at.
 *
 * The plugin performs HTTP through Obsidian's `requestUrl`, which runs in the
 * Electron main process, so the renderer's CDP Network domain never sees it.
 * Binding an actual socket is therefore the only way to observe whether a
 * request was made — and its absence is then equally meaningful.
 *
 * Requests fail by default, which is exactly the condition the probe-failure
 * fallback is meant to handle.
 */
export async function startProbeListener({ status = 503, body = "probe listener: unavailable" } = {}) {
  const requests = [];
  const server = http.createServer((req, res) => {
    requests.push({ method: req.method, path: req.url });
    res.writeHead(status, { "content-type": "text/plain" });
    res.end(body);
  });

  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const { port } = server.address();
  const origin = `http://127.0.0.1:${port}`;

  let closed = false;
  return {
    origin,
    port,
    capabilitiesUrl: `${origin}/v1/capabilities`,
    parseUrl: `${origin}/v1/parse`,
    requests: () => [...requests],
    async close() {
      if (closed) return;
      closed = true;
      await new Promise((resolve) => server.close(resolve));
    },
  };
}
