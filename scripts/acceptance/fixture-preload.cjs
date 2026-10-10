// Only the acceptance CLI process and its own children receive this preload.
// All application fetches terminate in the owned loopback fixture service.
const fixture = new URL(process.env.ARXIV_ACCEPTANCE_FIXTURE_URL);
if (fixture.protocol !== "http:" || fixture.hostname !== "127.0.0.1") throw new Error("Acceptance fixture must be an owned loopback HTTP service");
const originalFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = async (input, init = {}) => {
  const request = input instanceof Request ? input : undefined;
  const target = new URL(request?.url ?? String(input));
  const method = init.method ?? request?.method ?? "GET";
  const body = init.body ?? (request && !["GET", "HEAD"].includes(method) ? await request.clone().text() : undefined);
  const proxy = new URL("/proxy", fixture);
  proxy.searchParams.set("url", target.href);
  return originalFetch(proxy, {
    ...init, method, body,
    headers: init.headers ?? request?.headers,
    signal: init.signal ?? request?.signal,
    redirect: "error",
  });
};
