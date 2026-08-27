/**
 * A small CDP client over the WebSocket that Node 22 provides natively, so the
 * harness needs no browser-automation dependency.
 */

/**
 * Obsidian shows `starter.html` when no vault is open. Testing that window
 * would silently produce evidence about the wrong thing, so it is refused.
 */
export function selectVaultTarget(targets) {
  const pages = targets.filter((target) => target.type === "page");
  if (pages.length === 0) {
    throw new Error("no page target exposed by Obsidian; the renderer never started");
  }
  const vault = pages.find((page) => page.url.includes("index.html"));
  if (!vault) {
    throw new Error(
      `no vault window found; Obsidian is showing ${pages.map((p) => p.url).join(", ")} — the vault did not open`,
    );
  }
  return vault;
}

export function createCdpClient({ url, createSocket = (target) => new WebSocket(target) }) {
  const socket = createSocket(url);
  const pending = new Map();
  const subscribers = new Map();
  let nextId = 1;
  let closedReason = null;

  const readyPromise = new Promise((resolve, reject) => {
    socket.addEventListener("open", () => resolve(), { once: true });
    socket.addEventListener(
      "error",
      (event) => reject(new Error(`could not connect to ${url}: ${event?.message ?? "socket error"}`)),
      { once: true },
    );
  });

  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id !== undefined && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) reject(new Error(`CDP ${message.error.code}: ${message.error.message}`));
      else resolve(message.result);
      return;
    }
    if (message.method) {
      for (const handler of subscribers.get(message.method) ?? []) handler(message.params);
    }
  });

  socket.addEventListener("close", () => {
    closedReason ??= new Error("CDP connection closed before the request was answered");
    for (const { reject } of pending.values()) reject(closedReason);
    pending.clear();
  });

  return {
    ready: () => readyPromise,
    send(method, params = {}) {
      if (closedReason) return Promise.reject(closedReason);
      const id = nextId++;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        socket.send(JSON.stringify({ id, method, params }));
      });
    },
    on(method, handler) {
      if (!subscribers.has(method)) subscribers.set(method, []);
      subscribers.get(method).push(handler);
    },
    close() {
      closedReason ??= new Error("CDP connection closed by the harness");
      socket.close();
    },
  };
}

/**
 * Evaluate in the renderer and surface a thrown expression as a rejection. A
 * silent `undefined` would let a broken assertion look like a passing one.
 */
export async function evaluate(client, expression) {
  const result = await client.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    const { exception, text } = result.exceptionDetails;
    throw new Error(exception?.description ?? text ?? "evaluation threw");
  }
  return result.result?.value;
}
