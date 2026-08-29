const ERROR_TYPES = new Set(["error", "assert"]);
const WARNING_TYPES = new Set(["warning", "warn"]);

function renderArgument(argument) {
  if (argument === null || argument === undefined) return "undefined";
  if ("value" in argument && argument.value !== undefined) return String(argument.value);
  if (argument.description) return argument.description;
  if (argument.className) return `[${argument.className}]`;
  return `[${argument.type ?? "unknown"}]`;
}

/**
 * Collects console output and uncaught exceptions for the lifetime of a
 * session, so "no console errors" becomes a checkable condition rather than an
 * impression.
 *
 * The harness attaches before the vault's trust dialog is accepted, and plugins
 * do not load until it is, so the whole plugin startup happens inside this
 * collection window.
 */
export async function createDiagnostics(client, { ignore = [] } = {}) {
  const collected = [];
  const suppressed = [];

  const record = (entry) => {
    if (ignore.some((pattern) => pattern.test(entry.text))) suppressed.push(entry);
    else collected.push(entry);
  };

  client.on("Runtime.consoleAPICalled", (params) => {
    record({
      source: "console",
      level: params.type,
      text: (params.args ?? []).map(renderArgument).join(" "),
    });
  });

  client.on("Runtime.exceptionThrown", (params) => {
    const { exception, text } = params.exceptionDetails ?? {};
    record({
      source: "pageerror",
      level: "error",
      text: exception?.description ?? text ?? "uncaught exception",
    });
  });

  await client.send("Runtime.enable");

  const entries = () => [...collected];
  const errors = () =>
    collected.filter((entry) => entry.source === "pageerror" || ERROR_TYPES.has(entry.level));
  const warnings = () => collected.filter((entry) => WARNING_TYPES.has(entry.level));

  return {
    entries,
    errors,
    warnings,
    ignored: () => [...suppressed],
    assertNoErrors() {
      const found = errors();
      if (found.length === 0) return;
      const detail = found.map((entry) => `  [${entry.source}] ${entry.text}`).join("\n");
      throw new Error(`the renderer reported ${found.length} error(s):\n${detail}`);
    },
  };
}
