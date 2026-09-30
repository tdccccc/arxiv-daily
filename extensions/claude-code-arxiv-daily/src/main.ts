import * as fs from "node:fs/promises";
import { parseArgs } from "node:util";
import { executeAgentCommand } from "./agent";

const MAX_INPUT_BYTES = 300 * 1024;

async function main(): Promise<void> {
  const abort = new AbortController();
  process.once("SIGINT", () => abort.abort());
  process.once("SIGTERM", () => abort.abort());
  try {
    const { values, positionals } = parseArgs({
      options: { workspace: { type: "string" }, input: { type: "string" } },
      allowPositionals: true, strict: true,
    });
    if (positionals.length > 1) throw new Error("Supply one command; put parameters in a JSON input file or stdin");
    const command = positionals[0] ?? "help";
    if (command === "help") {
      console.log(JSON.stringify({ ok: true, data: {
        usage: "node arxiv-agent.cjs COMMAND --workspace PATH [--input REQUEST.json]",
        commands: ["init", "status", "library", "recent", "paper", "save", "read", "confirm-direction"],
        input: "A JSON object in --input or stdin; omitted input means {}",
        output: "One JSON result; errors have ok:false and a nonzero exit code",
      } }));
      return;
    }
    let raw = "";
    if (values.input) {
      if ((await fs.stat(values.input)).size > MAX_INPUT_BYTES) throw new Error("Input exceeds 300 KiB");
      raw = await fs.readFile(values.input, "utf8");
    } else if (!process.stdin.isTTY) {
      let bytes = 0;
      const chunks: Buffer[] = [];
      for await (const chunk of process.stdin) {
        const buffer = Buffer.from(chunk);
        bytes += buffer.byteLength;
        if (bytes > MAX_INPUT_BYTES) throw new Error("Input exceeds 300 KiB");
        chunks.push(buffer);
      }
      raw = Buffer.concat(chunks).toString("utf8");
    }
    if (Buffer.byteLength(raw) > MAX_INPUT_BYTES) throw new Error("Input exceeds 300 KiB");
    let input: Record<string, unknown>;
    try { input = raw.trim() ? JSON.parse(raw) : {}; }
    catch { throw new Error("Invalid JSON input; use a JSON request file instead of shell interpolation"); }
    const data = await executeAgentCommand(command, values.workspace ?? process.cwd(), input, { signal: abort.signal });
    console.log(JSON.stringify({ ok: true, data }));
  } catch (error) {
    console.log(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) }));
    process.exitCode = abort.signal.aborted ? 130 : 1;
  }
}

void main();
