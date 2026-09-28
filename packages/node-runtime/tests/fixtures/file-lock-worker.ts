import * as fs from "node:fs/promises";
import * as path from "node:path";
import { NodeFileLock } from "../../src/file-lock";

async function main() {
  const [vault, lockRoot, mode, count = "10"] = process.argv.slice(2) as [string, string, string, string?];
  const locks = new NodeFileLock(vault, { lockRoot });
  if (mode === "hold") {
    const lease = await locks.acquire("counter", { wait: true });
    if (!lease) throw new Error("lock unavailable");
    process.send?.("acquired");
    await new Promise<void>((resolve) => process.once("message", () => resolve()));
    await lease.release();
  } else {
    for (let i = 0; i < Number(count); i += 1) {
      const lease = await locks.acquire("counter", { wait: true });
      if (!lease) throw new Error("lock unavailable");
      try {
        const marker = await fs.open(path.join(vault, "inside"), "wx");
        await marker.close();
        const value = Number(await fs.readFile(path.join(vault, "counter"), "utf8"));
        await new Promise((resolve) => setTimeout(resolve, 4));
        await fs.writeFile(path.join(vault, "counter"), String(value + 1));
        await fs.unlink(path.join(vault, "inside"));
      } finally {
        await lease.release();
      }
    }
  }
  process.disconnect?.();
}
void main().catch((error) => { console.error(error); process.exit(1); });
