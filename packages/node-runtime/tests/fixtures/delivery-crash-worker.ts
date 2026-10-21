import { createRequire } from "node:module";
import { DEFAULT_SETTINGS, deliverDailyEmailIfEnabled, sampleDailyDigest } from "@arxiv-daily/core";
import { NodeStorageAdapter } from "../../src/storage-adapter";
import type { NativeStorageBinding } from "../../src/native-private-storage";

/**
 * Starts an automatic delivery and kills itself right after the Nth exclusive
 * create returns its file but before the content is written, leaving the
 * zero-length file a real crash would. Usage: <vault> <binding|-> <date> <kill-on>
 */
async function main() {
  const [vault, bindingPath, date, killOn] = process.argv.slice(2) as [string, string, string, "claim" | "decision" | "result"];
  const killAt = { claim: 1, decision: 2, result: 3 }[killOn];
  let creates = 0;
  const nativeBinding = bindingPath === "-" ? undefined : createRequire(__filename)(bindingPath) as NativeStorageBinding;
  const storage = new NodeStorageAdapter(vault, {
    ...(nativeBinding ? { nativeBinding } : {}),
    afterTargetCreated: () => {
      creates += 1;
      if (creates === killAt) process.kill(process.pid, "SIGKILL");
    },
  });
  await deliverDailyEmailIfEnabled(sampleDailyDigest({ date }), {
    storage,
    http: { request: async () => ({ status: 200, headers: {}, bodyText: JSON.stringify({ id: "worker-message" }) }) },
    output: DEFAULT_SETTINGS.output,
    email: { enabled: true, mode: "self", to: "crash@example.test", fromEmail: "from@example.test", apiKey: "re_test" },
    sleep: async () => {},
  });
  // Reaching this point means the kill never happened: fail loudly instead of hanging the test.
  process.exit(3);
}
void main().catch((error) => { console.error(error); process.exit(1); });
