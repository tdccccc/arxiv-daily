import type { WorkbenchAsset } from "./server";

declare const __ARXIV_DAILY_WORKBENCH_ASSETS__: Record<string, WorkbenchAsset>;

/** The official build embeds all browser assets; runtime never depends on the checkout. */
export function workbenchAssets(): Record<string, WorkbenchAsset> {
  if (typeof __ARXIV_DAILY_WORKBENCH_ASSETS__ === "undefined") throw new Error("Build the CLI before opening the workbench: npm run build --workspace apps/cli");
  return __ARXIV_DAILY_WORKBENCH_ASSETS__;
}
