import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  ARXIV_DAILY_DOCS_URL,
  ARXIV_DAILY_ISSUES_URL,
  ARXIV_DAILY_REPO_URL,
  buildBugReportUrl,
  buildFeatureRequestUrl,
} from "../src/feedback";

describe("feedback links", () => {
  it("points author and issue URLs at the public repository", () => {
    expect(ARXIV_DAILY_REPO_URL).toBe("https://github.com/tdccccc/arxiv-daily");
    expect(ARXIV_DAILY_ISSUES_URL).toBe(
      "https://github.com/tdccccc/arxiv-daily/issues",
    );
    expect(ARXIV_DAILY_DOCS_URL).toContain("/docs/getting-started.md");
  });

  it("builds a low-friction bug report URL with only the plugin version", () => {
    const url = buildBugReportUrl("0.3.1");
    expect(url.startsWith(`${ARXIV_DAILY_ISSUES_URL}/new?`)).toBe(true);
    const body = new URL(url).searchParams.get("body") ?? "";
    expect(body).toContain("- arXiv Daily: 0.3.1");
    expect(body).not.toContain("Obsidian:");
    expect(body).not.toContain("Platform:");
    expect(body).not.toContain("Steps to reproduce");
    expect(body).not.toContain("apiKey");
    expect(body).not.toMatch(/sk-[A-Za-z0-9]/);
  });

  it("opens a blank feature-request issue", () => {
    expect(buildFeatureRequestUrl()).toBe(`${ARXIV_DAILY_ISSUES_URL}/new`);
  });

  it("keeps dual manifests identical and includes authorUrl", () => {
    const root = JSON.parse(
      readFileSync(resolve(process.cwd(), "../manifest.json"), "utf-8"),
    );
    const plugin = JSON.parse(
      readFileSync(resolve(process.cwd(), "manifest.json"), "utf-8"),
    );
    expect(plugin).toEqual(root);
    expect(plugin.authorUrl).toBe(ARXIV_DAILY_REPO_URL);
  });

  it("wires settings and dashboard feedback entry points", () => {
    const settings = readFileSync(
      resolve(process.cwd(), "src/settings/tab.ts"),
      "utf-8",
    );
    const dashboard = readFileSync(
      resolve(process.cwd(), "src/dashboard/view.ts"),
      "utf-8",
    );
    expect(settings).toContain("Help & feedback");
    expect(settings).toContain("buildBugReportUrl");
    expect(settings).toContain("buildFeatureRequestUrl");
    expect(settings).toContain("A short description is enough");
    expect(settings).toContain("Write freely");
    expect(dashboard).toContain("Report a bug");
    expect(dashboard).toContain("Request a feature");
    expect(dashboard).toContain("buildBugReportUrl");
  });
});
