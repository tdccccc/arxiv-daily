import { describe, expect, it, vi } from "vitest";
import {
  ARXIV_DAILY_DASHBOARD_VIEW,
  executeObsidianCommand,
  openDashboardView,
  openMarkdownFileOnce,
} from "../src/dashboard/view";

describe("openDashboardView", () => {
  it("reveals an existing dashboard leaf", async () => {
    const leaf = { setViewState: vi.fn() };
    const workspace = {
      getLeavesOfType: vi.fn().mockReturnValue([leaf]),
      getLeaf: vi.fn(),
      revealLeaf: vi.fn().mockResolvedValue(undefined),
    };

    await openDashboardView({ app: { workspace } } as any);

    expect(workspace.getLeavesOfType).toHaveBeenCalledWith(
      ARXIV_DAILY_DASHBOARD_VIEW,
    );
    expect(workspace.revealLeaf).toHaveBeenCalledWith(leaf);
    expect(workspace.getLeaf).not.toHaveBeenCalled();
    expect(leaf.setViewState).not.toHaveBeenCalled();
  });

  it("creates a dashboard leaf when none exists", async () => {
    const leaf = { setViewState: vi.fn().mockResolvedValue(undefined) };
    const workspace = {
      getLeavesOfType: vi.fn().mockReturnValue([]),
      getLeaf: vi.fn().mockReturnValue(leaf),
      revealLeaf: vi.fn().mockResolvedValue(undefined),
    };

    await openDashboardView({ app: { workspace } } as any);

    expect(workspace.getLeaf).toHaveBeenCalledWith(true);
    expect(leaf.setViewState).toHaveBeenCalledWith({
      type: ARXIV_DAILY_DASHBOARD_VIEW,
      active: true,
    });
    expect(workspace.revealLeaf).toHaveBeenCalledWith(leaf);
  });
});

describe("openMarkdownFileOnce", () => {
  it("reveals an already open markdown file", async () => {
    const leaf = {
      getViewState: vi.fn().mockReturnValue({
        state: { file: "arxiv-daily/papers/2606.12345.md" },
      }),
    };
    const workspace = {
      getLeavesOfType: vi.fn().mockReturnValue([leaf]),
      revealLeaf: vi.fn().mockResolvedValue(undefined),
      openLinkText: vi.fn().mockResolvedValue(undefined),
    };

    await openMarkdownFileOnce(
      { workspace },
      "arxiv-daily/papers/2606.12345.md",
    );

    expect(workspace.getLeavesOfType).toHaveBeenCalledWith("markdown");
    expect(workspace.revealLeaf).toHaveBeenCalledWith(leaf);
    expect(workspace.openLinkText).not.toHaveBeenCalled();
  });

  it("opens the markdown file when no existing leaf matches", async () => {
    const workspace = {
      getLeavesOfType: vi.fn().mockReturnValue([
        { view: { file: { path: "arxiv-daily/papers/2606.54321.md" } } },
      ]),
      revealLeaf: vi.fn().mockResolvedValue(undefined),
      openLinkText: vi.fn().mockResolvedValue(undefined),
    };

    await openMarkdownFileOnce(
      { workspace },
      "arxiv-daily/papers/2606.12345.md",
    );

    expect(workspace.revealLeaf).not.toHaveBeenCalled();
    expect(workspace.openLinkText).toHaveBeenCalledWith(
      "arxiv-daily/papers/2606.12345.md",
      "",
      false,
    );
  });
});

describe("executeObsidianCommand", () => {
  it("uses executeCommandById when available", async () => {
    const executeCommandById = vi.fn().mockReturnValue(true);

    const executed = await executeObsidianCommand(
      { commands: { executeCommandById } },
      "arxiv-daily-run-for-date",
    );

    expect(executed).toBe(true);
    expect(executeCommandById).toHaveBeenCalledWith(
      "arxiv-daily-run-for-date",
    );
  });

  it("falls back to command callbacks when executeCommandById is unavailable", async () => {
    const callback = vi.fn();

    const executed = await executeObsidianCommand(
      {
        commands: {
          commands: {
            "arxiv-daily-run-for-date": { callback },
          },
        },
      },
      "arxiv-daily-run-for-date",
    );

    expect(executed).toBe(true);
    expect(callback).toHaveBeenCalledTimes(1);
  });

  it("returns false for missing commands", async () => {
    await expect(
      executeObsidianCommand({ commands: { commands: {} } }, "missing"),
    ).resolves.toBe(false);
  });
});
