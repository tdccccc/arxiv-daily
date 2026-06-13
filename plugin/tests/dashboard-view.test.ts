import { describe, expect, it, vi } from "vitest";
import {
  ARXIV_DAILY_DASHBOARD_VIEW,
  openDashboardView,
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
