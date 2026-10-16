// The notice body uses Obsidian's fragment helpers; the stub supplies them.
import "obsidian";
import { describe, expect, it, vi } from "vitest";
import {
  LIBRARY_CHANGE_NUDGE_ACTION,
  buildLibraryChangeNudgeFragment,
  describeLibraryChangeNudge,
  describeNewLibraryPapers,
} from "../src/library/change-nudge";

describe("library change nudge text", () => {
  it.each([0, 1, 2, -1, Number.NaN])("says nothing for %s new papers", (count) => {
    expect(describeLibraryChangeNudge(count)).toBeNull();
  });

  it("names the count and what it means in plain language", () => {
    expect(describeLibraryChangeNudge(3)).toBe(
      "3 new papers in your library aren't reflected in your direction suggestions yet.",
    );
    expect(describeLibraryChangeNudge(12)).toBe(
      "12 new papers in your library aren't reflected in your direction suggestions yet.",
    );
  });

  it("uses the singular for one paper wherever the count is named", () => {
    expect(describeNewLibraryPapers(1)).toBe("1 new paper");
    expect(describeNewLibraryPapers(2)).toBe("2 new papers");
  });

  it("offers one action, named for what it does", () => {
    expect(LIBRARY_CHANGE_NUDGE_ACTION).toBe("Update direction suggestions");
  });
});

describe("library change nudge notice body", () => {
  it("has no body below the nudge count", () => {
    expect(buildLibraryChangeNudgeFragment(2, vi.fn())).toBeNull();
  });

  it("carries the sentence and one button that runs the action", () => {
    const onAction = vi.fn();
    const fragment = buildLibraryChangeNudgeFragment(3, onAction)!;
    expect(fragment.textContent).toContain("3 new papers in your library");
    const buttons = Array.from(fragment.querySelectorAll("button"));
    expect(buttons.map((button) => button.textContent)).toEqual(["Update direction suggestions"]);
    expect(onAction).not.toHaveBeenCalled();
    buttons[0]!.click();
    expect(onAction).toHaveBeenCalledOnce();
  });
});
