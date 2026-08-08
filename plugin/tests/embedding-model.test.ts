import { afterEach, describe, expect, it } from "vitest";
import {
  alignElectronReleaseProbe,
  describeRuntimeProbe,
} from "../src/hosts/obsidian/embedding-model";

const originalRelease = process.release;
const originalElectron = Object.getOwnPropertyDescriptor(process.versions, "electron");

function setElectronMarker(value: string): void {
  Object.defineProperty(process.versions, "electron", {
    value,
    configurable: true,
    writable: true,
  });
}

function clearElectronMarker(): void {
  delete (process.versions as Record<string, unknown>).electron;
}

afterEach(() => {
  Object.defineProperty(process, "release", {
    value: originalRelease,
    configurable: true,
    writable: true,
  });
  if (originalElectron) {
    Object.defineProperty(process.versions, "electron", originalElectron);
  } else {
    clearElectronMarker();
  }
});

describe("alignElectronReleaseProbe", () => {
  it("aligns release.name to electron when the electron marker is present", () => {
    setElectronMarker("33.4.11");
    Object.defineProperty(process, "release", {
      value: { name: "node" },
      configurable: true,
      writable: true,
    });

    alignElectronReleaseProbe();

    expect(process.release.name).toBe("electron");
  });

  it("is idempotent once aligned", () => {
    setElectronMarker("33.4.11");
    Object.defineProperty(process, "release", {
      value: { name: "electron" },
      configurable: true,
      writable: true,
    });

    alignElectronReleaseProbe();
    alignElectronReleaseProbe();

    expect(process.release.name).toBe("electron");
  });

  it("leaves a real Node process untouched (no electron marker)", () => {
    clearElectronMarker();
    Object.defineProperty(process, "release", {
      value: { name: "node" },
      configurable: true,
      writable: true,
    });

    alignElectronReleaseProbe();

    expect(process.release.name).toBe("node");
  });
});

describe("describeRuntimeProbe", () => {
  it("reports the electron marker when present", () => {
    setElectronMarker("33.4.11");
    Object.defineProperty(process, "release", {
      value: { name: "node" },
      configurable: true,
      writable: true,
    });
    expect(describeRuntimeProbe()).toContain("electron 33.4.11");
  });

  it("reports the node release name without the marker", () => {
    clearElectronMarker();
    Object.defineProperty(process, "release", {
      value: { name: "node" },
      configurable: true,
      writable: true,
    });
    expect(describeRuntimeProbe()).toBe("process.release.name=node (node)");
  });
});
