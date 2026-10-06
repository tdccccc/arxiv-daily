// @vitest-environment happy-dom
import { afterEach, expect, it, vi } from "vitest";
import { mountSidebar } from "../src/workbench/web/sidebar";

const cleanup: Array<() => void> = [];
afterEach(() => { cleanup.splice(0).forEach(fn => fn()); document.body.innerHTML = ""; vi.restoreAllMocks(); });

function setup(initial: { sidebarWidth: number | null; sidebarCollapsed: boolean } = { sidebarWidth: 560, sidebarCollapsed: false }) {
  Object.defineProperty(window, "innerWidth", { value: 1440, configurable: true, writable: true });
  const root = document.createElement("div"); root.dataset.view = "list";
  root.innerHTML = '<div class="header-actions"></div><div class="workspace"><aside class="library-pane"></aside><main class="reading-pane"></main><aside class="toc-pane"></aside></div>';
  document.body.append(root);
  let preferences = { ...initial };
  const request = vi.fn(async (_url: string, body?: unknown) => {
    if (body) preferences = { ...body as typeof initial };
    return { ...preferences };
  });
  const stop = mountSidebar(root, { request }); cleanup.push(stop);
  return { root, request, stop, preferences: () => preferences, separator: () => root.querySelector<HTMLElement>('[role="separator"]')! };
}

it("loads saved width and persists keyboard adjustment and collapse independently", async () => {
  const { root, request, separator, preferences } = setup();
  await vi.waitFor(() => expect(root.style.getPropertyValue("--sidebar-width")).toBe("560px"));
  expect(separator().getAttribute("aria-orientation")).toBe("vertical");
  separator().dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
  await vi.waitFor(() => expect(preferences().sidebarWidth).toBe(576));
  root.querySelector<HTMLButtonElement>(".sidebar-toggle")!.click();
  await vi.waitFor(() => expect(preferences().sidebarCollapsed).toBe(true));
  expect(root.classList.contains("sidebar-collapsed")).toBe(true);
  expect(preferences().sidebarWidth).toBe(576);
  expect(request.mock.calls[0]?.[0]).toBe("api/preferences");
});

it("bounds pointer resizing to preserve the reader and restores the chosen width after viewport changes", async () => {
  const { root, separator, preferences } = setup({ sidebarWidth: 700, sidebarCollapsed: false });
  await vi.waitFor(() => expect(separator()).toBeTruthy());
  separator().dispatchEvent(new PointerEvent("pointerdown", { button: 0, pointerId: 1, clientX: 700, bubbles: true }));
  window.dispatchEvent(new PointerEvent("pointermove", { pointerId: 1, clientX: 5000 }));
  window.dispatchEvent(new PointerEvent("pointerup", { pointerId: 1, clientX: 5000 }));
  await vi.waitFor(() => expect(preferences().sidebarWidth).toBeLessThanOrEqual(900));
  expect(Number.parseInt(root.style.getPropertyValue("--sidebar-width"))).toBeLessThanOrEqual(1012);
  const saved = preferences().sidebarWidth;
  Object.defineProperty(window, "innerWidth", { value: 900, configurable: true }); window.dispatchEvent(new Event("resize"));
  expect(Number.parseInt(root.style.getPropertyValue("--sidebar-width"))).toBeLessThanOrEqual(472);
  expect(preferences().sidebarWidth).toBe(saved);
  Object.defineProperty(window, "innerWidth", { value: 1440, configurable: true }); window.dispatchEvent(new Event("resize"));
  expect(Number.parseInt(root.style.getPropertyValue("--sidebar-width"))).toBe(saved);
});

it("keeps local interaction when a slow preference load finishes later", async () => {
  let resolveLoad!: (value: unknown) => void;
  const root = document.createElement("div");root.dataset.view = "list";
  root.innerHTML = '<div class="header-actions"></div><div class="workspace"><aside class="library-pane"></aside><main class="reading-pane"></main><aside class="toc-pane"></aside></div>';
  document.body.append(root);
  const request = vi.fn((url: string, body?: unknown) => body ? Promise.resolve(body) : new Promise(resolve => { resolveLoad = resolve; }));
  cleanup.push(mountSidebar(root, { request: request as <T>(url: string, body?: unknown) => Promise<T> }));
  const separator = root.querySelector<HTMLElement>('[role="separator"]');
  expect(separator).toBeTruthy();
  separator!.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
  const chosen = root.style.getPropertyValue("--sidebar-width");
  resolveLoad({ sidebarWidth: 300, sidebarCollapsed: true });
  await new Promise(resolve => setTimeout(resolve, 10));
  expect(root.style.getPropertyValue("--sidebar-width")).toBe(chosen);
  expect(root.classList.contains("sidebar-collapsed")).toBe(false);
});

it("adapts to mobile and reports failed preference saves without blocking local resizing", async () => {
  const { root, request, separator } = setup();
  await vi.waitFor(() => expect(root.style.getPropertyValue("--sidebar-width")).toBe("560px"));
  request.mockRejectedValueOnce(new Error("write failed"));
  separator().dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }));
  await vi.waitFor(() => expect(root.textContent).toContain("布局偏好未保存"));
  expect(root.style.getPropertyValue("--sidebar-width")).toBe("544px");
  Object.defineProperty(window, "innerWidth", { value: 390, configurable: true }); window.dispatchEvent(new Event("resize"));
  expect(separator().hidden).toBe(true);
});
