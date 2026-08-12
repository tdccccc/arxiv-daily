/* Test-only stub for the `obsidian` package. Mirrors the runtime surface we use. */

export class Notice {
  static calls: Array<{ message: string; timeoutMs?: number }> = [];

  constructor(message: string, timeoutMs?: number) {
    Notice.calls.push({ message, timeoutMs });
  }
}

export function normalizePath(p: string): string {
  return p;
}

export function requireApiVersion(_version: string): boolean {
  return true;
}

export function setIcon(_parent: HTMLElement, _iconId: string): void {}

/**
 * Mirrors the official loader's contract: resolves to a pdf.js library
 * object and makes it reachable via `window.pdfjsLib` (the production path
 * reads the window global, so the mock sets it like the real loader does).
 */
export async function loadPdfJs(): Promise<{ version: string }> {
  const lib = { version: "mock-pdfjs" };
  (globalThis as unknown as { pdfjsLib?: { version: string } }).pdfjsLib = lib;
  return lib;
}

export class TFile {
  constructor(readonly path: string = "") {}
}

export interface Vault {
  adapter: {
    read(path: string): Promise<string>;
    write(path: string, content: string): Promise<void>;
    exists(path: string): Promise<boolean>;
    mkdir(path: string): Promise<void>;
    rename(from: string, to: string): Promise<void>;
    copy(from: string, to: string): Promise<void>;
    remove(path: string): Promise<void>;
    readBinary?(path: string): Promise<ArrayBuffer>;
    writeBinary?(path: string, content: ArrayBuffer): Promise<void>;
  };
}

export interface WorkspaceLeaf {
  setViewState(state: any): Promise<void>;
}

export interface Workspace {
  getLeavesOfType(type: string): WorkspaceLeaf[];
  getLeaf(newLeaf?: boolean): WorkspaceLeaf | null;
  setActiveLeaf(leaf: WorkspaceLeaf, options?: { focus?: boolean }): void;
  revealLeaf(leaf: WorkspaceLeaf): Promise<void>;
  detachLeavesOfType(type: string): Promise<void>;
  openLinkText(path: string, sourcePath: string, newLeaf?: boolean): Promise<void>;
}

export interface App {
  vault?: Vault;
  workspace?: Workspace;
}

export class Plugin {}

export class ItemView {
  contentEl: HTMLElement = (globalThis as any).document?.createElement?.("div") ?? ({} as any);
  constructor(readonly leaf: WorkspaceLeaf) {}
  getViewType(): string { return ""; }
  getDisplayText(): string { return ""; }
  getIcon(): string { return ""; }
  onOpen(): Promise<void> | void {}
  onClose(): Promise<void> | void {}
}

export class PluginSettingTab {
  readonly containerEl: HTMLElement;

  constructor(_app: App, _plugin: Plugin) {
    this.containerEl = document.createElement("div");
  }

  update(): void {}
}

export class TextComponent {
  readonly inputEl: HTMLInputElement;
  private callback?: (value: string) => unknown;

  constructor(container: HTMLElement) {
    this.inputEl = document.createElement("input");
    this.inputEl.type = "text";
    container.appendChild(this.inputEl);
  }

  setPlaceholder(value: string): this {
    this.inputEl.placeholder = value;
    return this;
  }

  setValue(value: string): this {
    this.inputEl.value = value;
    return this;
  }

  onChange(callback: (value: string) => unknown): this {
    this.callback = callback;
    this.inputEl.addEventListener("change", () => {
      void this.callback?.(this.inputEl.value);
    });
    return this;
  }

  async trigger(value: string): Promise<void> {
    this.inputEl.value = value;
    await this.callback?.(value);
  }
}

export class TextAreaComponent {
  readonly inputEl: HTMLTextAreaElement;
  private callback?: (value: string) => unknown;

  constructor(container: HTMLElement) {
    this.inputEl = document.createElement("textarea");
    container.appendChild(this.inputEl);
  }

  setValue(value: string): this {
    this.inputEl.value = value;
    return this;
  }

  onChange(callback: (value: string) => unknown): this {
    this.callback = callback;
    return this;
  }
}

export class DropdownComponent {
  readonly selectEl: HTMLSelectElement;
  private callback?: (value: string) => unknown;

  constructor(container: HTMLElement) {
    this.selectEl = document.createElement("select");
    container.appendChild(this.selectEl);
  }

  addOption(value: string, label: string): this {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    this.selectEl.appendChild(option);
    return this;
  }

  setValue(value: string): this {
    this.selectEl.value = value;
    return this;
  }

  onChange(callback: (value: string) => unknown): this {
    this.callback = callback;
    this.selectEl.addEventListener("change", () => {
      void this.callback?.(this.selectEl.value);
    });
    return this;
  }

  async trigger(value: string): Promise<void> {
    if (!Array.from(this.selectEl.options).some((option) => option.value === value)) {
      this.addOption(value, value);
    }
    this.selectEl.value = value;
    await this.callback?.(value);
  }
}

export class ButtonComponent {
  readonly buttonEl: HTMLButtonElement;
  private callback?: () => unknown;

  constructor(container: HTMLElement) {
    this.buttonEl = document.createElement("button");
    container.appendChild(this.buttonEl);
  }

  setButtonText(value: string): this {
    this.buttonEl.textContent = value;
    return this;
  }

  setDisabled(value: boolean): this {
    this.buttonEl.disabled = value;
    return this;
  }

  setCta(): this { return this; }

  onClick(callback: () => unknown): this {
    this.callback = callback;
    this.buttonEl.addEventListener("click", () => { void this.callback?.(); });
    return this;
  }
}

export class Setting {
  static readonly instances: Setting[] = [];
  readonly settingEl: HTMLElement;
  readonly nameEl: HTMLElement;
  readonly descEl: HTMLElement;
  readonly controlEl: HTMLElement;
  readonly components: Array<TextComponent | TextAreaComponent | ToggleComponent | DropdownComponent | ButtonComponent> = [];

  constructor(container: HTMLElement) {
    this.settingEl = document.createElement("div");
    this.settingEl.className = "setting-item";
    this.nameEl = document.createElement("div");
    this.nameEl.className = "setting-item-name";
    this.descEl = document.createElement("div");
    this.descEl.className = "setting-item-description";
    this.controlEl = document.createElement("div");
    this.controlEl.className = "setting-item-control";
    this.settingEl.append(this.nameEl, this.descEl, this.controlEl);
    container.appendChild(this.settingEl);
    Setting.instances.push(this);
  }

  static reset(): void {
    Setting.instances.length = 0;
  }

  setName(value: string) { this.nameEl.textContent = value; return this; }
  setDesc(value: string) { this.descEl.textContent = value; return this; }
  setHeading() { return this; }
  addText(cb: (component: TextComponent) => unknown) {
    const component = new TextComponent(this.controlEl);
    this.components.push(component);
    cb(component);
    return this;
  }
  addTextArea(cb: (component: TextAreaComponent) => unknown) {
    const component = new TextAreaComponent(this.controlEl);
    this.components.push(component);
    cb(component);
    return this;
  }
  addToggle(cb: (component: ToggleComponent) => unknown) {
    const component = new ToggleComponent(this.controlEl);
    this.components.push(component);
    cb(component);
    return this;
  }
  addDropdown(cb: (component: DropdownComponent) => unknown) {
    const component = new DropdownComponent(this.controlEl);
    this.components.push(component);
    cb(component);
    return this;
  }
  addButton(cb: (component: ButtonComponent) => unknown) {
    const component = new ButtonComponent(this.controlEl);
    this.components.push(component);
    cb(component);
    return this;
  }
}

export class ToggleComponent {
  static readonly instances: ToggleComponent[] = [];
  readonly toggleEl: HTMLInputElement;
  value = false;
  private callback?: (value: boolean) => unknown;

  constructor(container: HTMLElement) {
    this.toggleEl = document.createElement("input");
    this.toggleEl.type = "checkbox";
    container.appendChild(this.toggleEl);
    ToggleComponent.instances.push(this);
  }

  setValue(value: boolean): this {
    this.value = value;
    this.toggleEl.checked = value;
    return this;
  }

  onChange(callback: (value: boolean) => unknown): this {
    this.callback = callback;
    return this;
  }

  async trigger(value: boolean): Promise<void> {
    this.value = value;
    await this.callback?.(value);
  }

  static reset(): void {
    ToggleComponent.instances.length = 0;
  }
}

export class Modal {
  static opened: Modal[] = [];
  titleEl: HTMLElement = (globalThis as any).document?.createElement?.("div") ?? ({} as any);
  contentEl: HTMLElement = (globalThis as any).document?.createElement?.("div") ?? ({} as any);
  constructor(_app: App) {}
  open(): void {
    Modal.opened.push(this);
    this.onOpen();
  }
  close(): void {
    this.onClose();
  }
  onOpen(): void {}
  onClose(): void {}
}

export class MenuItem {
  setTitle(_t: string) { return this; }
  setIcon(_icon: string | null) { return this; }
  setDisabled(_d: boolean) { return this; }
  onClick(_cb: (evt: MouseEvent | KeyboardEvent) => any) { return this; }
}

export class Menu {
  addItem(cb: (item: MenuItem) => any): this {
    cb(new MenuItem());
    return this;
  }
  addSeparator(): this { return this; }
  showAtMouseEvent(_evt: MouseEvent): this { return this; }
  showAtPosition(_p: { x: number; y: number }): this { return this; }
}

export async function requestUrl(_opts: any): Promise<{ status: number; text: string }> {
  return { status: 200, text: "" };
}
