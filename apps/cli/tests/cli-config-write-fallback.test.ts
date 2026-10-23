import { expect, it, vi } from "vitest";
import { parse } from "smol-toml";
import { patch } from "@decimalturn/toml-patch";
import { patchConfigToml } from "../src/config-write";

vi.mock("@decimalturn/toml-patch", () => ({ patch: vi.fn() }));

it.each(["throws", "invalid", "wrong values"])("preserves every configuration value when the editor %s", (failure) => {
  vi.mocked(patch).mockImplementation(() => {
    if (failure === "throws") throw new Error("unsupported edit");
    return failure === "invalid" ? "invalid = [" : 'vault_root = "/wrong"';
  });
  const next = {
    vault_root: "/notes", llm: { api_key: "secret", model: "模型\nsecond line", thinking: true },
    output: { max_daily_papers: 25, score: 0.125 },
    arxiv: { categories: ["cs.AI", "astro-ph.CO"], topics: [{ name: "题目", directions: [{ id: "d", text: "quoted \"value\"" }] }] },
  };
  const saved = patchConfigToml('# comment\nvault_root = "/notes"\n', next);
  expect(parse(saved)).toEqual(next);
});
