import { createHash, randomBytes } from "node:crypto";
import { describe, expect, it } from "vitest";
import { sha256Hex } from "../src/utils/digest";

const reference = (input: string | Uint8Array): string => createHash("sha256").update(input).digest("hex");

describe("sha256Hex", () => {
  it("matches the published SHA-256 test vectors", () => {
    expect(sha256Hex("")).toBe("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    expect(sha256Hex("abc")).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    expect(sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))
      .toBe("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  });

  it("matches node:crypto at every length around the block and padding boundaries", () => {
    const data = randomBytes(1_200);
    for (let length = 0; length <= 300; length += 1) {
      expect(sha256Hex(data.subarray(0, length)), `length ${length}`).toBe(reference(data.subarray(0, length)));
    }
    for (const length of [511, 512, 513, 1_000, 1_199, 1_200]) {
      expect(sha256Hex(data.subarray(0, length)), `length ${length}`).toBe(reference(data.subarray(0, length)));
    }
  });

  it("hashes unaligned views and large inputs like node:crypto", () => {
    const backing = randomBytes(3 * 1024 * 1024 + 17);
    const view = backing.subarray(5, backing.length - 3);
    expect(sha256Hex(view)).toBe(reference(view));
  });

  it("encodes strings as UTF-8 and is repeatable across calls", () => {
    const text = "Galaxy cluster 星系团 \u{1F680} naïve";
    expect(sha256Hex(text)).toBe(reference(text));
    expect(sha256Hex(text)).toBe(sha256Hex(text));
    expect(sha256Hex("a".repeat(64))).toBe(reference("a".repeat(64)));
  });
});
