/**
 * Synchronous, host-neutral SHA-256 (hex). Core cannot use Node's crypto, and
 * Web Crypto is async, while many call sites (block checksums, term buckets)
 * are synchronous, so this stays pure JS. It sits on the read path of every
 * full-text search (each object read verifies its checksum), so it is written
 * for speed: no per-call padding copy of the whole input, no per-word DataView
 * calls, no destructuring or closures in the round loop, and one reused
 * message schedule. Output is bit-identical to any SHA-256.
 */
const K = new Int32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const HEX: string[] = Array.from({ length: 256 }, (_, value) => value.toString(16).padStart(2, "0"));
const textEncoder = new TextEncoder();
// Shared scratch (JS is single-threaded and sha256Hex never yields), so no per-call allocation.
const W = new Int32Array(64);
const H = new Int32Array(8);

/** Processes 64-byte blocks of `bytes` from `offset` for `blockCount` blocks into H. */
function compress(bytes: Uint8Array, offset: number, blockCount: number): void {
  let h0 = H[0]!, h1 = H[1]!, h2 = H[2]!, h3 = H[3]!, h4 = H[4]!, h5 = H[5]!, h6 = H[6]!, h7 = H[7]!;
  let position = offset;
  for (let block = 0; block < blockCount; block += 1, position += 64) {
    for (let i = 0; i < 16; i += 1) {
      const p = position + i * 4;
      W[i] = (bytes[p]! << 24) | (bytes[p + 1]! << 16) | (bytes[p + 2]! << 8) | bytes[p + 3]!;
    }
    for (let i = 16; i < 64; i += 1) {
      const x = W[i - 15]!;
      const y = W[i - 2]!;
      const s0 = ((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3);
      const s1 = ((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10);
      W[i] = (W[i - 16]! + s0 + W[i - 7]! + s1) | 0;
    }
    let a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
    // Eight rounds per iteration with the working variables renamed instead of shifted.
    for (let i = 0; i < 64; i += 8) {
      { const t1 = (h + ((((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7)))) + ((e & f) ^ (~e & g)) + K[i + 0]! + W[i + 0]!) | 0;
        d = (d + t1) | 0;
        h = (t1 + (((((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10)))) + ((a & b) ^ (a & c) ^ (b & c)))) | 0; }
      { const t1 = (g + ((((d >>> 6) | (d << 26)) ^ ((d >>> 11) | (d << 21)) ^ ((d >>> 25) | (d << 7)))) + ((d & e) ^ (~d & f)) + K[i + 1]! + W[i + 1]!) | 0;
        c = (c + t1) | 0;
        g = (t1 + (((((h >>> 2) | (h << 30)) ^ ((h >>> 13) | (h << 19)) ^ ((h >>> 22) | (h << 10)))) + ((h & a) ^ (h & b) ^ (a & b)))) | 0; }
      { const t1 = (f + ((((c >>> 6) | (c << 26)) ^ ((c >>> 11) | (c << 21)) ^ ((c >>> 25) | (c << 7)))) + ((c & d) ^ (~c & e)) + K[i + 2]! + W[i + 2]!) | 0;
        b = (b + t1) | 0;
        f = (t1 + (((((g >>> 2) | (g << 30)) ^ ((g >>> 13) | (g << 19)) ^ ((g >>> 22) | (g << 10)))) + ((g & h) ^ (g & a) ^ (h & a)))) | 0; }
      { const t1 = (e + ((((b >>> 6) | (b << 26)) ^ ((b >>> 11) | (b << 21)) ^ ((b >>> 25) | (b << 7)))) + ((b & c) ^ (~b & d)) + K[i + 3]! + W[i + 3]!) | 0;
        a = (a + t1) | 0;
        e = (t1 + (((((f >>> 2) | (f << 30)) ^ ((f >>> 13) | (f << 19)) ^ ((f >>> 22) | (f << 10)))) + ((f & g) ^ (f & h) ^ (g & h)))) | 0; }
      { const t1 = (d + ((((a >>> 6) | (a << 26)) ^ ((a >>> 11) | (a << 21)) ^ ((a >>> 25) | (a << 7)))) + ((a & b) ^ (~a & c)) + K[i + 4]! + W[i + 4]!) | 0;
        h = (h + t1) | 0;
        d = (t1 + (((((e >>> 2) | (e << 30)) ^ ((e >>> 13) | (e << 19)) ^ ((e >>> 22) | (e << 10)))) + ((e & f) ^ (e & g) ^ (f & g)))) | 0; }
      { const t1 = (c + ((((h >>> 6) | (h << 26)) ^ ((h >>> 11) | (h << 21)) ^ ((h >>> 25) | (h << 7)))) + ((h & a) ^ (~h & b)) + K[i + 5]! + W[i + 5]!) | 0;
        g = (g + t1) | 0;
        c = (t1 + (((((d >>> 2) | (d << 30)) ^ ((d >>> 13) | (d << 19)) ^ ((d >>> 22) | (d << 10)))) + ((d & e) ^ (d & f) ^ (e & f)))) | 0; }
      { const t1 = (b + ((((g >>> 6) | (g << 26)) ^ ((g >>> 11) | (g << 21)) ^ ((g >>> 25) | (g << 7)))) + ((g & h) ^ (~g & a)) + K[i + 6]! + W[i + 6]!) | 0;
        f = (f + t1) | 0;
        b = (t1 + (((((c >>> 2) | (c << 30)) ^ ((c >>> 13) | (c << 19)) ^ ((c >>> 22) | (c << 10)))) + ((c & d) ^ (c & e) ^ (d & e)))) | 0; }
      { const t1 = (a + ((((f >>> 6) | (f << 26)) ^ ((f >>> 11) | (f << 21)) ^ ((f >>> 25) | (f << 7)))) + ((f & g) ^ (~f & h)) + K[i + 7]! + W[i + 7]!) | 0;
        e = (e + t1) | 0;
        a = (t1 + (((((b >>> 2) | (b << 30)) ^ ((b >>> 13) | (b << 19)) ^ ((b >>> 22) | (b << 10)))) + ((b & c) ^ (b & d) ^ (c & d)))) | 0; }
    }
    h0 = (h0 + a) | 0; h1 = (h1 + b) | 0; h2 = (h2 + c) | 0; h3 = (h3 + d) | 0;
    h4 = (h4 + e) | 0; h5 = (h5 + f) | 0; h6 = (h6 + g) | 0; h7 = (h7 + h) | 0;
  }
  H[0] = h0; H[1] = h1; H[2] = h2; H[3] = h3; H[4] = h4; H[5] = h5; H[6] = h6; H[7] = h7;
}

export function sha256Hex(input: string | Uint8Array): string {
  const bytes = typeof input === "string" ? textEncoder.encode(input) : input;
  H[0] = 0x6a09e667; H[1] = 0xbb67ae85 | 0; H[2] = 0x3c6ef372; H[3] = 0xa54ff53a | 0;
  H[4] = 0x510e527f; H[5] = 0x9b05688c | 0; H[6] = 0x1f83d9ab; H[7] = 0x5be0cd19;
  // Whole blocks are hashed straight from the input; only the tail (< 64 bytes plus padding) is copied.
  const fullBlocks = Math.floor(bytes.length / 64);
  if (fullBlocks > 0) compress(bytes, 0, fullBlocks);
  const remainder = bytes.length - fullBlocks * 64;
  const tail = new Uint8Array(remainder + 9 > 64 ? 128 : 64);
  tail.set(bytes.subarray(fullBlocks * 64));
  tail[remainder] = 0x80;
  const bitLength = bytes.length * 8;
  const view = new DataView(tail.buffer);
  view.setUint32(tail.length - 8, Math.floor(bitLength / 0x1_0000_0000));
  view.setUint32(tail.length - 4, bitLength >>> 0);
  compress(tail, 0, tail.length / 64);
  let hex = "";
  for (let i = 0; i < 8; i += 1) {
    const word = H[i]!;
    hex += HEX[(word >>> 24) & 0xff]! + HEX[(word >>> 16) & 0xff]! + HEX[(word >>> 8) & 0xff]! + HEX[word & 0xff]!;
  }
  return hex;
}
