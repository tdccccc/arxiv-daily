# PdfJsDocumentParser 对大写 UTF-16LE BOM（FFFE）按大端误解码成乱码，且标题抽取未校验 HTML 数字实体码点范围

状态：已修复（分支 `fix/2026-11-22-review`）

> 来源：2026-11-22 main `2557b90a` 代码审查（PDF 文档解析与标题抽取）
> 严重程度：低 · 置信度：已复现（离线实测：`/Title <FFFE48006900>` 被误解码为 `"䠀椀"` `U+4800 U+6900`）

## 问题与依据

PDF 元数据兜底解析与标题抽取存在两处边界缺陷：

1. **大写 `FFFE` 十六进制 `/Title` 的字节序判定大小写敏感，导致 UTF-16LE 标题全部乱码**：
   - `packages/core/src/documents/pdfjs-document-parser.ts:260-279`（`rawInfoTitle`）：
     ```ts
     const content = hex[1]!;
     const utf16 = /^(?:feff|fffe)/i.test(content);
     ...
     if (utf16) {
       const littleEndian = content.startsWith("fffe");
       const codeUnits: number[] = [];
       for (let index = 2; index + 1 < out.length; index += 2) {
         codeUnits.push(littleEndian
           ? out[index]! | (out[index + 1]! << 8)
           : (out[index]! << 8) | out[index + 1]!);
       }
       return String.fromCharCode(...codeUnits);
     }
     return String.fromCharCode(...out);
     ```
   - 第 263 行用不区分大小写的 `/^(?:feff|fffe)/i` 识别出 UTF-16 BOM，但第 269 行 `content.startsWith("fffe")` 却是**区分大小写**的！
   - PDF 规范中十六进制字符串经常输出大写（如 `<FFFE48006900>` 表示 `"Hi"`）。此时 `utf16` 为 `true`，`littleEndian` 为 `false`，代码错误地按大端解码把 `"Hi"` 解成了 `U+4800 U+6900`（`䠀椀`）。
   - 此外第 276/278 行对未限定上限的十六进制匹配 `([0-9A-Fa-f]{2,})` 直接使用 `String.fromCharCode(...out)` 展开传参，若异常 PDF 中 `/Title <...>` 超长（数万字节），会触发 `RangeError: Maximum call stack size exceeded`。
2. **`title-extraction.ts` 解码数字 HTML 实体未校验 Unicode 码点上限**：
   - `packages/core/src/library/fulltext/title-extraction.ts:187-188` 直接对正则捕获结果调用 `String.fromCodePoint(parseInt(...))`；当 PDF 文本或元数据包含越界数字实体（如 `&#99999999;` 或 `&#x110000;` 或 `&#0;` 以外超出 `0x10FFFF` 的值）时，`String.fromCodePoint()` 会抛出 `RangeError: Invalid code point`，导致整篇 PDF 的标题抽取失败。

## 原因

1. 第 269 行忘记对 `content` 做 `.toLowerCase()`（或用 `/^fffe/i.test(content)`），且未像字面量分支（限制 `{1,400}`）那样限制十六进制 `/Title` 长度或使用 `TextDecoder`。
2. `String.fromCodePoint` 调用前缺少 `code >= 0 && code <= 0x10ffff` 边界保护。

## 建议和验收

- `rawInfoTitle`（`pdfjs-document-parser.ts:260-279`）：
  - 限制十六进制 `/Title` 匹配长度上限（如 `{2,2048}`）或改用 `TextDecoder("utf-16le" / "utf-16be")` / 分块转换避免调用栈溢出；
  - 将 `const littleEndian = content.startsWith("fffe");` 改为 `const littleEndian = /^fffe/i.test(content);`。
- `title-extraction.ts:187-188`：调用 `String.fromCodePoint` 前校验 `Number.isInteger(code) && code >= 0 && code <= 0x10ffff`（且跳过非法代理项），越界时保留原实体文本。
- 验收：输入含 `/Title <FFFE48006900>` 的 PDF 正确解析出 `"Hi"`；超长十六进制 `/Title` 与越界 HTML 数字实体（如 `&#99999999;`）均不抛 `RangeError`。

## 修复后应补的测试

- `packages/core/tests/pdfjs-document-parser.test.ts`：
  - 构造 `/Title <FFFE48006900>`（大写 BOM）与 `/Title <fffe48006900>`（小写 BOM），断言均解出 `"Hi"`；
  - 构造超长十六进制 `/Title`（如 100 KB 十六进制串），断言不抛 `RangeError`。
- `packages/core/tests/title-extraction.test.ts`：包含 `&#99999999;`、`&#x110000;` 的标题文本安全解码不抛异常。

## 修复情况

`rawInfoTitle` 用 `/^fffe/i` 判定小端，十六进制串按块转字符串避免超长 `/Title` 触发调用栈溢出；`title-extraction` 在 `String.fromCodePoint` 前校验码点范围与代理项，越界/非法实体保留原文本。测试：`pdfjs-document-parser` + `title-extraction` 33 项通过（大写/小写 BOM、100 KB 十六进制、越界实体）。
