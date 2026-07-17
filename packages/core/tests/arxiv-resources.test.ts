import { describe, expect, it } from "vitest";
import { modernArxivResources } from "../src/utils/arxiv";

const expected = {
  id: "2606.12345",
  absUrl: "https://arxiv.org/abs/2606.12345",
  pdfUrl: "https://arxiv.org/pdf/2606.12345",
  htmlUrl: "https://arxiv.org/html/2606.12345",
  sourceUrl: "https://arxiv.org/e-print/2606.12345",
  atomUrl: "https://export.arxiv.org/api/query?id_list=2606.12345&max_results=1",
};

describe("modernArxivResources", () => {
  it.each([
    "2606.12345",
    "2606.12345v2",
    "arXiv:2606.12345",
    "https://arxiv.org/pdf/2606.12345v2?download=1",
    "https://arxiv.org/pdf/2606.12345.pdf",
    "http://arxiv.org/abs/2606.12345",
    "https://www.arxiv.org/abs/2606.12345#section",
    "https://arxiv.org/html/2606.12345",
    "https://arxiv.org/e-print/2606.12345",
  ])("canonicalizes modern ID input %j and derives trusted URLs", (input) => {
    expect(modernArxivResources(input)).toEqual(expected);
  });

  it.each([
    "",
    "hep-th/9901001",
    "2606.12345/../../x",
    "https://user@arxiv.org/abs/2606.12345",
    "https://arxiv.org:444/abs/2606.12345",
    "https://evil.test/abs/2606.12345",
    "https://evil.arxiv.org/abs/2606.12345",
    "https://export.arxiv.org/abs/2606.12345",
    "https://arxiv.org//abs/2606.12345",
    "https://arxiv.org/abs//2606.12345",
    "https://arxiv.org/%61bs/2606.12345",
    "https://arxiv.org/abs/2606%2e12345",
    "https://arxiv.org/abs/2606.12345/extra",
    "https://arxiv.org/abs/2606.12345.pdf",
    "https://arxiv.org/api/query?id_list=2606.12345",
  ])("rejects non-modern or untrusted input %j", (value) => {
    expect(modernArxivResources(value)).toBeNull();
  });
});
