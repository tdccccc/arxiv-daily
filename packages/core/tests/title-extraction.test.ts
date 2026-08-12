import { describe, expect, it } from "vitest";
import type { PdfLayoutLine } from "../src/library/fulltext/ports";
import { extractTitleFromFirstPage } from "../src/library/fulltext/title-extraction";

/** Build a typographic layout from [text, fontSize, topFraction] tuples. */
function layout(
  lines: ReadonlyArray<readonly [string, number, number]>,
): readonly PdfLayoutLine[] {
  return lines.map(([text, fontSize, topFraction]) => ({ text, fontSize, topFraction }));
}

describe("extractTitleFromFirstPage (text fallback)", () => {
  it("extracts the first plausible title line", () => {
    expect(extractTitleFromFirstPage([
      "Attention Is All You Need\nAshish Vaswani\nAbstract body.",
    ])).toBe("Attention Is All You Need");
  });

  it("skips arXiv and reprint headers before the title", () => {
    expect(extractTitleFromFirstPage([
      "arXiv:1706.03762v3 [cs.CL] 2 Aug 2023\nAttention Is All You Need\nAbstract body.",
    ])).toBe("Attention Is All You Need");
    expect(extractTitleFromFirstPage([
      "Submitted to the Astrophysical Journal\nPublished as a conference paper at ICLR 2015\nVery Deep Convolutional Networks\nAbstract body.",
    ])).toBe("Very Deep Convolutional Networks");
    expect(extractTitleFromFirstPage([
      "Draft version January 31, 2019\n"
        + "Preprint typeset using LATEX style emulateapj v. 12/16/11\n"
        + "THE PAN-STARRS1 SURVEYS\n"
        + "K. C. Chambers, E. A. Magnier, N. Metcalfe",
    ])).toBe("THE PAN-STARRS1 SURVEYS");
  });

  it("skips MNRAS Advance Access publication banners before the title", () => {
    expect(extractTitleFromFirstPage([
      "Advance Access publication 2021 January 21\n"
        + "Photometric redshifts for galaxy clusters with neural networks\n"
        + "A. Author, B. Collaborator\n"
        + "Abstract body.",
    ])).toBe("Photometric redshifts for galaxy clusters with neural networks");
    expect(extractTitleFromFirstPage([
      "Advance Access publication 2013 May 1\n"
        + "Galaxy cluster mass estimation from weak lensing\n"
        + "C. Kind, D. Collaborator",
    ])).toBe("Galaxy cluster mass estimation from weak lensing");
    expect(extractTitleFromFirstPage([
      "Advance Access publication 2020 August 31\n"
        + "Neural network photometric redshifts for DESI and Pan-STARRS\n"
        + "E. Beck",
    ])).toBe("Neural network photometric redshifts for DESI and Pan-STARRS");
  });

  it("skips permission-notice lines and lowercase continuations", () => {
    expect(extractTitleFromFirstPage([
      "Provided proper attribution is provided, Google hereby grants permission to\n"
        + "reproduce the tables and figures in this paper solely for use in journalistic\n"
        + "or scholarly works.\nAttention Is All You Need\nAshish Vaswani",
    ])).toBe("Attention Is All You Need");
  });

  it("cuts author lists sharing the title line at author markers", () => {
    expect(extractTitleFromFirstPage([
      "Language Models are Few-Shot Learners Tom B. Brown ∗ Benjamin Mann",
    ])).toBe("Language Models are Few-Shot Learners Tom B. Brown");
    expect(extractTitleFromFirstPage([
      "Deep Residual Learning for Image Recognition Kaiming He @microsoft.com",
    ])).toBe("Deep Residual Learning for Image Recognition Kaiming He");
    expect(extractTitleFromFirstPage([
      "Maxout Networks Ian J. Goodfellow Abstract Deep learning",
    ])).toBe("Maxout Networks Ian J. Goodfellow");
  });

  it("returns null for empty, header-only, or short pages", () => {
    expect(extractTitleFromFirstPage([])).toBeNull();
    expect(extractTitleFromFirstPage([""])).toBeNull();
    expect(extractTitleFromFirstPage(["arXiv:1706.03762v3 [cs.CL] 2 Aug 2023"])).toBeNull();
    expect(extractTitleFromFirstPage(["Short"])).toBeNull();
  });
});

describe("extractTitleFromFirstPage (typographic layout)", () => {
  // Real first-page layouts of the four MNRAS PDFs reported with
  // "Advance Access publication …" banner titles (journal page header, not
  // the paper title). Font sizes and positions are the pdf.js measurements.
  const mucesh: readonly PdfLayoutLine[] = layout([
    ["MNRAS 502, 2770–2786 (2021) doi:10.1093/mnras/stab164", 7.97, 0.049],
    ["Advance Access publication 2021 January 21", 8.97, 0.061],
    ["A machine learning approach to galaxy properties: joint redshift–stellar", 15.94, 0.106],
    ["mass probability distributions with Random Forest", 15.94, 0.128],
    ["S. Mucesh ,", 11.96, 0.167],
    ["W. G. Hartley,", 11.96, 0.167],
  ]);
  const carrasco: readonly PdfLayoutLine[] = layout([
    ["MNRAS 432, 1483–1501 (2013) doi:10.1093/mnras/stt574", 7.97, 0.049],
    ["Advance Access publication 2013 May 1", 8.97, 0.061],
    ["TPZ: photometric redshift PDFs and ancillary information by using", 15.94, 0.106],
    ["prediction trees and random forests", 15.94, 0.13],
    ["Matias Carrasco Kind", 14.35, 0.172],
    ["and Robert J. Brunner", 14.35, 0.172],
  ]);
  const beck: readonly PdfLayoutLine[] = layout([
    ["MNRAS 500, 1633–1644 (2021) doi:10.1093/mnras/staa2587", 7.97, 0.049],
    ["Advance Access publication 2020 August 31", 8.97, 0.061],
    ["PS1-STRM: neural network source classification and photometric redshift", 15.94, 0.106],
    ["catalogue for PS1 3π DR1", 15.94, 0.128],
    ["R´obert Beck,", 11.96, 0.167],
    ["Istv´an Szapudi,", 11.96, 0.167],
  ]);
  const luo: readonly PdfLayoutLine[] = layout([
    ["MNRAS 535, 1844–1855 (2024) https://doi.org/10.1093/mnras/stae2446", 7.97, 0.049],
    ["Advance Access publication 2024 October 26", 8.97, 0.062],
    ["Photometric redshift estimation for CSST survey with LSTM neural", 15.94, 0.106],
    ["networks", 15.94, 0.128],
    ["Zhijian Luo ,", 11.96, 0.167],
    ["Yicheng Li,", 11.96, 0.167],
  ]);

  it("selects the largest-font title over the MNRAS page banner", () => {
    expect(extractTitleFromFirstPage([""], [mucesh]))
      .toBe("A machine learning approach to galaxy properties: joint redshift–stellar "
        + "mass probability distributions with Random Forest");
    expect(extractTitleFromFirstPage([""], [carrasco]))
      .toBe("TPZ: photometric redshift PDFs and ancillary information by using "
        + "prediction trees and random forests");
    expect(extractTitleFromFirstPage([""], [beck]))
      .toBe("PS1-STRM: neural network source classification and photometric redshift "
        + "catalogue for PS1 3π DR1");
    expect(extractTitleFromFirstPage([""], [luo]))
      .toBe("Photometric redshift estimation for CSST survey with LSTM neural networks");
  });

  it("skips a journal masthead inside the top strip when a nearly-as-large band follows", () => {
    // A&A "Astronomy & Astrophysics" logo at 17.04; the title at 16.35.
    const lines = layout([
      ["A&A 517, A92 (2010)", 9.96, 0.07],
      ["Astronomy", 17.04, 0.089],
      ["&", 15.14, 0.101],
      ["Astrophysics", 17.04, 0.113],
      ["The universal galaxy cluster pressure profile from a representative", 16.35, 0.172],
      ["sample of nearby systems (REXCESS) and the", 16.35, 0.196],
      ["M. Arnaud", 13.95, 0.23],
    ]);
    expect(extractTitleFromFirstPage([""], [lines]))
      .toBe("The universal galaxy cluster pressure profile from a representative "
        + "sample of nearby systems (REXCESS) and the");
  });

  it("skips an arXiv stamp at the largest font", () => {
    const lines = layout([
      ["Weighing The Giants - III. Methods and Measurements of Accurate", 17.22, 0.118],
      ["Galaxy Cluster Weak-Lensing Masses", 17.22, 0.143],
      ["Douglas E. Applegate", 14.35, 0.193],
      ["arXiv:1208.0605v2 [astro-ph.CO] 18 Apr 2014", 20.0, 0.711],
    ]);
    expect(extractTitleFromFirstPage([""], [lines]))
      .toBe("Weighing The Giants - III. Methods and Measurements of Accurate "
        + "Galaxy Cluster Weak-Lensing Masses");
  });

  it("skips running heads above the page box", () => {
    const lines = layout([
      ["BIOINFORMATICS ORIGINAL PAPER", 18.93, -0.066],
      ["Permutation importance: a corrected feature importance measure", 15.94, 0.005],
      ["André Altmann", 12.95, 0.03],
    ]);
    expect(extractTitleFromFirstPage([""], [lines]))
      .toBe("Permutation importance: a corrected feature importance measure");
  });

  it("rejects section headings bigger than the title and falls to the next band", () => {
    const lines = layout([
      ["1. INTRODUCTION", 10.96, 0.553],
      ["BAYESIAN PHOTOMETRIC REDSHIFT ESTIMATION", 9.96, 0.116],
      ["NARCISO BENIç TEZ", 9.96, 0.136],
      ["Astronomy Department, University of California at Berkeley, 601 Campbell Hall, Berkeley, CA 94720-5030 ; benitezn=mars.berkeley.edu", 7.97, 0.156],
      ["ABSTRACT", 9.96, 0.186],
    ]);
    // Old scans set title and author in one face; the author suffix is a
    // documented limitation, the section heading must never win.
    expect(extractTitleFromFirstPage([""], [lines]))
      .toBe("BAYESIAN PHOTOMETRIC REDSHIFT ESTIMATION NARCISO BENIç TEZ");
  });

  it("breaks a same-font run at an author line", () => {
    const lines = layout([
      ["SAMPLE VARIANCE CONSIDERATIONS FOR CLUSTER SURVEYS", 9.96, 0.118],
      ["Wayne Hu", 9.96, 0.138],
      ["and Andrey V. Kravtsov", 9.96, 0.138],
      ["Received 2002 March 11; accepted 2002 October 31", 7.97, 0.151],
      ["ABSTRACT", 9.96, 0.178],
    ]);
    expect(extractTitleFromFirstPage([""], [lines]))
      .toBe("SAMPLE VARIANCE CONSIDERATIONS FOR CLUSTER SURVEYS");
  });

  it("extends the title with roman-numeral and lowercase continuation lines", () => {
    const euclid = layout([
      ["Astronomy", 17.04, 0.089],
      ["&", 15.14, 0.101],
      ["Astrophysics", 17.04, 0.113],
      ["Euclid preparation", 16.35, 0.176],
      ["VII. Forecast validation for Euclid cosmological probes", 13.63, 0.206],
      ["Euclid Collaboration", 10.91, 0.234],
    ]);
    expect(extractTitleFromFirstPage([""], [euclid]))
      .toBe("Euclid preparation VII. Forecast validation for Euclid cosmological probes");
    const erostia = layout([
      ["The SRG/eROSITA All-Sky Survey", 16.35, 0.148],
      ["Optical identification and properties of galaxy clusters and groups in the", 13.63, 0.179],
      ["western galactic hemisphere", 13.63, 0.198],
      ["M. Kluge", 10.91, 0.225],
    ]);
    expect(extractTitleFromFirstPage([""], [erostia]))
      .toBe("The SRG/eROSITA All-Sky Survey Optical identification and properties "
        + "of galaxy clusters and groups in the western galactic hemisphere");
    const quasars = layout([
      ["Discovery of intergalactic bridges connecting two faint z ∼ 3", 17.22, 0.148],
      ["quasars", 16.35, 0.172],
      ["Fabrizio Arrigoni Battaia", 10.91, 0.199],
    ]);
    expect(extractTitleFromFirstPage([""], [quasars]))
      .toBe("Discovery of intergalactic bridges connecting two faint z ∼ 3 quasars");
  });

  it("joins subscript fragments into the title run", () => {
    const lines = layout([
      ["Cosmological Constraints on Ω", 13.95, 0.11],
      ["m", 9.3, 0.108],
      ["and σ", 13.95, 0.11],
      ["8", 9.3, 0.108],
      ["from Cluster Abundances Using the GalWCat19", 13.95, 0.11],
      ["Optical-spectroscopic SDSS Catalog", 13.95, 0.129],
      ["Mohamed H. Abdullah", 9.96, 0.155],
    ]);
    expect(extractTitleFromFirstPage([""], [lines]))
      .toBe("Cosmological Constraints on Ω and σ from Cluster Abundances Using "
        + "the GalWCat19 Optical-spectroscopic SDSS Catalog");
  });

  it("accepts lowercase proper-noun title starts", () => {
    const redmapper = layout([
      ["redMaPPer – III. A detailed comparison of the Planck 2013", 15.94, 0.106],
      ["and SDSS DR8 redMaPPer cluster catalogues", 15.94, 0.131],
      ["E. Rozo,", 14.35, 0.173],
    ]);
    expect(extractTitleFromFirstPage([""], [redmapper]))
      .toBe("redMaPPer – III. A detailed comparison of the Planck 2013 "
        + "and SDSS DR8 redMaPPer cluster catalogues");
    const dustmaps = layout([
      ["dustmaps: A Python interface for maps of interstellar", 17.22, 0.179],
      ["dust", 17.22, 0.205],
      ["Gregory M. Green", 11.96, 0.24],
    ]);
    expect(extractTitleFromFirstPage([""], [dustmaps]))
      .toBe("dustmaps: A Python interface for maps of interstellar dust");
  });

  it("excludes preprint report numbers and DOIs per line", () => {
    const lines = layout([
      ["DES 2015-0146", 9.96, 0.035],
      ["SLAC PUB-16454", 9.96, 0.048],
      ["Fermilab PUB-16-012-E-PPD", 9.96, 0.061],
      ["DOI 10.3847/0067-0049/224/1/1", 9.96, 0.075],
      ["Draft version May 27, 2016", 7.97, 0.067],
      ["THE REDMAPPER GALAXY CLUSTER CATALOG FROM DES SCIENCE VERIFICATION DATA", 9.96, 0.137],
      ["E. S. Rykoff", 8.97, 0.157],
    ]);
    expect(extractTitleFromFirstPage([""], [lines]))
      .toBe("THE REDMAPPER GALAXY CLUSTER CATALOG FROM DES SCIENCE VERIFICATION DATA");
  });

  it("skips the max band when it is a short fragment and names would win", () => {
    // The last title word renders at a larger font than the rest of the title;
    // the author band would otherwise be chosen.
    const lines = layout([
      ["COSMOLIKE – cosmological likelihood analyses for photometric galaxy", 12.75, 0.106],
      ["surveys", 15.94, 0.13],
      ["Elisabeth Krause", 14.35, 0.172],
      ["and Tim Eifler", 14.35, 0.172],
    ]);
    expect(extractTitleFromFirstPage([""], [lines]))
      .toBe("COSMOLIKE – cosmological likelihood analyses for photometric galaxy surveys");
  });

  it("falls back to the text heuristic when the layout has no measurable fonts", () => {
    const lines = layout([
      ["Advance Access publication 2021 January 21", 0, 0],
      ["Photometric redshifts for galaxy clusters with neural networks", 0, 0],
    ]);
    expect(extractTitleFromFirstPage([], [lines])).toBeNull();
    expect(extractTitleFromFirstPage([
      "Advance Access publication 2021 January 21\n"
        + "Photometric redshifts for galaxy clusters with neural networks",
    ], [lines])).toBe("Photometric redshifts for galaxy clusters with neural networks");
  });
});

describe("extractTitleFromFirstPage (document metadata)", () => {
  const bannerPage = layout([
    ["Advance Access publication 2021 January 21", 8.97, 0.061],
    ["Photometric redshifts for galaxy clusters with neural networks", 15.94, 0.106],
  ]);

  it("prefers a usable metadata title over typographic extraction", () => {
    expect(extractTitleFromFirstPage(
      [""],
      [bannerPage],
      "Photometric redshifts for galaxy clusters with neural networks",
    )).toBe("Photometric redshifts for galaxy clusters with neural networks");
  });

  it("decodes the HTML entities pdf.js leaves in metadata", () => {
    expect(extractTitleFromFirstPage(
      [],
      undefined,
      "WISE-PS1-STRM: source classification for WISE&#x00D7;PS1 &ndash; redshifts",
    )).toBe("WISE-PS1-STRM: source classification for WISE×PS1 – redshifts");
  });

  it("rejects garbage metadata (paths, file names, page references, arXiv stamps, LaTeX)", () => {
    const garbage = [
      "C:\\Documents and Settings\\senthil.d\\Desktop\\btq134.dvi",
      "H:\\PS\\UCJ6EQ.PS",
      "log_jack_full.eps",
      "55682 702..715",
      "TLA_large",
      "arXiv:0910.5735v1  [astro-ph.CO]  30 Oct 2009",
      "Photo-$z$ Estimation with Normalizing Flow",
      "Microsoft Word - The Qitai radio telescope-online.doc",
    ];
    for (const title of garbage) {
      expect(extractTitleFromFirstPage([""], [bannerPage], title))
        .toBe("Photometric redshifts for galaxy clusters with neural networks");
    }
  });

  it("prefers a longer typographic title whose tokens cover the metadata", () => {
    // The metadata dropped the "z"; the typographic title covers it and is
    // longer, so the typographic title wins.
    const page = layout([
      ["Clusters of galaxies up to z = 1.5 identified from photometric data of", 15.94, 0.106],
      ["the Dark Energy Survey and unWISE", 15.94, 0.128],
    ]);
    expect(extractTitleFromFirstPage(
      [""],
      [page],
      "Clusters of galaxies up to = 1.5 identified from photometric data of the Dark Energy Survey",
    )).toBe("Clusters of galaxies up to z = 1.5 identified from photometric data of "
      + "the Dark Energy Survey and unWISE");
  });

  it("prefers metadata that fixes subscripts and series titles", () => {
    // Metadata restores the subscripts the text layer loses.
    const subscripts = layout([
      ["Cosmological Constraints on Ω and σ", 13.95, 0.11],
      ["from Cluster Abundances Using the GalWCat19", 13.95, 0.11],
      ["Optical-spectroscopic SDSS Catalog", 13.95, 0.129],
    ]);
    expect(extractTitleFromFirstPage(
      [""],
      [subscripts],
      "Cosmological Constraints on Ωm and σ8 from Cluster Abundances Using "
        + "the GalWCat19 Optical-spectroscopic SDSS Catalog",
    )).toBe("Cosmological Constraints on Ωm and σ8 from Cluster Abundances Using "
      + "the GalWCat19 Optical-spectroscopic SDSS Catalog");
  });
});

  it("does not extend the title with comma-separated author pairs", () => {
    // arXiv-style author block in a different band: the extension must not
    // attach it as a long phrase continuation.
    const lines = layout([
      ["A Survey of Large Language Models", 24, 0.094],
      ["Wayne Xin Zhao, Kun Zhou*, Junyi Li*, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min", 11, 0.13],
      ["Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen,", 11, 0.147],
    ]);
    expect(extractTitleFromFirstPage([""], [lines])).toBe("A Survey of Large Language Models");
  });

  it("keeps the metadata when the extended typographic title only adds author lines", () => {
    // Krause2017_1-style: the base title matches the metadata tokens, but the
    // continuation extension attached the author line; the metadata must win.
    const lines = layout([
      ["COSMOLIKE – cosmological likelihood analyses for photometric galaxy", 12.75, 0.106],
      ["surveys", 15.94, 0.13],
      ["Elisabeth Krause", 14.35, 0.172],
      ["and Tim Eifler", 14.35, 0.172],
    ]);
    expect(extractTitleFromFirstPage(
      [""],
      [lines],
      "cosmolike &ndash; cosmological likelihood analyses for photometric galaxy surveys",
    )).toBe("cosmolike – cosmological likelihood analyses for photometric galaxy surveys");
  });

  it("breaks the extension at Obsidian-style author lines with affiliation markers", () => {
    // Obsidian's pdf.js extracts "R´obert Beck" as "R' obert Beck" and merges
    // affiliation numbers; the marker pattern must still break the run.
    const lines = layout([
      ["PS1-STRM: neural network source classification and photometric", 15.94, 0.106],
      ["redshift catalogue for PS1 3 n DR1", 15.94, 0.128],
      ["R' obert Beck, 1,2 < lstv\"anSzapudi, 1,2<Heather Flewelling, 1 Conrad Holmberg, 1,3 Eugene Magnier 1 and Kenneth C. Chambers", 8.97, 0.167],
    ]);
    expect(extractTitleFromFirstPage([""], [lines]))
      .toBe("PS1-STRM: neural network source classification and photometric "
        + "redshift catalogue for PS1 3 n DR1");
  });
