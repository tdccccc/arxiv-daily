import { describe, expect, it } from "vitest";
import {
  extractAbstractFromPages,
  MAX_ABSTRACT_CHARS,
} from "../src/library/fulltext/abstract-extraction";

/**
 * Fixtures are real first pages from the frozen baseline corpus
 * (`baseline_library_207`), trimmed to the leading region that decides
 * extraction. Layout quirks are preserved verbatim — ligature damage,
 * superscript affiliation digits on their own lines, and all.
 */

/** A&A: `ABSTRACT` alone on a line, body follows, ends at `Key words:`. */
const AANDA_UPPERCASE_MARKER = `Astronomy
&
Astrophysics

A&A 517, A92 (2010)
DOI: 10.1051/0004-6361/200913416
c ESO 2010


The universal galaxy cluster pressure profile from a representative
sample of nearby systems (REXCESS) and the Y SZ – M 500 relation
M. Arnaud1 , G. W. Pratt1,2 , R. Piﬀaretti1 , H. Böhringer2 , J. H. Croston3 , and E. Pointecouteau4
1

Laboratoire AIM, IRFU/Service d’Astrophysique – CEA/DSM – CNRS – Université Paris Diderot, Bât. 709, CEA-Saclay,
91191 Gif-sur-Yvette Cedex, France
e-mail: Monique.Arnaud@cea.fr
Received 7 October 2009 / Accepted 10 April 2010
ABSTRACT

We investigate the regularity of cluster pressure profiles with REXCESS, a representative sample of 33 local (z < 0.2) clusters drawn
from the REFLEX catalogue and observed with XMM-Newton. The sample spans a mass range of 1014 M < M500 < 1015 M ,
where M500 is the mass corresponding to a density contrast of 500. We derive an average profile from observations scaled by mass
and redshift according to the standard self-similar model, and find that the dispersion about the mean is remarkably low.

Key words: galaxies: clusters: general – cosmology: observations

1. Introduction
The abundance of galaxy clusters is a sensitive probe of cosmology.`;

/** Kluwer: `Abstract.` starts the body inline; ends at `Keywords:`. */
const KLUWER_INLINE_MARKER = `Machine Learning, 45, 5–32, 2001
c 2001 Kluwer Academic Publishers. Manufactured in The Netherlands.


Random Forests
LEO BREIMAN
Statistics Department, University of California, Berkeley, CA 94720
Editor: Robert E. Schapire

Abstract. Random forests are a combination of tree predictors such that each tree depends on the values of a
random vector sampled independently and with the same distribution for all trees in the forest. The generalization
error for forests converges a.s. to a limit as the number of trees in the forest becomes large.
Keywords: classification, regression, ensemble

1.

Random forests`;

/** REVTeX: no `Abstract` word at all — the body follows `(Dated: …)`. */
const REVTEX_NO_MARKER = `Forecast constraints on neutrino mass from CSST galaxy clusters
Mingjing Chen,∗ Yufei Zhang,† and Wenjuan Fang‡
CAS Key Laboratory for Research in Galaxies and Cosmology,
Department of Astronomy, University of Science and Technology of China,
Hefei, Anhui, 230026, People’s Republic of China and
School of Astronomy and Space Science, University of Science and Technology of China,
Hefei, Anhui, 230026, People’s Republic of China

arXiv:2411.02752v1 [astro-ph.CO] 5 Nov 2024

Weiguang Cui¶
Departamento de Fı́sica Teórica, Universidad Autónoma de Madrid, Módulo 15, E-28049 Madrid, Spain
(Dated: November 6, 2024)
With the advent of next-generation surveys, constraints on cosmological parameters are anticipated to become more stringent, particularly for the total neutrino mass. This study forecasts these
constraints utilizing galaxy clusters from the Chinese Space Station Telescope (CSST). Employing
Fisher matrix analysis, we derive the constraint σ(Mν ) from cluster number counts, cluster power

I. INTRODUCTION

The standard cosmological model has been remarkably successful.`;

/** ApJ with a 50-author list: page 1 holds no abstract at all. */
const LONG_AUTHOR_LIST_PAGE_1 = `The Astrophysical Journal, 878:55 (25pp), 2019 June 10

https://doi.org/10.3847/1538-4357/ab1f10

© 2019. The American Astronomical Society. All rights reserved.

Cluster Cosmology Constraints from the 2500 deg2 SPT-SZ Survey: Inclusion of Weak
Gravitational Lensing Data from Magellan and the Hubble Space Telescope
S. Bocquet1,2,3 , J. P. Dietrich1,4 , T. Schrabback5, L. E. Bleem2,3, M. Klein1,6, S. W. Allen7,8,9,
M. L. N. Ashby10 , M. Bautz11, M. Bayliss11 , B. A. Benson3,12,13, M. Brodwin14 , E. Bulbul10,
R. Capasso1,4, J. E. Carlstrom2,3,12,15,16, C. L. Chang2,3,12, I. Chiu17, H-M. Cho18, A. Clocchiatti19,
1

Faculty of Physics, Ludwig-Maximilians-Universität, Scheinerstr. 1, D-81679 Munich, Germany
2`;

const LONG_AUTHOR_LIST_PAGE_2 = `Abstract
We present cosmological constraints from a galaxy cluster sample of 343 clusters, combining
X-ray, SZ, and weak-lensing data. We find that the cluster sample constrains the amplitude of
matter fluctuations to better than 5 per cent.

1. Introduction`;

/** Scanned page: a bibcode watermark and nothing else (4.2% of the corpus). */
const SCANNED_NO_TEXT_LAYER = `1965ARA&A...3....1A`;

describe("extractAbstractFromPages", () => {
  it("takes the body after an uppercase ABSTRACT marker on its own line", () => {
    const result = extractAbstractFromPages([AANDA_UPPERCASE_MARKER]);
    expect(result.route).toBe("marker");
    expect(result.abstract).toMatch(/^We investigate the regularity of cluster pressure profiles/);
    // The marker word itself is structure, not content.
    expect(result.abstract).not.toMatch(/ABSTRACT/);
    // Stops at the keyword line rather than running into the body.
    expect(result.abstract).not.toMatch(/Key words/);
    expect(result.abstract).not.toMatch(/1\. Introduction/);
    // Author and affiliation lines above the marker are excluded.
    expect(result.abstract).not.toMatch(/Arnaud/);
    expect(result.abstract).not.toMatch(/e-mail/);
  });

  it("takes the body after an inline `Abstract.` marker", () => {
    const result = extractAbstractFromPages([KLUWER_INLINE_MARKER]);
    expect(result.route).toBe("marker");
    expect(result.abstract).toMatch(/^Random forests are a combination of tree predictors/);
    expect(result.abstract).not.toMatch(/^Abstract/);
    expect(result.abstract).not.toMatch(/Keywords/);
    expect(result.abstract).not.toMatch(/LEO BREIMAN/);
  });

  it("takes the body after `(Dated: …)` when no Abstract marker exists", () => {
    const result = extractAbstractFromPages([REVTEX_NO_MARKER]);
    expect(result.route).toBe("dated");
    expect(result.abstract).toMatch(/^With the advent of next-generation surveys/);
    expect(result.abstract).not.toMatch(/Dated:/);
    expect(result.abstract).not.toMatch(/INTRODUCTION/);
    // The arXiv stamp and affiliation block sit above the abstract.
    expect(result.abstract).not.toMatch(/arXiv:2411/);
    expect(result.abstract).not.toMatch(/Madrid/);
  });

  it("finds an abstract pushed onto page 2 by a long author list", () => {
    const result = extractAbstractFromPages([LONG_AUTHOR_LIST_PAGE_1, LONG_AUTHOR_LIST_PAGE_2]);
    expect(result.route).toBe("marker");
    expect(result.abstract).toMatch(/^We present cosmological constraints/);
    expect(result.abstract).not.toMatch(/Bocquet/);
    // Stops at the section heading — caught by mutation check: without this
    // the page-2 route passed even with the terminator disabled.
    expect(result.abstract).not.toMatch(/Introduction/);
  });

  it("reports no usable text for a scanned page with no text layer", () => {
    const result = extractAbstractFromPages([SCANNED_NO_TEXT_LAYER]);
    expect(result.route).toBe("none");
    expect(result.abstract).toBeUndefined();
  });

  it("reports no usable text for empty input", () => {
    expect(extractAbstractFromPages([]).route).toBe("none");
    expect(extractAbstractFromPages(["", "   \n  "]).route).toBe("none");
  });

  it("falls back to leading body text when no marker is recognizable", () => {
    const page = `Some Journal Title Here 2019
${"An unstructured page of running prose that carries no abstract marker at all. ".repeat(20)}`;
    const result = extractAbstractFromPages([page]);
    expect(result.route).toBe("leading-text");
    expect(result.abstract).toBeTruthy();
    // The fallback is bounded — never the whole page.
    expect(result.abstract!.length).toBeLessThanOrEqual(MAX_ABSTRACT_CHARS);
  });

  it("bounds every route, so one odd layout cannot restore full-text volume", () => {
    const huge = `ABSTRACT\n${"word ".repeat(20_000)}`;
    const result = extractAbstractFromPages([huge]);
    expect(result.abstract!.length).toBeLessThanOrEqual(MAX_ABSTRACT_CHARS);
  });

  it("is deterministic and side-effect free", () => {
    expect(extractAbstractFromPages([AANDA_UPPERCASE_MARKER]))
      .toEqual(extractAbstractFromPages([AANDA_UPPERCASE_MARKER]));
  });

  it("reads only the leading pages it is given, never the whole document", () => {
    const body = ["ABSTRACT", "The real abstract body of this paper goes here and is long enough to keep."];
    const tail = Array.from({ length: 40 }, (_, i) => `Section ${i} body text that must never be indexed.`);
    const result = extractAbstractFromPages([body.join("\n"), ...tail]);
    expect(result.abstract).toMatch(/^The real abstract body/);
    expect(result.abstract).not.toMatch(/must never be indexed/);
  });
});
