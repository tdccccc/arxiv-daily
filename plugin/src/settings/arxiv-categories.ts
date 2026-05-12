export interface CategoryGroup {
  label: string;
  categories: Array<{ id: string; name: string }>;
}

export const ARXIV_CATEGORIES: CategoryGroup[] = [
  {
    label: "物理",
    categories: [
      { id: "astro-ph", name: "Astrophysics" },
      { id: "astro-ph.CO", name: "Cosmology" },
      { id: "astro-ph.GA", name: "Galaxies" },
      { id: "astro-ph.HE", name: "High Energy" },
      { id: "astro-ph.SR", name: "Solar & Stellar" },
      { id: "cond-mat", name: "Condensed Matter" },
      { id: "gr-qc", name: "GR & Quantum Cosmology" },
      { id: "hep-ex", name: "HEP Experiment" },
      { id: "hep-lat", name: "HEP Lattice" },
      { id: "hep-ph", name: "HEP Phenomenology" },
      { id: "hep-th", name: "HEP Theory" },
      { id: "nucl-ex", name: "Nuclear Experiment" },
      { id: "nucl-th", name: "Nuclear Theory" },
      { id: "physics", name: "Physics (General)" },
      { id: "quant-ph", name: "Quantum Physics" },
    ],
  },
  {
    label: "计算机",
    categories: [
      { id: "cs.AI", name: "Artificial Intelligence" },
      { id: "cs.AR", name: "Hardware Architecture" },
      { id: "cs.CL", name: "Computation & Language" },
      { id: "cs.CR", name: "Cryptography" },
      { id: "cs.CV", name: "Computer Vision" },
      { id: "cs.DS", name: "Data Structures" },
      { id: "cs.IR", name: "Information Retrieval" },
      { id: "cs.IT", name: "Information Theory" },
      { id: "cs.LG", name: "Machine Learning" },
      { id: "cs.MA", name: "Multiagent Systems" },
      { id: "cs.NE", name: "Neural & Evolutionary" },
      { id: "cs.NI", name: "Networking" },
      { id: "cs.RO", name: "Robotics" },
      { id: "cs.SE", name: "Software Engineering" },
    ],
  },
  {
    label: "数学",
    categories: [
      { id: "math.AG", name: "Algebraic Geometry" },
      { id: "math.AP", name: "Analysis of PDEs" },
      { id: "math.CO", name: "Combinatorics" },
      { id: "math.NT", name: "Number Theory" },
      { id: "math.OC", name: "Optimization" },
      { id: "math-ph", name: "Mathematical Physics" },
      { id: "math.PR", name: "Probability" },
      { id: "math.ST", name: "Statistics Theory" },
    ],
  },
  {
    label: "统计 / 生物 / 经济",
    categories: [
      { id: "stat.ML", name: "Machine Learning (Stats)" },
      { id: "stat.ME", name: "Methodology" },
      { id: "q-bio", name: "Quantitative Biology" },
      { id: "econ.EM", name: "Econometrics" },
    ],
  },
];
