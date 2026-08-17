export interface RetrievalJudgment {
  queryId: string;
  category: string;
  /** Paper key to human-assigned graded relevance; zero/absent means irrelevant. */
  grades: Readonly<Record<string, number>>;
}

export interface RetrievalMetrics {
  recall: number;
  mrr: number;
  ndcg: number;
}

export interface RetrievalModeReport {
  overall: RetrievalMetrics;
  categories: Readonly<Record<string, RetrievalMetrics>>;
}

export interface RetrievalEvaluationReport {
  k: number;
  modes: Readonly<Record<string, RetrievalModeReport>>;
}

export interface EvaluateRetrievalInput {
  judgments: readonly RetrievalJudgment[];
  rankings: Readonly<Record<string, Readonly<Record<string, readonly string[]>>>>;
  k: number;
}

export interface HybridRetrievalGateInput {
  denseMode: string;
  lexicalMode: string;
  hybridMode: string;
  lexicalCategories: readonly string[];
  semanticCategories: readonly string[];
  semanticRecallTolerance?: number;
}

/** Pure deterministic evaluation over fixed judgments and supplied rankings. */
export function evaluateRetrieval(input: EvaluateRetrievalInput): RetrievalEvaluationReport {
  if (!Number.isSafeInteger(input.k) || input.k < 1) throw new TypeError("evaluateRetrieval: k must be positive");
  const modes: Record<string, RetrievalModeReport> = {};
  for (const [mode, rankings] of Object.entries(input.rankings)) {
    const byCategory = new Map<string, RetrievalMetrics[]>();
    const all: RetrievalMetrics[] = [];
    for (const judgment of input.judgments) {
      const metrics = queryMetrics(judgment.grades, rankings[judgment.queryId] ?? [], input.k);
      all.push(metrics);
      const category = byCategory.get(judgment.category) ?? [];
      category.push(metrics);
      byCategory.set(judgment.category, category);
    }
    modes[mode] = {
      overall: average(all),
      categories: Object.fromEntries([...byCategory].map(([category, metrics]) => [category, average(metrics)])),
    };
  }
  return { k: input.k, modes };
}

/** Acceptance gates from P3: parity overall, one lexical win, semantic recall preserved. */
export function assertHybridRetrievalGates(
  report: RetrievalEvaluationReport,
  gates: HybridRetrievalGateInput,
): void {
  const dense = requireMode(report, gates.denseMode);
  const lexical = requireMode(report, gates.lexicalMode);
  const hybrid = requireMode(report, gates.hybridMode);
  const tolerance = gates.semanticRecallTolerance ?? 0.05;
  for (const category of gates.semanticCategories) {
    const hybridMetric = hybrid.categories[category];
    const denseMetric = dense.categories[category];
    if (!hybridMetric || !denseMetric) throw new Error(`missing category ${category}`);
    if (hybridMetric.recall + tolerance < denseMetric.recall) {
      throw new Error(`${category} hybrid Recall@${report.k} ${hybridMetric.recall} regressed from ${denseMetric.recall}`);
    }
  }
  for (const metric of ["recall", "mrr", "ndcg"] as const) {
    if (hybrid.overall[metric] + Number.EPSILON < dense.overall[metric]) {
      throw new Error(`hybrid ${metric} ${hybrid.overall[metric]} is below dense ${dense.overall[metric]}`);
    }
  }
  const improved = gates.lexicalCategories.some((category) => {
    const hybridMetric = hybrid.categories[category];
    const denseMetric = dense.categories[category];
    const lexicalMetric = lexical.categories[category];
    return !!hybridMetric && !!denseMetric && !!lexicalMetric
      && hybridMetric.ndcg > denseMetric.ndcg
      && lexicalMetric.ndcg > denseMetric.ndcg;
  });
  if (!improved) throw new Error("no lexical category strictly improves over dense");

}

function queryMetrics(grades: Readonly<Record<string, number>>, ranking: readonly string[], k: number): RetrievalMetrics {
  const relevant = Object.entries(grades).filter(([, grade]) => grade > 0);
  if (relevant.length === 0) return { recall: 0, mrr: 0, ndcg: 0 };
  const top = ranking.slice(0, k);
  let found = 0;
  let firstRank = 0;
  let dcg = 0;
  top.forEach((paperKey, index) => {
    const grade = grades[paperKey] ?? 0;
    if (grade <= 0) return;
    found += 1;
    if (firstRank === 0) firstRank = index + 1;
    dcg += (2 ** grade - 1) / Math.log2(index + 2);
  });
  const ideal = relevant
    .map(([, grade]) => grade)
    .sort((left, right) => right - left)
    .slice(0, k)
    .reduce((sum, grade, index) => sum + (2 ** grade - 1) / Math.log2(index + 2), 0);
  return {
    recall: found / relevant.length,
    mrr: firstRank === 0 ? 0 : 1 / firstRank,
    ndcg: ideal === 0 ? 0 : dcg / ideal,
  };
}

function average(metrics: readonly RetrievalMetrics[]): RetrievalMetrics {
  if (metrics.length === 0) return { recall: 0, mrr: 0, ndcg: 0 };
  const sum = metrics.reduce((total, current) => ({
    recall: total.recall + current.recall,
    mrr: total.mrr + current.mrr,
    ndcg: total.ndcg + current.ndcg,
  }), { recall: 0, mrr: 0, ndcg: 0 });
  return { recall: sum.recall / metrics.length, mrr: sum.mrr / metrics.length, ndcg: sum.ndcg / metrics.length };
}

function requireMode(report: RetrievalEvaluationReport, name: string): RetrievalModeReport {
  const mode = report.modes[name];
  if (!mode) throw new Error(`missing retrieval mode ${name}`);
  return mode;
}
