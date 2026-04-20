window.QUICKSCOPE_PAGE = {
  updated: "April 2026",
  headline: "Find where your LLM fails, with confidence.",
  subhead:
    "QuickScope searches dynamic benchmark spaces for configurations where a model fails reliably, not just once. It adaptively allocates model calls, tracks confidence bounds, and returns certified weak spots under a user-chosen utility.",
  byline: [
    { name: "Taylor Lundy", url: "https://www.cs.ubc.ca/~tlundy/" },
    { name: "Narun Krishnamurthi Raman", url: "https://www.narunraman.com" },
    { name: "Kevin Leyton-Brown", url: "https://www.cs.ubc.ca/~kevinlb/" },
  ],
  benchmarkScorePanels: {
    source: "LLM Stats via ZeroEval API",
    fetched: "April 18, 2026",
    note: "Selected top records per benchmark.",
    panels: [
      {
        benchmark: "AIME 2025",
        scores: [
          { model: "Kimi K2-Thinking", score: 1.0 },
          { model: "Grok-4 Heavy", score: 1.0 },
          { model: "GPT-5.2 Pro", score: 1.0 },
          { model: "GPT-5.2", score: 1.0 },
          { model: "Gemini 3 Pro", score: 1.0 },
          { model: "Claude Opus 4.6", score: 0.9979 },
        ],
      },
      {
        benchmark: "GPQA",
        scores: [
          { model: "Gemini 3.1 Pro", score: 0.943 },
          { model: "GPT-5.2 Pro", score: 0.932 },
          { model: "GPT-5.4", score: 0.928 },
          { model: "GPT-5.2", score: 0.924 },
          { model: "Gemini 3 Pro", score: 0.919 },
          { model: "Claude Opus 4.6", score: 0.913 },
        ],
      },
      {
        benchmark: "SWE-Bench Verified",
        scores: [
          { model: "Claude Opus 4.5", score: 0.809 },
          { model: "Claude Opus 4.6", score: 0.808 },
          { model: "Gemini 3.1 Pro", score: 0.806 },
          { model: "MiniMax M2.5", score: 0.802 },
          { model: "GPT-5.2", score: 0.8 },
          { model: "Claude Sonnet 4.6", score: 0.796 },
        ],
      },
      {
        benchmark: "MMLU-Pro",
        scores: [
          { model: "Qwen3.6 Plus", score: 0.885 },
          { model: "MiniMax M2.1", score: 0.88 },
          { model: "Qwen3.5 397B", score: 0.878 },
          { model: "Kimi K2.5", score: 0.871 },
          { model: "ERNIE 5.0", score: 0.87 },
          { model: "Qwen3.5 122B", score: 0.867 },
        ],
      },
      {
        benchmark: "HumanEval",
        scores: [
          { model: "MiniCPM-SALA", score: 0.9512 },
          { model: "Kimi K2 0905", score: 0.945 },
          { model: "Claude 3.5 Sonnet", score: 0.937 },
          { model: "GPT-5", score: 0.934 },
          { model: "Kimi K2 Instruct", score: 0.933 },
          { model: "Qwen2.5-Coder 32B", score: 0.927 },
        ],
      },
    ],
  },
  dynamicSearch: {
    configCode: [
      ["const", "params", " = {"],
      ["key", "  dataset_type_id", ": ", "number", "1"],
      ["key", "  depth", ": ", "number", "3"],
      ["key", "  num_children_per_node", ": ", "number", "2"],
      ["key", "  extra_links_per_node", ": ", "number", "0"],
      ["key", "  add_rand_desc", ": ", "number", "0"],
      ["key", "  order_id", ": ", "number", "1"],
      ["plain", "}"],
    ],
    generatorCode: [
      ["tokens", [["variable", "generator"], ["plain", " = "], ["function", "DyValScenarioGenerator"], ["plain", "("], ["key", "model"], ["plain", "="], ["variable", "model"], ["plain", ")"]]],
      ["tokens", [["variable", "rng"], ["plain", " = "], ["variable", "np"], ["plain", "."], ["variable", "random"], ["plain", "."], ["function", "default_rng"], ["plain", "("], ["number", "2"], ["plain", ")"]]],
      ["tokens", [["variable", "row"], ["plain", " = "], ["variable", "generator"], ["plain", "."], ["function", "generate"], ["plain", "("], ["variable", "params"], ["plain", ", "], ["key", "rng"], ["plain", "="], ["variable", "rng"], ["plain", ")."], ["variable", "iloc"], ["plain", "["], ["number", "0"], ["plain", "]"]]],
      ["tokens", [["variable", "question"], ["plain", " = "], ["variable", "row"], ["plain", "["], ["string", "\"scenario_text\""], ["plain", "]"]]],
    ],
    instanceQuestion:
      "Here is a description of an arithmetic problem: The value of aae is 6. The value of aad is 2. aaf gets its value by adding together the value of aad and aae. The value of aag is 5. The value of aah is 5. aai gets its value by subtracting the value of aah from the value of aag. aaj gets its value by adding together the value of aaf and aai. Compute the result of aaj.",
    spaces: [
      { benchmark: "DyVal", size: "4,440 templates" },
      { benchmark: "STEER-ME", size: "303 templates" },
      { benchmark: "Grid Reasoning", size: ">2.3M templates + continuous parameters" },
    ],
  },
  notes: [
    {
      label: "Certification reallocates budget",
      value: "On Grid Reasoning, certification widened top-10 intervals by 8.1% while narrowing ranks 11-100 by 20.0%.",
    },
    {
      label: "Utility changes what is found",
      value: "On DyVal, CWE-optimized configs scored 4.3x higher on CWE than ER-optimized configs, with only 6 of 100 overlapping.",
    },
    {
      label: "Repulsion diversifies certificates",
      value: "On DyVal, repulsion found arithmetic failures plus 6 certified linear-equation configurations instead of arithmetic alone.",
    },
  ],
  runs: [
    {
      id: "dyval-nano-er-cert",
      benchmark: "DyVal",
      model: "gpt-5.4-nano",
      utility: "error-rate",
      method: "COUP + cert 0.9",
      threshold: 0.9,
      summary: "Top-ranked DyVal configurations reach near-perfect re-evaluated utility; hard regions concentrate in deep, narrow arithmetic chains.",
      metrics: [
        { label: "Search space", value: "4,440 configs" },
        { label: "Top-10 re-eval", value: ">0.99" },
        { label: "Weak spot", value: "deep arithmetic" },
      ],
      series: [
        { batch: 0, calls: 0, incumbent: "random initial pool", mean: 0.12, lcb: 0.0, ucb: 1.0, certified: false },
        { batch: 100, calls: 2000, incumbent: "arithmetic / depth 9 / children 2", mean: 0.78, lcb: 0.55, ucb: 0.95, certified: false },
        { batch: 250, calls: 5000, incumbent: "arithmetic / depth 10 / children 2", mean: 0.91, lcb: 0.78, ucb: 0.98, certified: false },
        { batch: 500, calls: 10000, incumbent: "arithmetic / depth 10 / children 2", mean: 0.97, lcb: 0.91, ucb: 0.99, certified: true },
        { batch: 750, calls: 15000, incumbent: "arithmetic / depth 10 / children 2", mean: 0.99, lcb: 0.94, ucb: 1.0, certified: true },
        { batch: 1000, calls: 20000, incumbent: "arithmetic / depth 10 / children 2", mean: 0.99, lcb: 0.96, ucb: 1.0, certified: true },
      ],
      topConfigs: [
        { rank: 1, name: "arithmetic / depth 10 / children 2", mean: 0.99, lcb: 0.96, certified: true },
        { rank: 2, name: "arithmetic / depth 9 / children 2", mean: 0.98, lcb: 0.94, certified: true },
        { rank: 3, name: "logical deduction / depth 10", mean: 0.95, lcb: 0.90, certified: true },
      ],
    },
    {
      id: "dyval-nano-cwe",
      benchmark: "DyVal",
      model: "gpt-5.4-nano",
      utility: "complexity-weighted error",
      method: "COUP",
      threshold: null,
      summary: "Complexity-weighted error surfaces shallow templates where failures are surprising rather than merely high-complexity.",
      metrics: [
        { label: "CWE ratio", value: "4.3x ER" },
        { label: "Top-100 overlap", value: "6 configs" },
        { label: "Selected depths", value: "mostly 2-3" },
      ],
      series: [
        { batch: 0, calls: 0, incumbent: "random initial pool", mean: 0.04, lcb: 0.0, ucb: 1.0, certified: false },
        { batch: 100, calls: 2000, incumbent: "arithmetic / depth 3 / children 2", mean: 0.23, lcb: 0.11, ucb: 0.38, certified: false },
        { batch: 250, calls: 5000, incumbent: "arithmetic / depth 2 / children 2", mean: 0.34, lcb: 0.21, ucb: 0.48, certified: false },
        { batch: 500, calls: 10000, incumbent: "arithmetic / depth 2 / children 2", mean: 0.42, lcb: 0.31, ucb: 0.54, certified: false },
        { batch: 750, calls: 15000, incumbent: "linear equation / depth 3", mean: 0.45, lcb: 0.36, ucb: 0.56, certified: false },
        { batch: 1000, calls: 20000, incumbent: "linear equation / depth 3", mean: 0.46, lcb: 0.38, ucb: 0.56, certified: false },
      ],
      topConfigs: [
        { rank: 1, name: "linear equation / depth 3", mean: 0.46, lcb: 0.38, certified: false },
        { rank: 2, name: "arithmetic / depth 2 / children 2", mean: 0.44, lcb: 0.35, certified: false },
        { rank: 3, name: "arithmetic / depth 3 / children 2", mean: 0.41, lcb: 0.32, certified: false },
      ],
    },
    {
      id: "grid-nano-er-cert",
      benchmark: "Grid Reasoning",
      model: "gpt-5.4-nano",
      utility: "error-rate",
      method: "COUP + cert 0.9",
      threshold: 0.9,
      summary: "Certification trades a little top-end precision for sharper identification further down the ranking.",
      metrics: [
        { label: "Search space", value: ">2.3M configs" },
        { label: "CI top-10", value: "+8.1%" },
        { label: "CI ranks 11-100", value: "-20.0%" },
      ],
      series: [
        { batch: 0, calls: 0, incumbent: "random initial pool", mean: 0.1, lcb: 0.0, ucb: 1.0, certified: false },
        { batch: 100, calls: 2000, incumbent: "largest island / 18x20 / many islands", mean: 0.81, lcb: 0.58, ucb: 0.97, certified: false },
        { batch: 250, calls: 5000, incumbent: "shortest path / 24x25 / high blocking", mean: 0.9, lcb: 0.74, ucb: 0.99, certified: false },
        { batch: 500, calls: 10000, incumbent: "largest island / 22x23 / many islands", mean: 0.97, lcb: 0.91, ucb: 1.0, certified: true },
        { batch: 750, calls: 15000, incumbent: "largest island / 22x23 / many islands", mean: 0.99, lcb: 0.94, ucb: 1.0, certified: true },
        { batch: 1000, calls: 20000, incumbent: "largest island / 22x23 / many islands", mean: 0.99, lcb: 0.95, ucb: 1.0, certified: true },
      ],
      topConfigs: [
        { rank: 1, name: "largest island / 22x23 / many islands", mean: 0.99, lcb: 0.95, certified: true },
        { rank: 2, name: "shortest path / 24x25 / high blocking", mean: 0.98, lcb: 0.93, certified: true },
        { rank: 3, name: "largest island / 20x22 / large island size", mean: 0.96, lcb: 0.91, certified: true },
      ],
    },
    {
      id: "steer-mini-er-cert",
      benchmark: "STEER-ME",
      model: "gpt-5-mini",
      utility: "error-rate",
      method: "COUP + cert 0.7",
      threshold: 0.7,
      summary: "STEER-ME is noisier because 4-choice MCQA has high Bernoulli variance near chance error, so bounds tighten more slowly.",
      metrics: [
        { label: "Search space", value: "303 configs" },
        { label: "Chance error", value: "0.75" },
        { label: "Pattern", value: "numerical econ" },
      ],
      series: [
        { batch: 0, calls: 0, incumbent: "random initial pool", mean: 0.16, lcb: 0.0, ucb: 1.0, certified: false },
        { batch: 100, calls: 2000, incumbent: "producer surplus / politics / linear", mean: 0.62, lcb: 0.39, ucb: 0.82, certified: false },
        { batch: 250, calls: 5000, incumbent: "labor supply / technology / equilibrium", mean: 0.71, lcb: 0.52, ucb: 0.87, certified: false },
        { batch: 500, calls: 10000, incumbent: "intertemporal smoothing / sports / numerical", mean: 0.82, lcb: 0.66, ucb: 0.91, certified: false },
        { batch: 750, calls: 15000, incumbent: "intertemporal smoothing / sports / numerical", mean: 0.87, lcb: 0.72, ucb: 0.93, certified: true },
        { batch: 1000, calls: 20000, incumbent: "intertemporal smoothing / sports / numerical", mean: 0.88, lcb: 0.74, ucb: 0.94, certified: true },
      ],
      topConfigs: [
        { rank: 1, name: "intertemporal smoothing / sports / numerical", mean: 0.88, lcb: 0.74, certified: true },
        { rank: 2, name: "labor supply / technology / equilibrium", mean: 0.83, lcb: 0.70, certified: true },
        { rank: 3, name: "producer surplus / politics / linear", mean: 0.79, lcb: 0.66, certified: false },
      ],
    },
  ],
};

for (const run of window.QUICKSCOPE_PAGE.runs) {
  if (run.configTraces) continue;
  const traceCount = 40;
  const timepoints = Array.from({ length: 101 }, (_, index) => {
    const calls = index * 200;
    const lower = [...run.series].reverse().find((point) => point.calls <= calls) || run.series[0];
    const upper = run.series.find((point) => point.calls >= calls) || run.series[run.series.length - 1];
    const span = Math.max(1, upper.calls - lower.calls);
    const t = Math.max(0, Math.min(1, (calls - lower.calls) / span));
    const mix = (key) => lower[key] + (upper[key] - lower[key]) * t;
    return {
      batch: Math.round(calls / 20),
      calls,
      mean: mix("mean"),
      lcb: mix("lcb"),
      ucb: mix("ucb"),
    };
  });
  run.displayTopK = 200;
  run.configTraces = Array.from({ length: traceCount }, (_, configIndex) => {
    const base = run.topConfigs[configIndex % run.topConfigs.length];
    const rank = configIndex + 1;
    const tierDrop = Math.floor(configIndex / run.topConfigs.length) * 0.028;
    const withinDrop = (configIndex % run.topConfigs.length) * 0.012;
    const config = {
      name: rank <= run.topConfigs.length ? base.name : `${base.name} variant ${Math.floor(configIndex / run.topConfigs.length) + 1}`,
      rank,
      mean: Math.max(0.08, base.mean - tierDrop - withinDrop),
      lcb: Math.max(0.02, base.lcb - tierDrop - withinDrop * 1.1),
      certified: base.certified && configIndex < 24,
    };
    const finalMean = Math.max(0, Math.min(1, config.mean - configIndex * 0.015));
    const finalLcb = Math.max(0, Math.min(finalMean, config.lcb - configIndex * 0.012));
    const startMean = Math.max(0.04, finalMean * (0.15 + configIndex * 0.04));
    return {
      name: config.name,
      rank: config.rank,
      certified: config.certified,
      points: timepoints.map((point, pointIndex) => {
        const progress = point.calls / timepoints[timepoints.length - 1].calls;
        const wobble = Math.sin((pointIndex + 1) * (configIndex + 2)) * 0.018;
        const mean = Math.max(
          0,
          Math.min(1, startMean + (finalMean - startMean) * progress + wobble)
        );
        const band = Math.max(0.035, 0.42 * (1 - progress) + 0.035 + configIndex * 0.006);
        const lcb = Math.max(0, Math.min(mean, mean - band * 0.55));
        const ucb = Math.min(1, Math.max(mean, mean + band * 0.45));
        return {
          batch: point.batch,
          calls: point.calls,
          mean: pointIndex === timepoints.length - 1 ? finalMean : mean,
          lcb: pointIndex === timepoints.length - 1 ? finalLcb : lcb,
          ucb,
        };
      }),
    };
  });
}
