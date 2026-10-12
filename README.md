# Personalized LLM Routing

This repository contains two related but distinct workflows:

1. A real-model data collection and routing proof of concept using `build_dataset.py`, `parse_dataset.py`, and `compare_llms.py`.
2. A standalone synthetic experiment suite in `run_experiments.py`.

The experiment suite does not load the real-model dataset. It simulates its own contexts, model costs, and user feedback; results from it should not be described as measurements of the five APIs used in the data-collection pipeline.

## 1. Real-Model Data and Routing POC

### Workflow

```text
MMLU-Pro test questions
        |
        v
build_dataset.py: query five candidate models and judge each response
        |                                  |
        |                                  +--> summary.csv
        +--> responses.jsonl --> routing_dataset.jsonl
                                      |
                                      v
parse_dataset.py --> parsed_quality_vectors.txt
                                      |
                                      v
compare_llms.py --> routing replay CSVs and plots
```

### Environment and dependencies

The data builder calls the Lumen OpenAI-compatible API and requires `LUMEN_API_KEY`. It also downloads `TIGER-Lab/MMLU-Pro` through Hugging Face Datasets. Install the libraries used by the scripts with:

```bash
python -m pip install datasets matplotlib numpy openai pandas scipy tqdm
```

Set the API key in the shell where the builder will run:

```bash
export LUMEN_API_KEY="your-lumen-api-key"
```

### Step 1: Generate candidate responses and judge scores

Run from the repository root:

```bash
python build_dataset.py
```

The script loads the MMLU-Pro `test` split and selects 100 questions with seed `42`. It takes one question from each available category where possible, fills the remaining slots by sampling from unused rows, then shuffles the selection. The choices are formatted as lettered multiple-choice options.

Each question is sent to these five candidate models, in this order:

1. `nemotron-3-super-120b-a12b`
2. `deepseek-v4-flash`
3. `qwen3.6-35b-a3b`
4. `gemma-4-31b-it`
5. `qwen3.8-27b`

Candidate calls use temperature `0.0`, a 512-token maximum, and up to five attempts with exponential backoff. The responses are evaluated by `muse-glimmer-30b`, which is asked to score correctness, relevance, completeness, clarity, instruction following, safety, and satisfaction from 0 to 1. The parser accepts key/value scores and falls back to positional numeric scores; missing values become `0.0`, and all values are clamped to `[0, 1]`.

The six-dimensional routing feature vector is ordered as:

```text
[correctness, relevance, completeness, clarity,
 instruction_following, safety]
```

Judge satisfaction is retained as a separate metric. The builder also creates a binary `Y` label equal to 1 only when correctness is at least `0.80`, relevance at least `0.80`, completeness at least `0.70`, and safety at least `0.90`. This threshold-based `Y` is saved in the dataset, but **`compare_llms.py` does not use it**; that script generates its own synthetic Bernoulli outcomes, as described below.

Files are written beneath `llm_routing_poc_data/`:

| File | Contents |
| --- | --- |
| `responses.jsonl` | One candidate response record per successful model call, including query, model, response, usage, and call status. |
| `routing_dataset.jsonl` | One judged record per successful response, including `judge_metrics`, the six-value `quality_vector`, and threshold-based `Y`. |
| `summary.csv` | Per-model mean judge metrics and satisfaction rate. |

Existing candidate responses are reused by `(query_id, model)` when the builder is rerun. Judging is not cached: the script removes and rebuilds `routing_dataset.jsonl` and re-judges the collected responses on each run. Failed candidate calls are omitted from the judged dataset.

### Step 2: Export the quality vectors

Run:

```bash
python parse_dataset.py
```

The script reads `llm_routing_poc_data/routing_dataset.jsonl` and writes `llm_routing_poc_data/parsed_quality_vectors.txt`. This is a formatting/export step, not a scoring step: it copies each record's existing `quality_vector` (with fallbacks for other supported field names) next to its query ID. Malformed JSON lines are skipped. The output is a pipe-delimited text table with a JSON list in the right-hand column, for example:

```text
Query Num/ID    | Model Quality Vector / Scores
0               | [0.8, 0.9, 0.7, 0.8, 0.9, 1.0]
```

### Step 3: Run the contextual routing POC

Before running `compare_llms.py`, check `DATA_DIR` near the top of the script. It currently points to `/home/ameya-anjarlekar/llm-routing-poc/llm_routing_poc_data`, which may not be this repository's output directory. Change it to the directory containing `parsed_quality_vectors.txt` on your machine. Then run:

```bash
python compare_llms.py
```

The loader requires integer query IDs and JSON lists of exactly six values in `[0, 1]`. There is a current edge case in `parse_dataset.py`: it selects IDs using truthiness, so the valid ID `0` is replaced with a text fallback such as `Query #1`; the comparator then rejects that ID because it is not an integer. Fix the parser's fallback handling or convert that first exported ID back to integer `0` before running the comparator.

The comparator groups **every five consecutive rows** into one query's five model actions, in the `MODELS` order above, and checks that all five rows in each block have the same query ID. Therefore, the input must contain exactly one successful row for every model for each query, in query-major/model-minor order. Because failed calls are omitted by the builder and the parser does not sort or fill missing models, incomplete query blocks must be repaired or filtered consistently before this stage; otherwise grouping will fail or associate scores with the wrong model.

The comparator normalizes a raw vector `q` to `x = q / sqrt(6)`, ensuring the context norm is no greater than 1. It assumes a fixed, hidden six-dimensional preference vector formed by normalizing `[0.60, 0.50, 0.40, 0.30, 0.20, 0.10]`. For each query/model pair it calculates `p = theta* dot x`, then samples one Bernoulli outcome using that probability. These outcomes are generated once and reused across 100 randomized query-order replays; the learner observes feedback only for the selected model. Thus the judge scores act as contextual features, while the POC's satisfaction outcomes are simulated and are not the judge's `Y` labels or ground-truth MMLU correctness.

The POC uses the five candidate models as its action set, with costs proportional to parameter counts `[120, 284, 35, 30.7, 27]` billion and normalized by the largest count. It sets `epsilon = 0.20`, `delta = 0.05`, regularization `beta = 1.0`, confidence parameter `zeta = 2.5`, empirical alpha scale `0.005`, and reference `rho = 0.15` at horizon 100. The reference `eta` is `0.5`; note that `compute_eta()` currently returns this constant regardless of epsilon. The script's header describes the five real models as the complete action set with no fallback.

The default replay results are written to `llm_routing_poc_data/poc_results/poc_default_trials.csv`. The epsilon sweep uses `[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]` and writes `poc_epsilon_sweep.csv`; the script also saves `poc_cost_vs_epsilon.png`, `poc_satisfaction_vs_epsilon.png`, `poc_model_selection_vs_epsilon.png`, and `poc_config.json` in `poc_results/`. Results report average cost, average simulated satisfaction, the fraction of replay trials meeting the satisfaction target, and model-use frequencies.

## 2. Synthetic Experiment Suite (`run_experiments.py`)

Run the entire suite from the repository root:

```bash
python run_experiments.py
```

This file is standalone and runs all nine experiments sequentially. It does not read `llm_routing_poc_data/parsed_quality_vectors.txt`, call an LLM API, or use the real model names from the first part. It creates five simulated model actions, eight-dimensional contexts, synthetic binary feedback, and a synthetic fallback action. Output is written to `icml_llm_routing_results/`; trial-level and error-bar artifacts are generally in its `errorbars/` subdirectory.

### Shared synthetic setup

| Setting | Value |
| --- | --- |
| Random seed | `12345` |
| Main horizon | `T = 1000` |
| Main target | `epsilon = 0.20`, so required average satisfaction is `0.80` |
| Main chance-constraint level | `delta = 0.05`, so target probability is `0.95` |
| Context dimension / model actions | `D = 8`, `K = 5` |
| Model costs | `[0.10, 0.25, 0.40, 0.60, 0.80]` |
| Fallback | Action index `K` (the sixth action), cost `1.0`, UCB `1.0` |
| GLM/context constants | `L = 3.0`, `S = 1.0`, `beta = 1.0`, `zeta = 2.5`, `xi = 0.35` |
| Model context radii | `[1.05, 1.1625, 1.275, 1.3875, 1.50]` |
| Reference values | `T0 = 100`, `eta0 = 0.5`, `rho0 = 0.15` |
| Confidence calibration | Theoretical alpha is multiplied by a frozen empirical scale of `0.005`. |

For horizons other than the reference horizon, `eta(T) = 0.5 * sqrt(100/T)` and `rho(T) = 0.15 * sqrt(100/T)`. The logistic feedback model samples binary satisfaction from the generated context and preference parameters. Contexts are generated inside model-specific spherical caps, with configurable user/context mixing. Results use Student-t 95% confidence intervals for means and exact Clopper-Pearson 95% intervals for proportions. The main method is compared with removal of UCB, queueing, and the `rho` mechanism. The fallback action is available when required to satisfy the simulated safety/queue constraints.

Sample counts are configured near the top of the script: main performance `1000`, chance constraint `5000`, epsilon sweep `1000` per value, delta sweep `1000` per value, personalization `1000`, ablation `1000` per method, context robustness `1000` per value, horizon scaling `1000` per horizon, and confidence calibration `1000`. The horizon sweep is `[100, 250, 500, 1000, 2000]`.

### Experiments

| # | Experiment and design | Main outputs |
| --- | --- | --- |
| 1 | **Main performance.** At the main settings and `T = 1000`, compare `Ours` with `No-UCB`, `No-Queue`, and `No-Rho`. Reports mean cost, satisfaction, chance-constraint probability, and fallback rate. | `errorbars/main_performance_trials.csv`, `main_performance_errorbars.csv`, and cost/satisfaction/constraint 95% CI plots. |
| 2 | **Chance constraint.** Run the full method for 5000 trials at `epsilon = 0.20`, `delta = 0.05`; estimate the probability that episode satisfaction reaches `0.80`, with an exact binomial interval. Also report mean cost/satisfaction and compare `rho` with calculated `rho_max*`. | `errorbars/chance_constraint_trials.csv`, `chance_constraint.csv`, and `chance_constraint_distribution.png`. |
| 3 | **Epsilon sweep.** Vary `epsilon` across `0.10, 0.15, 0.20, 0.25, 0.30`, keeping the horizon and other main settings fixed. Compare achieved satisfaction with target `1 - epsilon` and track cost. | `errorbars/epsilon_sweep_trials.csv`, `epsilon_sweep_errorbars.csv`, and cost/satisfaction 95% CI plots. |
| 4 | **Delta sweep.** Vary `delta` across `0.01, 0.05, 0.10` at `epsilon = 0.20`; compare observed constraint probability with target `1 - delta`, and report cost and satisfaction. | `errorbars/delta_sweep_trials.csv`, `delta_sweep_errorbars.csv`, and a constraint-probability 95% CI plot. |
| 5 | **Personalization.** Compare a global learner with personalized learners across five simulated users whose preference directions are aligned to their corresponding model directions with alignment `0.7`. Both methods use the same counterfactual environments; per-user cost, satisfaction, constraint probability, and improvements are summarized. | `errorbars/personalization_corrected_trials.csv` and `personalization_corrected_summary.csv`. |
| 6 | **Ablations.** Repeat the four method configurations from the main experiment (`Ours`, `No-UCB`, `No-Queue`, `No-Rho`) using the ablation trial budget; compare cost, satisfaction, and constraint probability. | `errorbars/ablation_trials.csv`, `ablation_errorbars.csv`, and cost/satisfaction/constraint 95% CI plots. |
| 7 | **Context robustness.** Hold the preference direction essentially fixed and vary context/user mixing `xi` over `0.0, 0.20, 0.35, 0.50, 0.75, 1.0`; report cost, satisfaction, and constraint performance. | `errorbars/robustness_trials.csv`, `robustness_errorbars.csv`, and associated plots. |
| 8 | **Finite-horizon cost-gap scaling.** Run horizons `100, 250, 500, 1000, 2000`; scale `eta` and `rho` with the horizon, compute a stationary oracle policy for comparison, and estimate the empirical exponent of cumulative cost gap. Oracle policy compositions are recorded as well. | `regret_scaling.csv`, `regret_scaling_errorbars.csv`, `oracle_policy_composition_trials.csv`, and scaling plots in `icml_llm_routing_results/`. |
| 9 | **Confidence calibration.** Across 1000 synthetic trials, compare preference-estimation errors with theoretical and empirically scaled confidence widths. The frozen scale `0.005` is assessed empirically. | `confidence_calibration.csv` and confidence-calibration plots in `icml_llm_routing_results/`. |

The Experiment 8 exponent is an empirical scaling estimate, not a proof of a theoretical `O(sqrt(T))` guarantee. Likewise, Experiment 9's `alpha_scale = 0.005` is empirical calibration and should not be interpreted as a theoretically guaranteed 95% confidence bound. These caveats are also printed when the suite finishes.

### Reproducibility and outputs

The experiment suite creates its output directories automatically. It writes raw trial CSVs to support re-analysis, summary CSVs with confidence intervals, and PNG plots. Increasing the `N_*` constants tightens Monte Carlo estimates at the cost of runtime; the chance-constraint experiment alone defaults to 5000 trials, and the full script runs every experiment on each invocation. To run only one experiment interactively, import `run_experiments` from Python and call the desired `experiment_*()` function; importing the module also creates the output directories and prints the suite banner.
