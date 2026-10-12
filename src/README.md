# Configurable LLM Routing Pipeline

This directory contains configurable copies of the dataset builder, score exporter, and routing POC, plus a Bash script that runs them in order. The original scripts in the repository root are not changed by this workflow.

## Run It

From the repository root:

```bash
conda activate llm_routing
export LUMEN_API_KEY="your-lumen-api-key"
./src/run_routing.sh 100 "model-a,model-b" "120,80"
```

The three positional arguments are:

| Argument | Meaning |
| --- | --- |
| `T` | Positive number of prompts to sample from the `TIGER-Lab/MMLU-Pro` test split. It cannot exceed the available number of rows. |
| Model list | Comma-separated model IDs. Their order defines the routing action order and must match the compute-value order. |
| Compute list | One finite positive numeric compute value per model. Values are normalized by their maximum to become routing costs in `[0, 1]`. |

The shell runner rejects an invalid prompt count, empty model IDs, a model/compute count mismatch, non-positive or non-numeric compute values, and a missing `LUMEN_API_KEY` before starting generation. Candidate and judge calls use the Lumen OpenAI-compatible API. The judge model remains `muse-glimmer-30b`, configured in `build_dataset.py`.

## Pipeline

```text
run_routing.sh
  for prompt 1..T:
    build_dataset.py helpers -> answers from each configured model
    build_dataset.py judge   -> judged metrics and threshold feedback
    parse_dataset.py         -> export all vectors collected so far
    compare_llms.StreamingRouter.step -> route this prompt and update state
  after prompt T:
    write summaries, step log, plots, and final configuration
```

The default output directory is `src/llm_routing_poc_data/`. The runner changes its working directory to `src/`, so relative paths and default outputs are stable regardless of the caller's current directory. The online learner receives one update per prompt in dataset order; it does not wait for all `T` prompts before its first decision.

### `run_routing.sh`

This is the public entry point. It takes exactly `T`, the comma-separated model IDs, and the comma-separated compute values. It validates inputs and credentials, then invokes `run_routing.py` once with those settings. A nonzero exit stops the pipeline.

Run the help/usage message by invoking it with the wrong number of arguments:

```bash
./src/run_routing.sh
```

### `build_dataset.py`

This copy contains the response-generation, judge, and scoring helpers used by the streaming runner. It can also be run by itself in its original batch mode. Its required standalone options are:

```text
--models MODEL_A,MODEL_B,...
--n-queries T
```

`--output-dir` optionally overrides the default `llm_routing_poc_data` directory. The builder samples `T` MMLU-Pro test questions with the existing seeded category-balanced selection, asks every configured candidate model for an answer, and judges each successful answer on correctness, relevance, completeness, clarity, instruction following, safety, and satisfaction. Candidate sampling uses temperature `0.0`, a 512-token maximum, and up to five attempts with exponential backoff.

Its six-value feature vector is `[correctness, relevance, completeness, clarity, instruction_following, safety]`. The separately saved binary `Y` is 1 when correctness and relevance are each at least `0.80`, completeness is at least `0.70`, and safety is at least `0.90`. The comparator does not use this saved `Y`; it simulates its own Bernoulli feedback from the feature vectors.

The output files are:

| File | Description |
| --- | --- |
| `responses.jsonl` | Candidate response, model ID, query ID, category, usage, and API success status. |
| `routing_dataset.jsonl` | Successful response records with judge metrics, six-dimensional quality vector, binary `Y`, and usage. |
| `summary.csv` | Per-model averages for each judge metric and the threshold-based satisfaction rate. |

Successful cached responses are reused only when both the model ID and the exact sampled question match. Failed calls are retried on a later run. In streaming mode, a candidate call still missing after retries stops the run before judging/routing that prompt rather than emitting misaligned actions.

### `parse_dataset.py`

This stage formats, but does not calculate, the quality vectors stored in `routing_dataset.jsonl`. It accepts `--input-file` and `--output-file`; defaults are `llm_routing_poc_data/routing_dataset.jsonl` and `llm_routing_poc_data/parsed_quality_vectors.txt`. In the online workflow, it is called after each prompt has been judged and rewrites the export with all vectors collected so far; the just-added prompt's `K` vectors are then passed to the router.

The exported table has a query ID before `|` and a JSON list of six scores after it. Query ID `0` is preserved as an integer in this copy. Malformed JSON records are skipped; valid records without a recognized query ID get a text fallback, which the comparator will reject because routing inputs require integer IDs.

### `compare_llms.py`

The required options for standalone batch mode are `--models MODEL_A,MODEL_B,...` and `--compute-values VALUE_A,VALUE_B,...`. Optional `--quality-file` and `--output-dir` override the default parsed-vector input and `poc_results` output directory. The model and compute lists must have equal length; IDs must be unique; values must be finite and positive.

The loader expects one six-score row per model for every query, in query-major order and the exact model order passed to the CLI. It groups `K` consecutive rows into each prompt and rejects inconsistent query IDs or malformed score vectors. Scores must be in `[0, 1]`.

For each raw quality vector `q`, the comparator uses the normalized feature `x = q / sqrt(6)`. Its `StreamingRouter.step()` keeps the design matrix, estimate, and dual queue between calls. Given one prompt's `K` contexts and judge-derived threshold outcomes, it chooses one model, observes only that model's outcome, updates its state, and returns the step result. The streaming runner calls this before moving to the next prompt, in original dataset order. The configured compute values are normalized by their maximum to form action costs; there is no fallback action.

When `compare_llms.py` is run by itself, it retains its batch POC behavior and uses simulated Bernoulli outcomes from a fixed preference vector. The streaming runner instead uses the builder's judge-threshold `Y` for the chosen action. After the online pass ends, it also runs the existing randomized epsilon sweep on the collected contexts/outcomes as a retrospective diagnostic; this sweep does not affect the online decisions. The sweep uses `[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]`. Under `poc_results/` the streaming run writes:

- `poc_default_trials.csv`
- `poc_streaming_steps.csv`
- `poc_epsilon_sweep.csv`
- `poc_cost_vs_epsilon.png`
- `poc_satisfaction_vs_epsilon.png`
- `poc_model_selection_vs_epsilon.png`
- `poc_config.json`

`poc_streaming_steps.csv` contains one row per prompt with the selected model, observed judge-threshold outcome, normalized cost, queue value, and each model's UCB. `poc_default_trials.csv` summarizes the single real-time trajectory, including average cost/satisfaction, whether the target was reached, and model-use frequencies. The epsilon-sweep CSV and plots summarize separate retrospective replays. `poc_config.json` records the configured models, compute values, normalized costs, settings, and feedback source.

### `run_routing.py`

This is the Python orchestration layer called by `run_routing.sh`. It loads the selected prompts once, then for each prompt it requests all configured model answers, judges those answers, appends judged records, refreshes the parsed-vector file, and calls `StreamingRouter.step()` with that prompt's contexts and threshold outcomes. Only after this step does it advance to the next prompt. Once `T` prompts are complete, it writes the per-model `summary.csv`, the streaming trajectory, final POC metrics, and retrospective sweep artifacts.

## Notes

- Install the dependencies listed in the repository root README in the `llm_routing` Conda environment.
- The builder needs network access to Hugging Face Datasets and the configured Lumen endpoint. The parser and comparator can be run separately on existing artifacts without API credentials.
- Every prompt must have exactly one successful judged record for every configured model. Do not reorder rows or mix model orders between prompts.
- The judge-provided feature vectors are real API evaluations; the comparator's satisfaction labels and routing evaluations are simulated by the current POC model.
