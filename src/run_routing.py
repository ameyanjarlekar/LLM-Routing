import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

import build_dataset as dataset
import compare_llms as routing
import parse_dataset as parser


def parse_args():
    argument_parser = argparse.ArgumentParser(
        description="Generate, judge, and route one prompt at a time."
    )
    argument_parser.add_argument("--models", required=True)
    argument_parser.add_argument("--compute-values", required=True)
    argument_parser.add_argument("--n-queries", required=True, type=int)
    argument_parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "llm_routing_poc_data"),
    )
    args = argument_parser.parse_args()

    args.models = [model.strip() for model in args.models.split(",")]
    if not args.models or any(not model for model in args.models):
        argument_parser.error("--models must contain non-empty model IDs.")
    if len(set(args.models)) != len(args.models):
        argument_parser.error("--models must not contain duplicates.")
    if args.n_queries < 1:
        argument_parser.error("--n-queries must be greater than zero.")

    try:
        args.compute_values = np.asarray(
            [float(value.strip()) for value in args.compute_values.split(",")],
            dtype=float,
        )
    except ValueError:
        argument_parser.error("--compute-values must contain numeric values.")
    if len(args.compute_values) != len(args.models):
        argument_parser.error("Provide one compute value for each model.")
    if not np.all(np.isfinite(args.compute_values)) or np.any(args.compute_values <= 0):
        argument_parser.error("Compute values must be finite and greater than zero.")
    return args


def load_cached_responses(responses_file):
    records = {}
    if responses_file.exists():
        with responses_file.open("r", encoding="utf-8") as response_stream:
            for line in response_stream:
                if line.strip():
                    record = json.loads(line)
                    records[(record["query_id"], record["model"])] = record
    return records


def read_latest_prompt_vectors(quality_file, query_id, model_count):
    rows = []
    with quality_file.open("r", encoding="utf-8") as vector_stream:
        for line_number, line in enumerate(vector_stream, start=1):
            line = line.strip()
            if not line or line.startswith("-") or "|" not in line:
                continue
            left, right = line.split("|", 1)
            if left.strip().lower() in {"query num/id", "query num", "query id"}:
                continue
            rows.append((int(left.strip()), json.loads(right.strip())))

    current_rows = rows[-model_count:]
    if len(current_rows) != model_count or any(
        row_query_id != query_id for row_query_id, _ in current_rows
    ):
        raise ValueError(
            f"Parsed vectors do not contain exactly {model_count} rows "
            f"for query {query_id}."
        )
    vectors = np.asarray([vector for _, vector in current_rows], dtype=float)
    if vectors.shape != (model_count, routing.D):
        raise ValueError(
            f"Expected vectors with shape {(model_count, routing.D)}, "
            f"got {vectors.shape}."
        )
    if not np.all(np.isfinite(vectors)) or np.any(vectors < 0.0) or np.any(vectors > 1.0):
        raise ValueError("Parsed quality scores must be finite values in [0, 1].")
    return vectors


def make_model_summary(records, models):
    summary_rows = []
    for model in models:
        model_records = [record for record in records if record["model"] == model]
        if not model_records:
            continue
        count = len(model_records)
        row = {"model": model, "n": count}
        for metric in dataset.METRIC_KEYS:
            row[f"mean_{metric}"] = sum(
                record["judge_metrics"][metric] for record in model_records
            ) / count
        row["satisfaction_rate"] = sum(record["Y"] for record in model_records) / count
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def write_results(output_dir, models, compute_values, router, step_rows, vectors, outcomes):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = router.summarize()
    summary_row = {
        "trial": 0,
        "avg_satisfaction": summary["avg_satisfaction"],
        "avg_cost": summary["avg_cost"],
        "required_satisfaction": summary["required_satisfaction"],
        "constraint_satisfied": summary["constraint_satisfied"],
    }
    for index, model in enumerate(models):
        summary_row[f"usage_{model}"] = summary["usage_frequency"][index]

    pd.DataFrame([summary_row]).to_csv(
        output_dir / "poc_default_trials.csv", index=False
    )
    pd.DataFrame(step_rows).to_csv(output_dir / "poc_streaming_steps.csv", index=False)

    epsilon_results = routing.run_epsilon_sweep(
        X=np.asarray(vectors, dtype=float),
        Y=np.asarray(outcomes, dtype=int),
    )
    routing.plot_main_poc_figures(epsilon_results)

    configuration = {
        "models": models,
        "compute_values": compute_values.tolist(),
        "costs": routing.COSTS.tolist(),
        "D": routing.D,
        "epsilon": routing.EPSILON,
        "delta": routing.DELTA,
        "T": len(step_rows),
        "N_TRIALS": routing.N_TRIALS,
        "eta_reference": routing.ETA_REFERENCE,
        "rho_reference": routing.RHO_REFERENCE,
        "T_reference": routing.T_REFERENCE,
        "alpha_scale": routing.ALPHA_SCALE,
        "beta": routing.BETA,
        "satisfaction_feedback": "judge threshold for the selected model",
        "context_normalization": "x = raw_quality_vector / sqrt(D)",
        "routing_order": "sequential prompt order",
        "fallback": False,
        "main_figures": [
            "poc_cost_vs_epsilon.png",
            "poc_satisfaction_vs_epsilon.png",
            "poc_model_selection_vs_epsilon.png",
        ],
    }
    with (output_dir / "poc_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(configuration, config_file, indent=2)

    print("\nStreaming run summary:")
    print(pd.DataFrame([summary_row]).to_string(index=False))
    print(f"\nStep-by-step decisions: {output_dir / 'poc_streaming_steps.csv'}")
    print(f"Final POC results: {output_dir}")


def main(args=None):
    args = args or parse_args()
    api_key = os.environ.get("LUMEN_API_KEY")
    if not api_key:
        raise SystemExit("Set LUMEN_API_KEY before generating model responses.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_file = output_dir / "responses.jsonl"
    dataset_file = output_dir / "routing_dataset.jsonl"
    quality_file = output_dir / "parsed_quality_vectors.txt"
    summary_file = output_dir / "summary.csv"
    results_dir = output_dir / "poc_results"

    dataset.client = OpenAI(base_url=dataset.LUMEN_BASE_URL, api_key=api_key)
    dataset.MODELS = args.models
    dataset.N_QUERIES = args.n_queries
    dataset.OUTPUT_DIR = output_dir
    dataset.RESPONSES_FILE = responses_file
    dataset.DATASET_FILE = dataset_file
    dataset.SUMMARY_FILE = summary_file

    routing.MODELS = args.models
    routing.K = len(args.models)
    routing.COMPUTE_VALUES = args.compute_values
    routing.COSTS = args.compute_values / np.max(args.compute_values)
    routing.OUTPUT_DIR = results_dir
    routing.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    queries = dataset.load_queries(n=args.n_queries, seed=dataset.SEED)
    if len(queries) != args.n_queries:
        raise RuntimeError(
            f"Requested {args.n_queries} prompts but selected {len(queries)}."
        )

    cached_responses = load_cached_responses(responses_file)
    learner = routing.StreamingRouter(
        horizon=args.n_queries,
        epsilon=routing.EPSILON,
    )
    all_judged_records = []
    step_rows = []
    contexts = []
    outcomes = []

    with dataset_file.open("w", encoding="utf-8") as judged_stream, responses_file.open(
        "a", encoding="utf-8"
    ) as responses_stream:
        for query_id, row in queries.iterrows():
            prompt = dataset.format_question(row)
            prompt_responses = []

            for model in args.models:
                cache_key = (int(query_id), model)
                cached = cached_responses.get(cache_key)
                if (
                    cached
                    and cached.get("success")
                    and cached.get("question") == prompt
                ):
                    response_record = cached
                else:
                    result = dataset.call_model(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512,
                        temperature=0.0,
                    )
                    response_record = {
                        "query_id": int(query_id),
                        "category": str(row["category"]),
                        "question": prompt,
                        "model": model,
                        "response": result["content"],
                        "usage": result["usage"],
                        "success": result["success"],
                    }
                    responses_stream.write(json.dumps(response_record) + "\n")
                    responses_stream.flush()
                    cached_responses[cache_key] = response_record

                if not response_record.get("success"):
                    raise RuntimeError(
                        f"Response failed for query {query_id}, model {model}; "
                        "rerun to retry this query/model pair."
                    )
                prompt_responses.append(response_record)

            prompt_vectors = []
            prompt_outcomes = []
            for response_record in prompt_responses:
                scores = dataset.judge_response(
                    query=prompt,
                    response=response_record["response"],
                )
                quality_vector = dataset.make_quality_vector(scores)
                satisfaction = dataset.compute_satisfaction(scores)
                judged_record = {
                    "query_id": int(query_id),
                    "category": response_record["category"],
                    "question": prompt,
                    "model": response_record["model"],
                    "response": response_record["response"],
                    "judge_metrics": scores,
                    "quality_vector": quality_vector,
                    "Y": satisfaction,
                    "usage": response_record["usage"],
                }
                judged_stream.write(json.dumps(judged_record) + "\n")
                judged_stream.flush()
                all_judged_records.append(judged_record)
                prompt_vectors.append(quality_vector)
                prompt_outcomes.append(satisfaction)

            parser.extract_quality_vectors(
                str(dataset_file),
                str(quality_file),
            )
            raw_contexts = read_latest_prompt_vectors(
                quality_file,
                int(query_id),
                len(args.models),
            )
            contexts.append(raw_contexts / np.sqrt(routing.D))
            outcomes.append(prompt_outcomes)

            step = learner.step(contexts[-1], prompt_outcomes)
            selected_model = args.models[step["selected_action"]]
            step_row = {
                "step": step["step"],
                "query_id": int(query_id),
                "selected_action": step["selected_action"],
                "selected_model": selected_model,
                "observed_satisfaction": step["observed_y"],
                "selected_model_compute": args.compute_values[step["selected_action"]],
                "normalized_cost": step["cost"],
                "lambda": step["lambda"],
            }
            for index, model in enumerate(args.models):
                step_row[f"ucb_{model}"] = step["ucbs"][index]
            step_rows.append(step_row)
            print(
                f"Prompt {query_id + 1}/{args.n_queries}: selected "
                f"{selected_model}; satisfaction={step['observed_y']}; "
                f"cost={step['cost']:.4f}"
            )

    make_model_summary(all_judged_records, args.models).to_csv(
        summary_file,
        index=False,
    )
    write_results(
        results_dir,
        args.models,
        args.compute_values,
        learner,
        step_rows,
        contexts,
        outcomes,
    )


if __name__ == "__main__":
    main()