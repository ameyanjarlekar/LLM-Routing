import os
import json
import re
import time
import random
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

LUMEN_BASE_URL = "https://lumen.ncsa.illinois.edu/v1"
client = None

# Candidate routing models. The CLI can replace this list.
MODELS = [
    "nemotron-3-super-120b-a12b",
    "deepseek-v4-flash",
    "qwen3.6-35b-a3b",
    "gemma-4-31b-it",
    "qwen3.8-27b",
]

# Cheap judge.
JUDGE_MODEL = "muse-glimmer-30b"

N_QUERIES = 100
SEED = 42

OUTPUT_DIR = Path("llm_routing_poc_data")

RESPONSES_FILE = OUTPUT_DIR / "responses.jsonl"
DATASET_FILE = OUTPUT_DIR / "routing_dataset.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "summary.csv"


# ============================================================
# RETRY WRAPPER
# ============================================================

def call_model(
    model,
    messages,
    max_tokens=1024,
    temperature=0.0,
    max_retries=5,
):
    """
    Call Lumen/OpenAI-compatible endpoint with retries.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            content = response.choices[0].message.content

            usage = None
            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return {
                "success": True,
                "content": content,
                "usage": usage,
            }

        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(
                f"\n[{model}] attempt {attempt + 1}/{max_retries} "
                f"failed: {repr(e)}"
            )
            time.sleep(wait)

    return {
        "success": False,
        "content": None,
        "usage": None,
    }


# ============================================================
# LOAD DIVERSE QUERIES
# ============================================================

def load_queries(n=100, seed=42):
    """
    Sample approximately uniformly across MMLU-Pro disciplines.
    """
    print("Loading MMLU-Pro...")

    ds = load_dataset(
        "TIGER-Lab/MMLU-Pro",
        split="test",
    )

    df = ds.to_pandas()
    if n < 1:
        raise ValueError("The number of prompts must be positive.")
    if n > len(df):
        raise ValueError(
            f"Requested {n} prompts, but the dataset contains only {len(df)}."
        )
    groups = []

    disciplines = sorted(df["category"].dropna().unique())
    rng = random.Random(seed)

    # First pass: one per category where possible.
    for category in disciplines:
        subset = df[df["category"] == category]

        if len(subset) > 0:
            row = subset.sample(
                n=1,
                random_state=rng.randint(0, 10**9),
            ).iloc[0]
            groups.append(row)

    selected = pd.DataFrame(groups)

    # Fill remaining slots uniformly from all categories.
    remaining = n - len(selected)

    if remaining > 0:
        used_indices = set(selected.index)
        remaining_df = df[~df.index.isin(used_indices)]

        extra = remaining_df.sample(
            n=min(remaining, len(remaining_df)),
            random_state=seed,
        )

        selected = pd.concat(
            [selected, extra],
            ignore_index=True,
        )

    selected = selected.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    selected = selected.head(n)

    print("\nSelected categories:")
    print(selected["category"].value_counts())

    return selected


# ============================================================
# FORMAT MMLU-PRO QUESTION
# ============================================================

def format_question(row):
    """
    Convert MMLU-Pro question + choices into a normal prompt.
    """
    question = row["question"]
    options = row["options"]

    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    text = question.strip() + "\n\n"

    for i, option in enumerate(options):
        text += f"{letters[i]}. {option}\n"

    return text


# ============================================================
# JUDGE (Robust key-value extraction)
# ============================================================

JUDGE_SYSTEM_PROMPT = """
You are evaluating an LLM response to a user query.
Rate each dimension on a scale from 0.0 to 1.0.

Provide your evaluation as simple line-by-line numbers like this:

correctness: 0.8
relevance: 1.0
completeness: 0.9
clarity: 0.9
instruction_following: 1.0
safety: 1.0
satisfaction: 0.9
"""

METRIC_KEYS = [
    "correctness",
    "relevance",
    "completeness",
    "clarity",
    "instruction_following",
    "safety",
    "satisfaction",
]

def judge_response(query, response):
    zero_scores = {k: 0.0 for k in METRIC_KEYS}

    if not response:
        return zero_scores

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"USER QUERY:\n{query}\n\nMODEL RESPONSE:\n{response}"},
    ]

    result = call_model(
        model=JUDGE_MODEL,
        messages=messages,
        max_tokens=1500,
        temperature=0.0,
    )

    print(result)
    if not result["success"] or not result["content"]:
        print("\n[DEBUG] Judge API call failed.")
        return zero_scores

    text = result["content"].strip()
    
    # Strip common markdown decorators (*, #, `)
    cleaned_text = re.sub(r"[\*\#\`]", "", text)

    scores = {}
    
    # 1. Flexible Key-Value Extraction
    for key in METRIC_KEYS:
        pattern = rf"{key}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if match:
            scores[key] = float(match.group(1))

    # 2. Fallback: Extract all valid floats between 0.0 and 1.0
    if len(scores) < len(METRIC_KEYS):
        found_floats = [
            float(val) for val in re.findall(r"\b[0-1](?:\.[0-9]+)?\b", cleaned_text)
        ]
        # Map sequential floats if at least 7 are found
        if len(found_floats) >= len(METRIC_KEYS):
            scores = {key: found_floats[i] for i, key in enumerate(METRIC_KEYS)}

    # If still unable to parse, print raw output to inspect
    if len(scores) < len(METRIC_KEYS):
        print("\n[DEBUG PARSE ERROR] Could not extract metrics from judge output:")
        print(f"--- RAW OUTPUT ---\n{text}\n------------------")
        return zero_scores

    # 3. Clamp values between 0.0 and 1.0
    final_scores = {}
    for key in METRIC_KEYS:
        val = scores.get(key, 0.0)
        final_scores[key] = max(0.0, min(1.0, float(val)))

    return final_scores

# ============================================================
# QUALITY VECTOR & SATISFACTION
# ============================================================

def make_quality_vector(judge_scores):
    return [
        judge_scores["correctness"],
        judge_scores["relevance"],
        judge_scores["completeness"],
        judge_scores["clarity"],
        judge_scores["instruction_following"],
        judge_scores["safety"],
    ]


def compute_satisfaction(judge_scores):
    return int(
        judge_scores["correctness"] >= 0.80
        and judge_scores["relevance"] >= 0.80
        and judge_scores["completeness"] >= 0.70
        and judge_scores["safety"] >= 0.90
    )


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and judge responses for a configurable model set."
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model IDs, in the desired routing/action order.",
    )
    parser.add_argument(
        "--n-queries",
        required=True,
        type=int,
        help="Number of prompts to sample from the MMLU-Pro test split.",
    )
    parser.add_argument(
        "--output-dir",
        default="llm_routing_poc_data",
        help="Directory for responses, judged records, and summary CSV.",
    )
    args = parser.parse_args()
    args.models = [model.strip() for model in args.models.split(",")]
    if not args.models or any(not model for model in args.models):
        parser.error("--models must contain one or more non-empty model IDs.")
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicate model IDs.")
    if args.n_queries < 1:
        parser.error("--n-queries must be greater than zero.")
    return args


def main(args=None):
    global client
    global MODELS, N_QUERIES, OUTPUT_DIR
    global RESPONSES_FILE, DATASET_FILE, SUMMARY_FILE

    args = args or parse_args()
    api_key = os.environ.get("LUMEN_API_KEY")
    if not api_key:
        raise SystemExit("Set LUMEN_API_KEY before generating model responses.")
    client = OpenAI(
        base_url=LUMEN_BASE_URL,
        api_key=api_key,
    )
    MODELS = args.models
    N_QUERIES = args.n_queries
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_FILE = OUTPUT_DIR / "responses.jsonl"
    DATASET_FILE = OUTPUT_DIR / "routing_dataset.jsonl"
    SUMMARY_FILE = OUTPUT_DIR / "summary.csv"

    random.seed(SEED)

    queries_df = load_queries(
        n=N_QUERIES,
        seed=SEED,
    )

    # --------------------------------------------------------
    # STEP 1:
    # Load existing responses or generate missing ones.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CHECKING / GENERATING CANDIDATE MODEL RESPONSES")
    print("=" * 80)

    existing_responses = {}
    if RESPONSES_FILE.exists():
        with open(RESPONSES_FILE, "r") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    # Key by (query_id, model) tuple
                    existing_responses[(rec["query_id"], rec["model"])] = rec
        print(f"Loaded {len(existing_responses)} existing responses from {RESPONSES_FILE}")

    all_records = []

    with open(RESPONSES_FILE, "a") as f:
        for query_id, row in tqdm(
            queries_df.iterrows(),
            total=len(queries_df),
        ):
            query = format_question(row)

            for model in MODELS:
                key = (int(query_id), model)

                # Reuse existing response if available
                if (
                    key in existing_responses
                    and existing_responses[key].get("success")
                    and existing_responses[key].get("question") == query
                ):
                    all_records.append(existing_responses[key])
                    continue

                print(
                    f"\nQuery {query_id + 1}/{N_QUERIES}"
                    f" | {model}"
                )

                result = call_model(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": query,
                        }
                    ],
                    max_tokens=512,
                    temperature=0.0,
                )

                record = {
                    "query_id": int(query_id),
                    "category": str(row["category"]),
                    "question": query,
                    "model": model,
                    "response": result["content"],
                    "usage": result["usage"],
                    "success": result["success"],
                }

                f.write(json.dumps(record) + "\n")
                f.flush()

                all_records.append(record)

    successful_pairs = {
        (record["query_id"], record["model"])
        for record in all_records
        if record["success"]
    }
    missing_responses = [
        (query_id, model)
        for query_id in range(N_QUERIES)
        for model in MODELS
        if (query_id, model) not in successful_pairs
    ]
    if missing_responses:
        preview = ", ".join(
            f"query {query_id}: {model}"
            for query_id, model in missing_responses[:10]
        )
        raise RuntimeError(
            f"Could not collect responses for {len(missing_responses)} "
            f"query/model pairs ({preview}). Re-run to retry failed calls."
        )

    # --------------------------------------------------------
    # STEP 2:
    # Judge every response.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("JUDGING RESPONSES")
    print("=" * 80)

    if DATASET_FILE.exists():
        DATASET_FILE.unlink()

    final_records = []

    with open(DATASET_FILE, "a") as f:
            for record in tqdm(
                all_records,
                total=len(all_records),
            ):
                if not record["success"]:
                    continue

                scores = judge_response(
                    query=record["question"],
                    response=record["response"],
                )

                quality_vector = make_quality_vector(scores)
                satisfaction = compute_satisfaction(scores)

                output_record = {
                    "query_id": record.get("query_id"),
                    # Safe key lookup with fallback if 'category' is missing in cached jsonl
                    "category": record.get("category", "unknown"),
                    "question": record.get("question"),
                    "model": record.get("model"),
                    "response": record.get("response"),
                    "judge_metrics": scores,
                    "quality_vector": quality_vector,
                    "Y": satisfaction,
                    "usage": record.get("usage"),
                }

                f.write(json.dumps(output_record) + "\n")
                f.flush()

                final_records.append(output_record)
    # --------------------------------------------------------
    # STEP 3:
    # Summary by model.
    # --------------------------------------------------------

    rows = []

    for model in MODELS:
        model_records = [
            r for r in final_records
            if r["model"] == model
        ]

        if len(model_records) == 0:
            continue

        rows.append({
            "model": model,
            "n": len(model_records),
            "mean_correctness": sum(r["judge_metrics"]["correctness"] for r in model_records) / len(model_records),
            "mean_relevance": sum(r["judge_metrics"]["relevance"] for r in model_records) / len(model_records),
            "mean_completeness": sum(r["judge_metrics"]["completeness"] for r in model_records) / len(model_records),
            "mean_clarity": sum(r["judge_metrics"]["clarity"] for r in model_records) / len(model_records),
            "mean_instruction_following": sum(r["judge_metrics"]["instruction_following"] for r in model_records) / len(model_records),
            "mean_safety": sum(r["judge_metrics"]["safety"] for r in model_records) / len(model_records),
            "mean_satisfaction": sum(r["judge_metrics"]["satisfaction"] for r in model_records) / len(model_records),
            "satisfaction_rate": sum(r["Y"] for r in model_records) / len(model_records),
        })

    summary = pd.DataFrame(rows)

    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    print(summary.to_string(index=False))

    print("\nSaved:")
    print(f"Responses: {RESPONSES_FILE}")
    print(f"Dataset:   {DATASET_FILE}")
    print(f"Summary:   {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
