import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================================================================
# REAL-LLM ROUTING POC
# ================================================================
#
# Five real LLMs are the complete action set.
#
# THERE IS NO FALLBACK.
#
# Each real LLM produces a six-dimensional quality vector:
#
#   [correctness,
#    relevance,
#    completeness,
#    clarity,
#    conciseness,
#    safety]
#
# These quality vectors are used as contextual features.
#
# The satisfaction model is:
#
#       Y_{t,a} ~ Bernoulli(theta*^T x_{t,a})
#
# where theta* is:
#
#   - unknown to the learner
#   - fixed throughout the experiment
#   - all coordinates strictly positive
#   - ||theta*||_2 = 1
#
# To ensure theta*^T x is always a valid probability, the
# quality vectors are normalized so that ||x||_2 <= 1.
#
# The learner only observes Y for the model it actually selects.
#
#
# MAIN POC FIGURES
# ----------------
#
# 1. Average cost vs epsilon
#
# 2. Mean satisfaction vs epsilon
#    with required satisfaction 1 - epsilon
#
# 3. Model-selection proportions vs epsilon
#    using 100% stacked bars
#
# ================================================================


# ================================================================
# PATHS
# ================================================================

DATA_DIR = (
    "/home/ameya-anjarlekar/"
    "llm-routing-poc/"
    "llm_routing_poc_data"
)

QUALITY_FILE = os.path.join(
    DATA_DIR,
    "parsed_quality_vectors.txt",
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "poc_results",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ================================================================
# MODELS
# ================================================================

MODELS = [
    "nemotron-3-super-120b-a12b",
    "deepseek-v4-flash",
    "qwen3.6-35b-a3b",
    "gemma-4-31b-it",
    "qwen3.8-27b",
]

K = len(MODELS)


# ================================================================
# MODEL PARAMETER COUNTS
# ================================================================
#
# Costs are proportional to total parameter count.
#
# Nemotron: 120B
# DeepSeek: 284B
# Qwen3.6: 35B
# Gemma4: 30.7B
# Qwen3.8: 27B
#
# Normalized by the largest model.
# ================================================================

PARAM_COUNTS_B = np.array(
    [
        120.0,
        284.0,
        35.0,
        30.7,
        27.0,
    ],
    dtype=float,
)

COSTS = (
    PARAM_COUNTS_B
    / np.max(PARAM_COUNTS_B)
)


# ================================================================
# PROBLEM PARAMETERS
# ================================================================

D = 6

EPSILON = 0.20
DELTA = 0.05

S = 1.0

# ------------------------------------------------
# Reference eta.
#
# At epsilon = 0.20:
#
#     eta = 0.50
#
# For other epsilon:
#
#     eta(epsilon)
#       = 0.50 * epsilon / 0.20
#
# Hence:
#
#     epsilon = 0.20 -> eta = 0.500
#     epsilon = 0.25 -> eta = 0.625
#     epsilon = 0.30 -> eta = 0.750
#     epsilon = 0.35 -> eta = 0.875
#     epsilon = 0.40 -> eta = 1.000
#
# ------------------------------------------------

ETA_REFERENCE = 0.5

# ------------------------------------------------
# Rho reference.
# ------------------------------------------------

RHO_REFERENCE = 0.15

# ------------------------------------------------
# Reference horizon for rho scaling.
# ------------------------------------------------

T_REFERENCE = 100

# ------------------------------------------------
# Same empirical UCB scale as synthetic experiments.
# ------------------------------------------------

ALPHA_SCALE = 0.005

# ------------------------------------------------
# Regularization.
# ------------------------------------------------

BETA = 1.0

# ------------------------------------------------
# Confidence-radius parameter.
# ------------------------------------------------

ZETA = 2.5

# ------------------------------------------------
# Number of random query-order replays.
# ------------------------------------------------

N_TRIALS = 100

RANDOM_SEED = 12345


# ================================================================
# HIDDEN THETA*
# ================================================================
#
# We want:
#
#   1. Every coordinate > 0
#   2. ||theta*||_2 = 1
#
# We deliberately do NOT normalize theta to sum to 1.
#
# Because ||theta*||_2 = 1 and contexts satisfy ||x||_2 <= 1,
#
#       theta*^T x <= 1
#
# by Cauchy-Schwarz.
#
# ================================================================

THETA_STAR = np.array(
    [
        0.60,
        0.50,
        0.40,
        0.30,
        0.20,
        0.10,
    ],
    dtype=float,
)

THETA_STAR = (
    THETA_STAR
    / np.linalg.norm(THETA_STAR)
)


# ================================================================
# THETA* VALIDATION
# ================================================================

assert np.all(
    THETA_STAR > 0
)

assert np.isclose(
    np.linalg.norm(
        THETA_STAR
    ),
    1.0,
)

assert np.isclose(
    np.linalg.norm(
        THETA_STAR
    ),
    S,
)


# ================================================================
# ETA
# ================================================================

def compute_eta(
    epsilon,
):
    """
    Epsilon-dependent primal step size.

    Reference point:

        epsilon = 0.20
        eta     = 0.50

    Therefore:

        eta(epsilon)
            = 0.50 * epsilon / 0.20
    """

    return (
        ETA_REFERENCE
    )


# ================================================================
# RHO
# ================================================================

def compute_rho(
    T,
):

    return (
        RHO_REFERENCE
        * math.sqrt(
            T_REFERENCE / T
        )
    )


# ================================================================
# LOAD QUALITY VECTORS
# ================================================================

def load_quality_vectors(
    filename,
):
    """
    File format:

        Query Num/ID | Model Quality Vector / Scores
        0 | [1.0, 1.0, 0.3, 1.0, 0.5, 1.0]
        0 | [1.0, 1.0, 0.6, 1.0, 0.9, 1.0]
        ...

    Five consecutive rows correspond to the five models.

    Returns
    -------
    X:
        Normalized contextual vectors.
        Shape = (T, K, D)

    Y:
        Bernoulli satisfaction outcomes.
        Shape = (T, K)

    query_ids:
        Shape = (T,)

    probabilities:
        True Bernoulli probabilities.
        Shape = (T, K)

    X_raw:
        Original quality vectors.
        Shape = (T, K, D)
    """

    raw_rows = []

    query_ids = []

    # ------------------------------------------------------------
    # Read file.
    # ------------------------------------------------------------

    with open(
        filename,
        "r",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            if line.startswith("-"):
                continue

            if "|" not in line:
                continue

            left, right = line.split(
                "|",
                1,
            )

            left = left.strip()
            right = right.strip()

            # ----------------------------------------------------
            # Skip header.
            # ----------------------------------------------------

            if left.lower() in {
                "query num/id",
                "query num",
                "query id",
            }:
                continue

            # ----------------------------------------------------
            # Query ID.
            # ----------------------------------------------------

            try:

                query_id = int(left)

            except ValueError as e:

                raise ValueError(
                    f"Could not parse query ID on "
                    f"line {line_number}: {left!r}"
                ) from e

            # ----------------------------------------------------
            # Quality vector.
            # ----------------------------------------------------

            try:

                vector = json.loads(
                    right
                )

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"Could not parse quality vector "
                    f"on line {line_number}: "
                    f"{right!r}"
                ) from e

            if not isinstance(
                vector,
                list,
            ):

                raise ValueError(
                    f"Line {line_number}: expected "
                    f"a list."
                )

            if len(vector) != D:

                raise ValueError(
                    f"Line {line_number}: expected "
                    f"{D} values, got "
                    f"{len(vector)}."
                )

            vector = np.asarray(
                vector,
                dtype=float,
            )

            raw_rows.append(
                vector
            )

            query_ids.append(
                query_id
            )

    # ------------------------------------------------------------
    # Basic validation.
    # ------------------------------------------------------------

    if len(raw_rows) == 0:

        raise ValueError(
            "No quality vectors found."
        )

    raw_rows = np.asarray(
        raw_rows,
        dtype=float,
    )

    query_ids = np.asarray(
        query_ids,
        dtype=int,
    )

    if len(raw_rows) % K != 0:

        raise ValueError(
            f"Found {len(raw_rows)} rows, "
            f"which is not divisible by "
            f"K={K}."
        )

    T = len(raw_rows) // K

    # ------------------------------------------------------------
    # Raw quality vectors.
    # ------------------------------------------------------------

    X_raw = raw_rows.reshape(
        T,
        K,
        D,
    )

    query_ids_matrix = query_ids.reshape(
        T,
        K,
    )

    # ------------------------------------------------------------
    # Verify query IDs.
    # ------------------------------------------------------------

    for t in range(T):

        if not np.all(
            query_ids_matrix[t]
            == query_ids_matrix[t, 0]
        ):

            raise ValueError(
                f"Query block {t} has inconsistent "
                f"IDs: {query_ids_matrix[t]}"
            )

    query_ids = (
        query_ids_matrix[:, 0]
    )

    # ------------------------------------------------------------
    # Validate raw quality scores.
    # ------------------------------------------------------------

    if not np.all(
        (X_raw >= 0.0)
        & (X_raw <= 1.0)
    ):

        bad = np.argwhere(
            (X_raw < 0.0)
            | (X_raw > 1.0)
        )[0]

        raise ValueError(
            f"Quality score outside [0,1] at "
            f"query={bad[0]}, "
            f"model={bad[1]}, "
            f"dimension={bad[2]}"
        )

    # ============================================================
    # CONTEXT NORMALIZATION
    # ============================================================
    #
    # Raw quality vector:
    #
    #       q in [0,1]^6
    #
    # We transform:
    #
    #       x = q / sqrt(D)
    #
    # Since ||q||_2 <= sqrt(D),
    #
    #       ||x||_2 <= 1.
    #
    # Therefore, because ||theta*||_2 = 1:
    #
    #       theta*^T x <= 1.
    #
    # ============================================================

    X = (
        X_raw
        / math.sqrt(D)
    )

    # ------------------------------------------------------------
    # Verify context norm.
    # ------------------------------------------------------------

    context_norms = np.linalg.norm(
        X,
        axis=2,
    )

    assert np.max(
        context_norms
    ) <= 1.0 + 1e-12

    # ============================================================
    # TRUE SATISFACTION PROBABILITIES
    # ============================================================

    probabilities = np.einsum(
        "d,tad->ta",
        THETA_STAR,
        X,
    )

    # ------------------------------------------------------------
    # Numerical validation.
    # ------------------------------------------------------------

    min_probability = np.min(
        probabilities
    )

    max_probability = np.max(
        probabilities
    )

    if min_probability < -1e-12:

        raise ValueError(
            "Negative Bernoulli probability "
            f"detected: {min_probability}"
        )

    if max_probability > 1.0 + 1e-12:

        raise ValueError(
            "Bernoulli probability > 1 detected: "
            f"{max_probability}"
        )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    # ============================================================
    # GENERATE OBSERVED Y
    # ============================================================
    #
    # Each query-model pair gets:
    #
    #       Y_{t,a} ~ Bernoulli(p_{t,a})
    #
    # This is generated ONCE.
    #
    # The same realized Y matrix is then used across all
    # query-order replay trials.
    #
    # ============================================================

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    Y = (
        rng.random(
            size=probabilities.shape
        )
        < probabilities
    ).astype(int)

    return (
        X,
        Y,
        query_ids,
        probabilities,
        X_raw,
    )


# ================================================================
# PRINT THETA*
# ================================================================

def print_theta_information():

    print()
    print("=" * 80)
    print("HIDDEN THETA*")
    print("=" * 80)

    print(
        "theta* =",
        THETA_STAR,
    )

    print(
        "||theta*|| =",
        np.linalg.norm(
            THETA_STAR
        ),
    )

    print(
        "all coordinates positive =",
        np.all(
            THETA_STAR > 0
        ),
    )

    print(
        "S =",
        S,
    )


# ================================================================
# COMPUTE KAPPA
# ================================================================

def compute_kappa():

    return 1.0


# ================================================================
# THEORETICAL ALPHA
# ================================================================

def compute_theoretical_alpha(
    t,
):

    if t <= 0:
        t = 1

    L = 1.0

    kappa = compute_kappa()

    delta_E = (
        t ** (-ZETA)
    )

    term1 = (
        0.5
        * math.sqrt(
            D
            * math.log(
                1.0
                + (
                    t
                    * L
                    * L
                    / (D * BETA)
                )
            )
            + 2.0
            * math.log(
                2.0 / delta_E
            )
        )
    )

    term2 = (
        math.sqrt(BETA)
        * S
    )

    alpha = (
        term1
        + term2
    ) / kappa

    return alpha


# ================================================================
# COMPUTE UCB
# ================================================================

def compute_ucbs(
    X_t,
    theta_hat,
    V_inv,
    t,
):
    """
    Linear Bernoulli model:

        p = theta^T x

    Therefore:

        UCB = x^T theta_hat
              + alpha ||x||_{V^{-1}}.

    The UCB itself is intentionally NOT clipped.
    """

    alpha = (
        ALPHA_SCALE
        * compute_theoretical_alpha(
            t
        )
    )

    ucbs = np.zeros(K)

    for a in range(K):

        x = X_t[a]

        mean = float(
            np.dot(
                x,
                theta_hat,
            )
        )

        # Mean probability must lie in [0,1].
        mean = np.clip(
            mean,
            0.0,
            1.0,
        )

        uncertainty = math.sqrt(
            max(
                0.0,
                np.dot(
                    x,
                    np.dot(
                        V_inv,
                        x,
                    ),
                ),
            )
        )

        # --------------------------------------------------------
        # IMPORTANT:
        # UCB is intentionally NOT clipped at 1.
        # --------------------------------------------------------

        ucbs[a] = (
            mean
            + alpha
            * uncertainty
        )

    return ucbs


# ================================================================
# SINGLE ONLINE TRIAL
# ================================================================

def run_online_trial(
    X,
    Y,
    epsilon,
    delta,
    query_order,
    use_queue=True,
):
    """
    Online routing with partial feedback.

    The learner only observes Y for the selected action.

    There is no fallback.
    """

    T = len(
        query_order
    )

    # ------------------------------------------------------------
    # Epsilon-dependent eta.
    # ------------------------------------------------------------

    eta = compute_eta(
        epsilon
    )

    # ------------------------------------------------------------
    # Horizon-scaled rho.
    # ------------------------------------------------------------

    rho = compute_rho(
        T
    )

    # ------------------------------------------------------------
    # Regularized design matrix.
    # ------------------------------------------------------------

    V = (
        BETA
        * np.eye(D)
    )

    b = np.zeros(D)

    theta_hat = np.zeros(D)

    # ------------------------------------------------------------
    # Dual queue.
    # ------------------------------------------------------------

    lam = 0.0

    # ------------------------------------------------------------
    # Logging.
    # ------------------------------------------------------------

    selected_actions = np.zeros(
        T,
        dtype=int,
    )

    observed_y = np.zeros(
        T,
        dtype=int,
    )

    observed_cost = np.zeros(
        T,
        dtype=float,
    )

    lambda_history = np.zeros(T)

    satisfaction_history = np.zeros(T)

    cost_history = np.zeros(T)

    ucb_history = np.zeros(
        (T, K)
    )

    # ============================================================
    # ONLINE LOOP
    # ============================================================

    for t_idx in range(T):

        t = t_idx + 1

        query_idx = (
            query_order[t_idx]
        )

        X_t = X[query_idx]

        # --------------------------------------------------------
        # Confidence matrix.
        # --------------------------------------------------------

        V_inv = np.linalg.inv(
            V
        )

        # --------------------------------------------------------
        # Compute UCB.
        # --------------------------------------------------------

        ucbs = compute_ucbs(
            X_t=X_t,
            theta_hat=theta_hat,
            V_inv=V_inv,
            t=t,
        )

        ucb_history[
            t_idx
        ] = ucbs

        # --------------------------------------------------------
        # Primal-dual action selection.
        #
        #       argmin_a
        #           c(a) - lambda_t UCB(a)
        #
        # NO FALLBACK.
        # --------------------------------------------------------

        if use_queue:

            multiplier = lam

        else:

            # No-queue ablation.
            multiplier = 1.0

        scores = (
            COSTS
            - multiplier
            * ucbs
        )

        action = int(
            np.argmin(scores)
        )

        # --------------------------------------------------------
        # Observe ONLY selected action.
        # --------------------------------------------------------

        y = int(
            Y[
                query_idx,
                action,
            ]
        )

        cost = float(
            COSTS[action]
        )

        selected_actions[
            t_idx
        ] = action

        observed_y[
            t_idx
        ] = y

        observed_cost[
            t_idx
        ] = cost

        # --------------------------------------------------------
        # Dual queue update.
        # --------------------------------------------------------

        if use_queue:

            lam = max(
                0.0,
                lam
                + eta
                * (
                    (1.0 - epsilon)
                    + rho
                    - y
                ),
            )

        lambda_history[
            t_idx
        ] = lam

        # --------------------------------------------------------
        # Update linear estimator using ONLY selected action.
        # --------------------------------------------------------

        x_selected = X_t[
            action
        ]

        V += np.outer(
            x_selected,
            x_selected,
        )

        b += (
            y
            * x_selected
        )

        theta_hat = np.linalg.solve(
            V,
            b,
        )

        # --------------------------------------------------------
        # Log.
        # --------------------------------------------------------

        satisfaction_history[
            t_idx
        ] = y

        cost_history[
            t_idx
        ] = cost

    # ============================================================
    # AGGREGATES
    # ============================================================

    avg_satisfaction = float(
        np.mean(
            observed_y
        )
    )

    avg_cost = float(
        np.mean(
            observed_cost
        )
    )

    required_satisfaction = (
        1.0 - epsilon
    )

    constraint_satisfied = (
        avg_satisfaction
        >= required_satisfaction
    )

    usage_counts = np.bincount(
        selected_actions,
        minlength=K,
    )

    usage_frequency = (
        usage_counts
        / T
    )

    return {
        "avg_satisfaction":
            avg_satisfaction,

        "avg_cost":
            avg_cost,

        "required_satisfaction":
            required_satisfaction,

        "constraint_satisfied":
            bool(
                constraint_satisfied
            ),

        "selected_actions":
            selected_actions,

        "observed_y":
            observed_y,

        "observed_cost":
            observed_cost,

        "lambda_history":
            lambda_history,

        "satisfaction_history":
            satisfaction_history,

        "cost_history":
            cost_history,

        "ucb_history":
            ucb_history,

        "usage_counts":
            usage_counts,

        "usage_frequency":
            usage_frequency,

        "eta":
            eta,

        "rho":
            rho,

        "final_theta_hat":
            theta_hat,
    }


# ================================================================
# RUN MULTIPLE QUERY-ORDER REPLAYS
# ================================================================

def run_replays(
    X,
    Y,
    epsilon,
    delta,
    n_trials=N_TRIALS,
):

    T = X.shape[0]

    rng = np.random.default_rng(
        RANDOM_SEED + 1
    )

    results = []

    for trial in range(
        n_trials
    ):

        query_order = (
            rng.permutation(T)
        )

        result = run_online_trial(
            X=X,
            Y=Y,
            epsilon=epsilon,
            delta=delta,
            query_order=query_order,
            use_queue=True,
        )

        result["trial"] = (
            trial
        )

        results.append(
            result
        )

    return results


# ================================================================
# SUMMARIZE REPLAYS
# ================================================================

def summarize_replays(
    results,
):

    satisfaction = np.array(
        [
            r["avg_satisfaction"]
            for r in results
        ]
    )

    costs = np.array(
        [
            r["avg_cost"]
            for r in results
        ]
    )

    feasible = np.array(
        [
            r[
                "constraint_satisfied"
            ]
            for r in results
        ],
        dtype=float,
    )

    usage = np.array(
        [
            r["usage_frequency"]
            for r in results
        ]
    )

    return {
        "mean_satisfaction":
            float(
                np.mean(
                    satisfaction
                )
            ),

        "std_satisfaction":
            float(
                np.std(
                    satisfaction,
                    ddof=1,
                )
            ),

        "mean_cost":
            float(
                np.mean(
                    costs
                )
            ),

        "std_cost":
            float(
                np.std(
                    costs,
                    ddof=1,
                )
            ),

        "constraint_probability":
            float(
                np.mean(
                    feasible
                )
            ),

        "mean_usage":
            np.mean(
                usage,
                axis=0,
            ),

        "std_usage":
            np.std(
                usage,
                axis=0,
                ddof=1,
            ),
    }


# ================================================================
# DEFAULT POC
# ================================================================

def run_default_poc(
    X,
    Y,
    true_probabilities,
):

    T = X.shape[0]

    print()
    print("=" * 80)
    print("REAL-LLM POC")
    print("=" * 80)

    print()
    print(
        "Number of queries:",
        T,
    )

    print(
        "Number of models:",
        K,
    )

    print(
        "Feature dimension:",
        D,
    )

    print()
    print(
        "Models and costs:"
    )

    for a, model in enumerate(
        MODELS
    ):

        print(
            f"  {a}: "
            f"{model:35s} "
            f"params={PARAM_COUNTS_B[a]:6.1f}B "
            f"cost={COSTS[a]:.6f}"
        )

    print()
    print(
        "epsilon =",
        EPSILON,
    )

    print(
        "delta   =",
        DELTA,
    )

    print()
    print(
        "eta =",
        compute_eta(
            EPSILON
        ),
    )

    print(
        "rho =",
        compute_rho(
            T
        ),
    )

    # ------------------------------------------------------------
    # True satisfaction probabilities.
    # ------------------------------------------------------------

    print()
    print(
        "True mean satisfaction probabilities:"
    )

    for a, model in enumerate(
        MODELS
    ):

        print(
            f"  {model:35s}: "
            f"{np.mean(true_probabilities[:, a]):.4f}"
        )

    print()
    print(
        "Overall true mean probability:",
        f"{np.mean(true_probabilities):.4f}",
    )

    # ------------------------------------------------------------
    # Realized Y.
    # ------------------------------------------------------------

    print()
    print(
        "Overall realized Bernoulli "
        "satisfaction:",
        f"{np.mean(Y):.4f}",
    )

    # ------------------------------------------------------------
    # Replays.
    # ------------------------------------------------------------

    results = run_replays(
        X=X,
        Y=Y,
        epsilon=EPSILON,
        delta=DELTA,
        n_trials=N_TRIALS,
    )

    summary = summarize_replays(
        results
    )

    # ------------------------------------------------------------
    # Results.
    # ------------------------------------------------------------

    print()
    print("-" * 80)
    print("DEFAULT RESULT")
    print("-" * 80)

    print(
        "Average satisfaction:",
        f"{summary['mean_satisfaction']:.4f}",
    )

    print(
        "Required satisfaction:",
        f"{1.0 - EPSILON:.4f}",
    )

    print(
        "Average cost:",
        f"{summary['mean_cost']:.4f}",
    )

    print(
        "Fraction of query-order replays "
        "satisfying target:",
        f"{summary['constraint_probability']:.4f}",
    )

    # ------------------------------------------------------------
    # Save trial CSV.
    # ------------------------------------------------------------

    rows = []

    for result in results:

        row = {
            "trial":
                result["trial"],

            "avg_satisfaction":
                result[
                    "avg_satisfaction"
                ],

            "avg_cost":
                result[
                    "avg_cost"
                ],

            "required_satisfaction":
                result[
                    "required_satisfaction"
                ],

            "constraint_satisfied":
                result[
                    "constraint_satisfied"
                ],
        }

        for a, model in enumerate(
            MODELS
        ):

            row[
                f"usage_{model}"
            ] = result[
                "usage_frequency"
            ][a]

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "poc_default_trials.csv",
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        "Saved:",
        output_file,
    )

    return (
        results,
        summary,
    )


# ================================================================
# EPSILON SWEEP
# ================================================================

def run_epsilon_sweep(
    X,
    Y,
):

    epsilon_values = [0.1,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
    ]

    rows = []

    print()
    print("=" * 80)
    print("EPSILON SWEEP")
    print("=" * 80)

    for epsilon in (
        epsilon_values
    ):

        eta = compute_eta(
            epsilon
        )

        print(
            f"Running epsilon={epsilon:.2f} "
            f"(eta={eta:.4f})"
        )

        results = run_replays(
            X=X,
            Y=Y,
            epsilon=epsilon,
            delta=DELTA,
            n_trials=N_TRIALS,
        )

        summary = summarize_replays(
            results
        )

        row = {
            "epsilon":
                epsilon,

            "eta":
                eta,

            "required_satisfaction":
                1.0 - epsilon,

            "mean_satisfaction":
                summary[
                    "mean_satisfaction"
                ],

            "std_satisfaction":
                summary[
                    "std_satisfaction"
                ],

            "mean_cost":
                summary[
                    "mean_cost"
                ],

            "std_cost":
                summary[
                    "std_cost"
                ],

            "constraint_probability":
                summary[
                    "constraint_probability"
                ],
        }

        # --------------------------------------------------------
        # Model selection proportions.
        # --------------------------------------------------------

        for a, model in enumerate(
            MODELS
        ):

            row[
                f"usage_{model}"
            ] = summary[
                "mean_usage"
            ][a]

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "poc_epsilon_sweep.csv",
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print()
    print(df)

    print()
    print(
        "Saved:",
        output_file,
    )

    return df


# ================================================================
# FIGURE 1:
# AVERAGE COST VS EPSILON
# ================================================================

def plot_epsilon_cost(
    df,
):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        df["epsilon"],
        df["mean_cost"],
        marker="o",
        linewidth=2,
    )

    plt.xlabel(
        r"$\epsilon$"
    )

    plt.ylabel(
        "Average cost"
    )

    plt.title(
        "Average Routing Cost vs. Constraint"
    )

    plt.xticks(
        df["epsilon"]
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        "poc_cost_vs_epsilon.png",
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        output_file,
    )


# ================================================================
# FIGURE 2:
# MEAN SATISFACTION VS EPSILON
# ================================================================

def plot_epsilon_satisfaction(
    df,
):

    plt.figure(
        figsize=(8, 5)
    )

    # ------------------------------------------------------------
    # Mean satisfaction.
    # ------------------------------------------------------------

    plt.plot(
        df["epsilon"],
        df["mean_satisfaction"],
        marker="o",
        linewidth=2,
        label="Mean satisfaction",
    )

    # ------------------------------------------------------------
    # Required satisfaction.
    # ------------------------------------------------------------

    plt.plot(
        df["epsilon"],
        df[
            "required_satisfaction"
        ],
        marker="s",
        linestyle="--",
        linewidth=1.5,
        label=r"Required $1-\epsilon$",
    )

    plt.xlabel(
        r"$\epsilon$"
    )

    plt.ylabel(
        "Satisfaction"
    )

    plt.title(
        "Mean Satisfaction vs. Constraint"
    )

    plt.xticks(
        df["epsilon"]
    )

    plt.ylim(
        0,
        1.05,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        "poc_satisfaction_vs_epsilon.png",
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        output_file,
    )


# ================================================================
# FIGURE 3:
# MODEL SELECTION PROPORTIONS VS EPSILON
# ================================================================
#
# Each epsilon gets one bar.
#
# Each bar sums to 100%.
#
# The segments show the fraction of queries routed to each model.
#
# ================================================================

def plot_model_selection_vs_epsilon(df):
    model_labels = [
        "Nemotron",
        "DeepSeek",
        "Qwen3.6",
        "Gemma4",
        "Qwen3.8",
    ]

    model_usage_columns = [
        f"usage_{model}" for model in MODELS
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    bottom = np.zeros(len(df))
    x = np.arange(len(df))

    # ------------------------------------------------------------
    # 100% stacked bars
    # ------------------------------------------------------------
    for model_label, usage_column in zip(
        model_labels,
        model_usage_columns,
    ):
        values = df[usage_column].to_numpy() * 100.0

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=model_label,
            width=0.65,
        )

        bottom += values

    # ------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("Model selection proportion (%)")
    ax.set_title("Model Selection vs. Constraint")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{epsilon:.2f}" for epsilon in df["epsilon"]]
    )

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # ------------------------------------------------------------
    # Percentage table below the plot
    # ------------------------------------------------------------
    table_data = []

    for model_label, usage_column in zip(
        model_labels,
        model_usage_columns,
    ):
        values = df[usage_column].to_numpy() * 100.0

        table_data.append([
            f"{v:.1f}%" if v >= 0.05 else "<0.1%"
            for v in values
        ])

    table = ax.table(
        cellText=table_data,
        rowLabels=model_labels,
        colLabels=[
            f"$\\epsilon$={epsilon:.2f}"
            for epsilon in df["epsilon"]
        ],
        cellLoc="center",
        rowLoc="center",
        loc="bottom",
        bbox=[0.0, -0.48, 1.0, 0.38],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # ------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.56),
        ncol=5,
        frameon=False,
    )

    plt.subplots_adjust(
        bottom=0.38,
        top=0.90,
        left=0.10,
        right=0.98,
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "poc_model_selection_vs_epsilon.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path

# ================================================================
# MAIN POC FIGURES
# ================================================================

def plot_main_poc_figures(
    epsilon_df,
):

    print()
    print("=" * 80)
    print("GENERATING MAIN POC FIGURES")
    print("=" * 80)

    # Figure 1.
    plot_epsilon_cost(
        epsilon_df
    )

    # Figure 2.
    plot_epsilon_satisfaction(
        epsilon_df
    )

    # Figure 3.
    plot_model_selection_vs_epsilon(
        epsilon_df
    )


# ================================================================
# MAIN
# ================================================================

def main():

    # ============================================================
    # LOAD DATA
    # ============================================================

    print()
    print("=" * 80)
    print("LOADING REAL-LLM DATA")
    print("=" * 80)

    (
        X,
        Y,
        query_ids,
        true_probabilities,
        X_raw,
    ) = load_quality_vectors(
        QUALITY_FILE
    )

    T = X.shape[0]

    print(
        "X shape:",
        X.shape,
    )

    print(
        "Y shape:",
        Y.shape,
    )

    print(
        "Number of queries:",
        T,
    )

    print(
        "Number of models:",
        K,
    )

    print(
        "Feature dimension:",
        D,
    )

    # ============================================================
    # VALIDATE
    # ============================================================

    assert X.shape == (
        T,
        K,
        D,
    )

    assert Y.shape == (
        T,
        K,
    )

    assert np.all(
        X >= 0.0
    )

    assert np.all(
        X <= 1.0
    )

    assert np.all(
        (Y == 0)
        | (Y == 1)
    )

    # ------------------------------------------------------------
    # Context norm.
    # ------------------------------------------------------------

    context_norms = np.linalg.norm(
        X,
        axis=2,
    )

    assert np.max(
        context_norms
    ) <= 1.0 + 1e-12

    # ------------------------------------------------------------
    # True probabilities.
    # ------------------------------------------------------------

    assert np.min(
        true_probabilities
    ) >= -1e-12

    assert np.max(
        true_probabilities
    ) <= 1.0 + 1e-12

    # ============================================================
    # THETA
    # ============================================================

    print_theta_information()

    # ============================================================
    # DATA-GENERATING PROCESS
    # ============================================================

    print()
    print("=" * 80)
    print("DATA-GENERATING PROCESS")
    print("=" * 80)

    print()
    print(
        "Context transformation:"
    )

    print(
        "x = raw_quality_vector / sqrt(6)"
    )

    print()
    print(
        "Satisfaction model:"
    )

    print(
        "Y[t,a] ~ Bernoulli(theta*^T x[t,a])"
    )

    print()
    print(
        "Maximum context norm:",
        f"{np.max(context_norms):.6f}",
    )

    print(
        "Maximum true probability:",
        f"{np.max(true_probabilities):.6f}",
    )

    print(
        "Minimum true probability:",
        f"{np.min(true_probabilities):.6f}",
    )

    # ============================================================
    # TRUE MODEL STATISTICS
    # ============================================================

    print()
    print("=" * 80)
    print("TRUE SATISFACTION PROBABILITIES")
    print("=" * 80)

    for a, model in enumerate(
        MODELS
    ):

        print(
            f"{model:35s}: "
            f"{np.mean(true_probabilities[:, a]):.4f}"
        )

    print()
    print(
        "Overall:",
        f"{np.mean(true_probabilities):.4f}",
    )

    print()
    print(
        "Realized Bernoulli Y:"
    )

    for a, model in enumerate(
        MODELS
    ):

        print(
            f"{model:35s}: "
            f"{np.mean(Y[:, a]):.4f}"
        )

    print()
    print(
        "Overall:",
        f"{np.mean(Y):.4f}",
    )

    # ============================================================
    # DEFAULT POC
    # ============================================================

    (
        default_results,
        default_summary,
    ) = run_default_poc(
        X=X,
        Y=Y,
        true_probabilities=
            true_probabilities,
    )

    # ============================================================
    # EPSILON SWEEP
    # ============================================================

    epsilon_df = (
        run_epsilon_sweep(
            X=X,
            Y=Y,
        )
    )

    # ============================================================
    # MAIN POC FIGURES
    # ============================================================

    plot_main_poc_figures(
        epsilon_df
    )

    # ============================================================
    # SAVE CONFIGURATION
    # ============================================================

    config = {
        "models":
            MODELS,

        "parameter_counts_B":
            PARAM_COUNTS_B.tolist(),

        "costs":
            COSTS.tolist(),

        "D":
            D,

        "epsilon":
            EPSILON,

        "delta":
            DELTA,

        "epsilon_sweep":
            [0.1,0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
            ],

        "T":
            int(T),

        "N_TRIALS":
            N_TRIALS,

        "eta_reference":
            ETA_REFERENCE,

        "eta_rule":
            "eta(epsilon) = ETA_REFERENCE * epsilon / EPSILON",

        "rho_reference":
            RHO_REFERENCE,

        "T_reference":
            T_REFERENCE,

        "alpha_scale":
            ALPHA_SCALE,

        "beta":
            BETA,

        "S":
            S,

        "theta_star":
            THETA_STAR.tolist(),

        "theta_star_norm":
            float(
                np.linalg.norm(
                    THETA_STAR
                )
            ),

        "theta_star_all_positive":
            bool(
                np.all(
                    THETA_STAR > 0
                )
            ),

        "context_normalization":
            "x = raw_quality_vector / sqrt(D)",

        "satisfaction_model":
            "Y ~ Bernoulli(theta_star^T x)",

        "fallback":
            False,

        "main_figures": [
            "poc_cost_vs_epsilon.png",
            "poc_satisfaction_vs_epsilon.png",
            "poc_model_selection_vs_epsilon.png",
        ],
    }

    config_file = os.path.join(
        OUTPUT_DIR,
        "poc_config.json",
    )

    with open(
        config_file,
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # ============================================================
    # FINISHED
    # ============================================================

    print()
    print("=" * 80)
    print("POC COMPLETE")
    print("=" * 80)

    print()
    print(
        "Results directory:",
        OUTPUT_DIR,
    )

    print()
    print("Generated:")

    print(
        "  poc_default_trials.csv"
    )

    print(
        "  poc_epsilon_sweep.csv"
    )

    print(
        "  poc_cost_vs_epsilon.png"
    )

    print(
        "  poc_satisfaction_vs_epsilon.png"
    )

    print(
        "  poc_model_selection_vs_epsilon.png"
    )

    print(
        "  poc_config.json"
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()
