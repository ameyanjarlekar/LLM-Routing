# ================================================================
# ICML PERSONALIZED LLM ROUTING
# COMPLETE STANDALONE EXPERIMENT SUITE
#
# Includes:
#   Experiment 1: Main performance
#   Experiment 2: Chance constraint
#   Experiment 3: Epsilon sweep
#   Experiment 4: Delta sweep
#   Experiment 5: Corrected personalization
#   Experiment 6: Ablations
#   Experiment 7: Context robustness
#   Experiment 8: Finite-horizon cost-gap scaling
#   Experiment 9: Confidence calibration
#
# Statistical reporting:
#   - 95% CIs for mean quantities
#   - Exact binomial CIs for chance-constraint probabilities
#   - Trial-level CSVs wherever appropriate
#
# IMPORTANT:
#   This is standalone. It does NOT depend on variables from
#   a previous notebook cell.
# ================================================================


# ================================================================
# 0. IMPORTS
# ================================================================

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.stats import t as student_t


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)


# ================================================================
# 1. OUTPUT DIRECTORIES
# ================================================================

OUT = Path(
    "icml_llm_routing_results"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

ERRORBAR_OUT = (
    OUT / "errorbars"
)

ERRORBAR_OUT.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 72)
print("ICML PERSONALIZED LLM ROUTING")
print("COMPLETE STANDALONE EXPERIMENT SUITE")
print("=" * 72)

print(
    "\nResults directory:"
)
print(
    OUT.resolve()
)


# ================================================================
# 2. GLOBAL EXPERIMENT SETTINGS
# ================================================================

SEED = 12345


# ------------------------------------------------
# Reference horizon / hyperparameters
# ------------------------------------------------

REFERENCE_T = 100

REFERENCE_ETA = 0.5

REFERENCE_RHO = 0.15

FROZEN_ALPHA_SCALE = 0.005


# ------------------------------------------------
# Main horizon
# ------------------------------------------------

BASE_T = 1000

T = BASE_T


# ------------------------------------------------
# Main safety parameters
# ------------------------------------------------

MAIN_EPSILON = 0.20

MAIN_DELTA = 0.05


# ------------------------------------------------
# Environment
# ------------------------------------------------

D = 8

K = 5

COSTS = np.array(
    [
        0.10,
        0.25,
        0.40,
        0.60,
        0.80,
    ],
    dtype=float,
)

FALLBACK = K

FALLBACK_COST = 1.0

FALLBACK_UCB = 1.0


# ------------------------------------------------
# GLM / context parameters
# ------------------------------------------------

L = 3.0

S = 1.0

BETA = 1.0

ZETA = 2.5

XI = 0.35


# ------------------------------------------------
# Context radii
# ------------------------------------------------

MODEL_RADII = np.array(
    [
        1.05,
        1.1625,
        1.275,
        1.3875,
        1.50,
    ],
    dtype=float,
)


# ------------------------------------------------
# Experiment sample sizes
# ------------------------------------------------
#
# These are deliberately explicit.
# Increase them later if you want tighter CIs.
# ------------------------------------------------

N_MAIN = 1000

N_CHANCE = 5000

N_EPS = 1000

N_DELTA = 1000

N_PERSONALIZATION = 1000

N_ABLATION = 1000

N_ROBUSTNESS = 1000

N_REGRET = 1000

N_CALIBRATION = 1000


# ------------------------------------------------
# Horizons
# ------------------------------------------------

HORIZONS = [
    100,
    250,
    500,
    1000,
    2000,
]


# ================================================================
# 3. BASIC MATH HELPERS
# ================================================================

def unit(x):
    """
    Return a unit-normalized vector.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    norm = np.linalg.norm(x)

    if norm < 1e-12:
        x = np.zeros_like(x)
        x[0] = 1.0
        norm = 1.0

    return x / norm


def horizon_scale(T_value):
    """
    sqrt(T0 / T)
    """

    return math.sqrt(
        REFERENCE_T / T_value
    )


def horizon_eta(T_value):
    """
    eta_T = eta_100 * sqrt(100/T)
    """

    return (
        REFERENCE_ETA
        * horizon_scale(T_value)
    )


def horizon_rho(T_value):
    """
    rho_T = rho_100 * sqrt(100/T)
    """

    return (
        REFERENCE_RHO
        * horizon_scale(T_value)
    )


# ================================================================
# 4. THEORETICAL RHO MAX
# ================================================================

def rho_max_star(
    T_value,
    epsilon,
    delta,
):
    """
    Exact finite-horizon safety margin.

    q = 1 - epsilon

    m = ceil(T q)

    Find mu such that:

        P[Binomial(T, mu) >= m] = 1-delta

    Then:

        rho_max = mu - q
    """

    from scipy.stats import binom

    q = 1.0 - epsilon

    m = int(
        math.ceil(
            T_value * q
        )
    )

    if m <= 0:
        return -q

    if m > T_value:
        return 1.0 - q

    target = 1.0 - delta

    def f(mu):

        probability = (
            binom.sf(
                m - 1,
                T_value,
                mu,
            )
        )

        return (
            probability
            - target
        )

    lo = q
    hi = 1.0

    if f(lo) >= 0:
        mu_star = lo

    elif f(hi) <= 0:
        mu_star = hi

    else:

        for _ in range(80):

            mid = (
                lo + hi
            ) / 2.0

            if f(mid) >= 0:
                hi = mid

            else:
                lo = mid

        mu_star = (
            lo + hi
        ) / 2.0

    return (
        mu_star - q
    )


# ================================================================
# 5. THEORETICAL ALPHA
# ================================================================

def sigma_prime(z):
    """
    Derivative of logistic link.
    """

    p = expit(z)

    return (
        p * (1.0 - p)
    )


KAPPA = sigma_prime(L)

if KAPPA <= 0:
    raise RuntimeError(
        "Invalid kappa."
    )


def theoretical_alpha(
    t,
):
    """
    Theoretical GLM-UCB radius.

    alpha_t =
        1/kappa [
            1/2 sqrt(
                d log(1+t L^2/(d beta))
                + 2 log(2/delta_E)
            )
            + sqrt(beta) S
        ]

    delta_E = t^{-zeta}
    """

    t = max(
        1,
        int(t),
    )

    delta_E = (
        t ** (-ZETA)
    )

    inside = (
        D
        * math.log(
            1.0
            + (
                t
                * L**2
                / (D * BETA)
            )
        )
        + 2.0
        * math.log(
            2.0 / delta_E
        )
    )

    return (
        1.0 / KAPPA
        * (
            0.5
            * math.sqrt(
                max(
                    0.0,
                    inside,
                )
            )
            + math.sqrt(BETA)
            * S
        )
    )


def empirical_alpha(
    t,
    scale=FROZEN_ALPHA_SCALE,
):
    """
    Empirically calibrated radius.

    The theoretical alpha is multiplied by a fixed
    scale factor. The scale is NOT changed with T.
    """

    return (
        scale
        * theoretical_alpha(t)
    )


# ================================================================
# 6. CONTEXT GENERATION
# ================================================================

def make_model_directions(
    rng,
):
    """
    Five model directions.
    """

    directions = []

    for _ in range(K):

        directions.append(
            unit(
                rng.normal(
                    size=D
                )
            )
        )

    return np.asarray(
        directions
    )


def make_user_directions(
    rng,
    n_users=5,
):
    """
    User-specific directions.
    """

    directions = []

    for _ in range(n_users):

        directions.append(
            unit(
                rng.normal(
                    size=D
                )
            )
        )

    return np.asarray(
        directions
    )


def generate_contexts(
    rng,
    theta_star,
    model_dirs,
    xi=XI,
    user_dir=None,
    shared_mode=False,
    T_override=None,
):
    """
    Generate model-specific contextual vectors.

    Contexts lie inside spherical caps around model
    directions, with additional user/context variation.

    Shape:

        (T, K, D)
    """

    if T_override is None:
        T_override = T

    X = np.zeros(
        (
            T_override,
            K,
            D,
        ),
        dtype=float,
    )

    theta_dir = unit(
        theta_star
    )

    if user_dir is None:
        user_dir = theta_dir

    user_dir = unit(
        user_dir
    )

    for t in range(
        T_override
    ):

        for a in range(K):

            base = model_dirs[a]

            # random tangent perturbation
            noise = rng.normal(
                size=D
            )

            noise = (
                noise
                - np.dot(
                    noise,
                    base,
                )
                * base
            )

            noise_norm = np.linalg.norm(
                noise
            )

            if noise_norm < 1e-12:

                noise = np.zeros(D)

            else:

                noise /= noise_norm

            # Uniform-ish random radius inside the cap.
            radius = (
                MODEL_RADII[a]
                * (
                    rng.random()
                    ** (
                        1.0
                        / max(
                            1,
                            D - 1,
                        )
                    )
                )
            )

            direction = unit(
                base
                + radius
                * noise
            )

            # User/context mixture.
            if shared_mode:

                direction = unit(
                    (
                        1.0 - xi
                    )
                    * direction
                    + xi
                    * theta_dir
                )

            else:

                direction = unit(
                    (
                        1.0 - xi
                    )
                    * direction
                    + xi
                    * user_dir
                )

            X[t, a] = (
                direction
                * min(
                    MODEL_RADII[a],
                    L,
                )
            )

    return X


# ================================================================
# 7. LOGISTIC MLE
# ================================================================

def logistic_mle(
    X,
    y,
):
    """
    Ridge logistic maximum likelihood estimate.

    Objective:
        negative log-likelihood
        + 0.5 beta ||theta||^2

    Newton-CG/L-BFGS style numerical optimization is used
    through scipy.optimize.minimize.

    For small sample sizes, returns zeros if optimization
    cannot be performed.
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if len(y) == 0:
        return np.zeros(D)

    theta0 = np.zeros(D)

    def objective(theta):

        z = X @ theta

        # stable log(1 + exp(z))
        log_term = np.logaddexp(
            0.0,
            z,
        )

        nll = np.sum(
            log_term
            - y * z
        )

        ridge = (
            0.5
            * BETA
            * np.dot(
                theta,
                theta,
            )
        )

        return (
            nll + ridge
        )

    def gradient(theta):

        z = X @ theta

        p = expit(z)

        grad = (
            X.T
            @ (
                p - y
            )
        )

        grad += (
            BETA
            * theta
        )

        return grad

    try:

        result = minimize(
            objective,
            theta0,
            jac=gradient,
            method="L-BFGS-B",
            options={
                "maxiter": 100,
                "ftol": 1e-8,
                "gtol": 1e-6,
            },
        )

        if result.success:

            return result.x

        return result.x

    except Exception:

        return theta0


# ================================================================
# 8. TRIAL CONFIG
# ================================================================

@dataclass
class TrialConfig:

    epsilon: float = MAIN_EPSILON

    delta: float = MAIN_DELTA

    xi: float = XI

    eta: float | None = None

    rho: float | None = None

    use_ucb: bool = True

    use_queue: bool = True

    use_rho: bool = True

    theta_star: np.ndarray | None = None


# ================================================================
# 9. SINGLE STANDARD TRIAL
# ================================================================

def run_trial(
    config,
    seed,
    T_run=None,
):
    """
    Run one standard routing trial.
    """

    if T_run is None:
        T_run = BASE_T

    rng = np.random.default_rng(
        seed
    )

    epsilon = (
        config.epsilon
    )

    eta = (
        horizon_eta(T_run)
        if config.eta is None
        else config.eta
    )

    rho = (
        horizon_rho(T_run)
        if config.rho is None
        else config.rho
    )

    if not config.use_rho:
        rho_used = 0.0
    else:
        rho_used = rho

    # ------------------------------------------------------------
    # True parameter
    # ------------------------------------------------------------

    if config.theta_star is None:

        theta_star = (
            unit(
                rng.normal(
                    size=D
                )
            )
            * S
        )

    else:

        theta_star = np.asarray(
            config.theta_star,
            dtype=float,
        ).copy()

    # ------------------------------------------------------------
    # Model directions
    # ------------------------------------------------------------

    model_dirs = (
        make_model_directions(
            rng
        )
    )

    # ------------------------------------------------------------
    # Contexts
    # ------------------------------------------------------------

    X = generate_contexts(
        rng=rng,
        theta_star=theta_star,
        model_dirs=model_dirs,
        xi=config.xi,
        user_dir=unit(theta_star),
        shared_mode=False,
        T_override=T_run,
    )

    # ------------------------------------------------------------
    # True probabilities
    # ------------------------------------------------------------

    P = expit(
        np.einsum(
            "tkd,d->tk",
            X,
            theta_star,
        )
    )

    # ------------------------------------------------------------
    # Learner state
    # ------------------------------------------------------------

    V = (
        BETA
        * np.eye(D)
    )

    history_X = []

    history_y = []

    lam = 0.0

    # ------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------

    action_history = np.zeros(
        T_run,
        dtype=int,
    )

    outcome_history = np.zeros(
        T_run,
        dtype=int,
    )

    cost_history = np.zeros(
        T_run
    )

    fallback_history = np.zeros(
        T_run
    )

    selected_true_p = np.zeros(
        T_run
    )

    selected_ucb = np.zeros(
        T_run
    )

    lambda_history = np.zeros(
        T_run
    )

    # ------------------------------------------------------------
    # Online loop
    # ------------------------------------------------------------

    for t in range(
        T_run
    ):

        # --------------------------------------------------------
        # Estimate theta
        # --------------------------------------------------------

        if len(history_y) > 0:

            theta_hat = logistic_mle(
                np.asarray(
                    history_X
                ),
                np.asarray(
                    history_y
                ),
            )

        else:

            theta_hat = np.zeros(D)

        Vinv = np.linalg.pinv(
            V
        )

        # --------------------------------------------------------
        # Confidence radius
        # --------------------------------------------------------

        alpha = empirical_alpha(
            t + 1,
            FROZEN_ALPHA_SCALE,
        )

        # --------------------------------------------------------
        # Scores
        # --------------------------------------------------------

        scores = np.zeros(
            K + 1
        )

        ucbs = np.zeros(
            K + 1
        )

        for a in range(K):

            x = X[t, a]

            mean = float(
                expit(
                    x @ theta_hat
                )
            )

            if config.use_ucb:

                uncertainty = math.sqrt(
                    max(
                        0.0,
                        x
                        @ Vinv
                        @ x,
                    )
                )

                ucb = (
                    mean
                    + alpha
                    * uncertainty
                )

            else:

                ucb = mean

            ucbs[a] = ucb

        # --------------------------------------------------------
        # Perfect fallback
        # --------------------------------------------------------

        ucbs[FALLBACK] = (
            FALLBACK_UCB
        )

        # --------------------------------------------------------
        # Queue multiplier
        #
        # No-Queue uses lambda=1 as a diagnostic ablation:
        # this removes dynamic dual pricing rather than
        # literally setting lambda to zero.
        # --------------------------------------------------------

        if config.use_queue:

            queue_multiplier = lam

        else:

            queue_multiplier = 1.0

        for a in range(
            K + 1
        ):

            scores[a] = (
                (
                    COSTS[a]
                    if a < K
                    else FALLBACK_COST
                )
                - queue_multiplier
                * ucbs[a]
            )

        action = int(
            np.argmin(scores)
        )

        # --------------------------------------------------------
        # Sample outcome
        # --------------------------------------------------------

        if action == FALLBACK:

            true_p = 1.0

            y = 1

            cost = FALLBACK_COST

            fallback = 1.0

        else:

            true_p = float(
                P[t, action]
            )

            y = int(
                rng.random()
                < true_p
            )

            cost = float(
                COSTS[action]
            )

            fallback = 0.0

        # --------------------------------------------------------
        # Record
        # --------------------------------------------------------

        action_history[t] = action

        outcome_history[t] = y

        cost_history[t] = cost

        fallback_history[t] = fallback

        selected_true_p[t] = true_p

        selected_ucb[t] = (
            ucbs[action]
        )

        lambda_history[t] = lam

        # --------------------------------------------------------
        # Update learner
        # --------------------------------------------------------

        if action < K:

            x_selected = (
                X[t, action].copy()
            )

            history_X.append(
                x_selected
            )

            history_y.append(
                y
            )

            V += np.outer(
                x_selected,
                x_selected,
            )

        # --------------------------------------------------------
        # Queue update
        # --------------------------------------------------------

        if config.use_queue:

            lam = max(
                0.0,
                lam
                + eta
                * (
                    1.0
                    - epsilon
                    + rho_used
                    - y
                ),
            )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    satisfaction = float(
        np.mean(
            outcome_history
        )
    )

    avg_cost = float(
        np.mean(
            cost_history
        )
    )

    fallback_rate = float(
        np.mean(
            fallback_history
        )
    )

    violation = float(
        satisfaction
        < (
            1.0
            - epsilon
        )
    )

    return {

        "avg_cost":
            avg_cost,

        "satisfaction":
            satisfaction,

        "fallback_rate":
            fallback_rate,

        "violation":
            violation,

        "eta":
            eta,

        "rho":
            rho_used,

        "epsilon":
            epsilon,

        "delta":
            config.delta,

        "xi":
            config.xi,

        "T":
            T_run,

        "actions":
            action_history,

        "outcomes":
            outcome_history,

        "costs":
            cost_history,

        "fallbacks":
            fallback_history,

        "selected_true_p":
            selected_true_p,

        "selected_ucb":
            selected_ucb,

        "lambda":
            lambda_history,

    }


# ================================================================
# 10. MULTIPLE TRIALS
# ================================================================

def run_trials(
    config,
    n_trials,
    seed0=SEED,
    T_run=None,
):
    """
    Run independent trials.
    """

    rows = []

    for i in range(
        n_trials
    ):

        result = run_trial(
            config=config,
            seed=seed0 + i,
            T_run=T_run,
        )

        row = {
            k: v
            for k, v in result.items()
            if np.isscalar(v)
        }

        row["trial"] = i

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ================================================================
# 11. CI HELPERS
# ================================================================

def mean_ci95(
    values,
):
    """
    95% Student-t CI for a mean.
    """

    x = np.asarray(
        values,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    n = len(x)

    if n == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    mean = float(
        np.mean(x)
    )

    if n == 1:

        return (
            mean,
            mean,
            mean,
        )

    se = (
        np.std(
            x,
            ddof=1,
        )
        / math.sqrt(n)
    )

    critical = student_t.ppf(
        0.975,
        n - 1,
    )

    half = (
        critical
        * se
    )

    return (
        mean,
        mean - half,
        mean + half,
    )


def proportion_ci95_exact(
    successes,
    n,
):
    """
    Exact Clopper-Pearson 95% CI.
    """

    successes = int(
        successes
    )

    n = int(n)

    if n <= 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    p = (
        successes
        / n
    )

    if successes == 0:

        lo = 0.0

    else:

        lo = beta_dist.ppf(
            0.025,
            successes,
            n - successes + 1,
        )

    if successes == n:

        hi = 1.0

    else:

        hi = beta_dist.ppf(
            0.975,
            successes + 1,
            n - successes,
        )

    return (
        float(p),
        float(lo),
        float(hi),
    )


# ================================================================
# 12. SUMMARY FUNCTION
# ================================================================

def summarize_trials(
    df,
):
    """
    Compact summary with 95% CIs.
    """

    cost_mean, cost_lo, cost_hi = (
        mean_ci95(
            df["avg_cost"]
        )
    )

    sat_mean, sat_lo, sat_hi = (
        mean_ci95(
            df["satisfaction"]
        )
    )

    successes = int(
        np.sum(
            1.0
            - df["violation"]
        )
    )

    cp, cp_lo, cp_hi = (
        proportion_ci95_exact(
            successes,
            len(df),
        )
    )

    return {

        "mean_cost":
            cost_mean,

        "cost_ci95_low":
            cost_lo,

        "cost_ci95_high":
            cost_hi,

        "mean_satisfaction":
            sat_mean,

        "satisfaction_ci95_low":
            sat_lo,

        "satisfaction_ci95_high":
            sat_hi,

        "constraint_probability":
            cp,

        "constraint_ci95_low":
            cp_lo,

        "constraint_ci95_high":
            cp_hi,

        "fallback_rate":
            float(
                df[
                    "fallback_rate"
                ].mean()
            ),

    }


# ================================================================
# 13. EXPERIMENT 1
#     MAIN PERFORMANCE
# ================================================================

def experiment_main():

    print(
        "\n" + "=" * 72
    )

    print(
        "EXPERIMENT 1: MAIN PERFORMANCE"
    )

    print(
        "=" * 72
    )

    methods = {

        "Ours":
            TrialConfig(
                use_ucb=True,
                use_queue=True,
                use_rho=True,
            ),

        "No-UCB":
            TrialConfig(
                use_ucb=False,
                use_queue=True,
                use_rho=True,
            ),

        "No-Queue":
            TrialConfig(
                use_ucb=True,
                use_queue=False,
                use_rho=True,
            ),

        "No-Rho":
            TrialConfig(
                use_ucb=True,
                use_queue=True,
                use_rho=False,
            ),

    }

    all_trials = []

    summaries = []

    for name, config in (
        methods.items()
    ):

        print(
            f"\nRunning {name}..."
        )

        df = run_trials(
            config,
            N_MAIN,
            seed0=SEED + 10000,
            T_run=BASE_T,
        )

        df["method"] = name

        all_trials.append(
            df
        )

        summary = (
            summarize_trials(
                df
            )
        )

        summary["method"] = name

        summaries.append(
            summary
        )

    trials = pd.concat(
        all_trials,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summaries
    )

    trials.to_csv(
        ERRORBAR_OUT
        / "main_performance_trials.csv",
        index=False,
    )

    summary.to_csv(
        ERRORBAR_OUT
        / "main_performance_errorbars.csv",
        index=False,
    )

    print(
        "\nMAIN RESULTS:"
    )

    print(
        summary[
            [
                "method",
                "mean_cost",
                "cost_ci95_low",
                "cost_ci95_high",
                "mean_satisfaction",
                "satisfaction_ci95_low",
                "satisfaction_ci95_high",
                "constraint_probability",
                "constraint_ci95_low",
                "constraint_ci95_high",
                "fallback_rate",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------

    y = summary[
        "mean_cost"
    ].to_numpy()

    yerr = np.vstack(
        [
            y
            - summary[
                "cost_ci95_low"
            ].to_numpy(),

            summary[
                "cost_ci95_high"
            ].to_numpy()
            - y,
        ]
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.bar(
        summary["method"],
        y,
        yerr=yerr,
        capsize=5,
    )

    plt.ylabel(
        "Average cost"
    )

    plt.title(
        "Main performance: routing cost"
    )

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "main_cost_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    # ------------------------------------------------------------
    # Satisfaction
    # ------------------------------------------------------------

    y = summary[
        "mean_satisfaction"
    ].to_numpy()

    yerr = np.vstack(
        [
            y
            - summary[
                "satisfaction_ci95_low"
            ].to_numpy(),

            summary[
                "satisfaction_ci95_high"
            ].to_numpy()
            - y,
        ]
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.bar(
        summary["method"],
        y,
        yerr=yerr,
        capsize=5,
    )

    plt.axhline(
        1.0 - MAIN_EPSILON,
        linestyle="--",
        label="Required satisfaction",
    )

    plt.ylabel(
        "Mean satisfaction"
    )

    plt.title(
        "Main performance: satisfaction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "main_satisfaction_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    return (
        trials,
        summary,
    )


# ================================================================
# 14. EXPERIMENT 2
#     CHANCE CONSTRAINT
# ================================================================

def experiment_chance_constraint():

    print(
        "\n" + "=" * 72
    )

    print(
        "EXPERIMENT 2: CHANCE CONSTRAINT"
    )

    print(
        "=" * 72
    )

    config = TrialConfig()

    df = run_trials(
        config,
        N_CHANCE,
        seed0=SEED + 20000,
        T_run=BASE_T,
    )

    successes = int(
        np.sum(
            1.0
            - df["violation"]
        )
    )

    cp, cp_lo, cp_hi = (
        proportion_ci95_exact(
            successes,
            len(df),
        )
    )

    summary = pd.DataFrame(
        [
            {
                "T":
                    BASE_T,

                "epsilon":
                    MAIN_EPSILON,

                "delta":
                    MAIN_DELTA,

                "target_satisfaction":
                    1.0 - MAIN_EPSILON,

                "target_probability":
                    1.0 - MAIN_DELTA,

                "observed_constraint_probability":
                    cp,

                "ci95_low":
                    cp_lo,

                "ci95_high":
                    cp_hi,

                "mean_satisfaction":
                    df[
                        "satisfaction"
                    ].mean(),

                "mean_cost":
                    df[
                        "avg_cost"
                    ].mean(),

                "rho":
                    horizon_rho(
                        BASE_T
                    ),

                "rho_max_star":
                    rho_max_star(
                        BASE_T,
                        MAIN_EPSILON,
                        MAIN_DELTA,
                    ),
            }
        ]
    )

    df.to_csv(
        ERRORBAR_OUT
        / "chance_constraint_trials.csv",
        index=False,
    )

    summary.to_csv(
        ERRORBAR_OUT
        / "chance_constraint.csv",
        index=False,
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Satisfaction distribution
    # ------------------------------------------------------------

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.hist(
        df["satisfaction"],
        bins=30,
        alpha=0.8,
    )

    plt.axvline(
        1.0 - MAIN_EPSILON,
        linestyle="--",
        label="Required satisfaction",
    )

    plt.xlabel(
        "Episode satisfaction"
    )

    plt.ylabel(
        "Number of trials"
    )

    plt.title(
        "Chance-constraint satisfaction distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "chance_constraint_distribution.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    return (
        df,
        summary,
    )


# ================================================================
# 15. EXPERIMENT 3
#     EPSILON SWEEP
# ================================================================

def experiment_epsilon():

    print(
        "\n" + "=" * 72
    )

    print(
        "EXPERIMENT 3: EPSILON SWEEP"
    )

    print(
        "=" * 72
    )

    epsilon_values = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
    ]

    trial_rows = []

    summary_rows = []

    for eps in epsilon_values:

        print(
            f"\nRunning epsilon={eps:.2f}"
        )

        config = TrialConfig(
            epsilon=eps,
        )

        df = run_trials(
            config,
            N_EPS,
            seed0=SEED + 30000,
            T_run=BASE_T,
        )

        df["epsilon"] = eps

        trial_rows.append(
            df
        )

        s = summarize_trials(
            df
        )

        s["epsilon"] = eps

        s[
            "required_satisfaction"
        ] = 1.0 - eps

        summary_rows.append(
            s
        )

    trials = pd.concat(
        trial_rows,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "epsilon"
    )

    trials.to_csv(
        ERRORBAR_OUT
        / "epsilon_sweep_trials.csv",
        index=False,
    )

    summary.to_csv(
        ERRORBAR_OUT
        / "epsilon_sweep_errorbars.csv",
        index=False,
    )

    print(
        summary.to_string(
            index=False
        )
    )

    x = summary[
        "required_satisfaction"
    ].to_numpy()

    # Cost
    y = summary[
        "mean_cost"
    ].to_numpy()

    yerr = np.vstack(
        [
            y
            - summary[
                "cost_ci95_low"
            ].to_numpy(),

            summary[
                "cost_ci95_high"
            ].to_numpy()
            - y,
        ]
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=5,
    )

    plt.xlabel(
        "Required satisfaction"
    )

    plt.ylabel(
        "Average cost"
    )

    plt.title(
        "Epsilon sweep: cost"
    )

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "epsilon_cost_quality_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    # Satisfaction
    y = summary[
        "mean_satisfaction"
    ].to_numpy()

    yerr = np.vstack(
        [
            y
            - summary[
                "satisfaction_ci95_low"
            ].to_numpy(),

            summary[
                "satisfaction_ci95_high"
            ].to_numpy()
            - y,
        ]
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=5,
        label="Achieved",
    )

    plt.plot(
        x,
        x,
        linestyle="--",
        label="Target",
    )

    plt.xlabel(
        "Required satisfaction"
    )

    plt.ylabel(
        "Mean satisfaction"
    )

    plt.title(
        "Epsilon sweep: satisfaction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "epsilon_satisfaction_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    return (
        trials,
        summary,
    )


# ================================================================
# 16. EXPERIMENT 4
#     DELTA SWEEP
# ================================================================

def experiment_delta():

    print(
        "\n" + "=" * 72
    )

    print(
        "EXPERIMENT 4: DELTA SWEEP"
    )

    print(
        "=" * 72
    )

    delta_values = [
        0.01,
        0.05,
        0.10,
    ]

    trial_rows = []

    summary_rows = []

    for delta in delta_values:

        print(
            f"\nRunning delta={delta:.2f}"
        )

        config = TrialConfig(
            epsilon=MAIN_EPSILON,
            delta=delta,
        )

        df = run_trials(
            config,
            N_DELTA,
            seed0=SEED + 40000,
            T_run=BASE_T,
        )

        df["delta"] = delta

        trial_rows.append(
            df
        )

        successes = int(
            np.sum(
                1.0
                - df["violation"]
            )
        )

        cp, cp_lo, cp_hi = (
            proportion_ci95_exact(
                successes,
                len(df),
            )
        )

        summary_rows.append(
            {
                "delta":
                    delta,

                "required_probability":
                    1.0 - delta,

                "constraint_probability":
                    cp,

                "constraint_ci95_low":
                    cp_lo,

                "constraint_ci95_high":
                    cp_hi,

                "mean_cost":
                    df[
                        "avg_cost"
                    ].mean(),

                "mean_satisfaction":
                    df[
                        "satisfaction"
                    ].mean(),
            }
        )

    trials = pd.concat(
        trial_rows,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "delta"
    )

    trials.to_csv(
        ERRORBAR_OUT
        / "delta_sweep_trials.csv",
        index=False,
    )

    summary.to_csv(
        ERRORBAR_OUT
        / "delta_sweep_errorbars.csv",
        index=False,
    )

    print(
        summary.to_string(
            index=False
        )
    )

    x = summary[
        "required_probability"
    ].to_numpy()

    y = summary[
        "constraint_probability"
    ].to_numpy()

    yerr = np.vstack(
        [
            y
            - summary[
                "constraint_ci95_low"
            ].to_numpy(),

            summary[
                "constraint_ci95_high"
            ].to_numpy()
            - y,
        ]
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=5,
        label="Observed",
    )

    plt.plot(
        x,
        x,
        linestyle="--",
        label="Required",
    )

    plt.xlabel(
        "Required chance-constraint probability"
    )

    plt.ylabel(
        "Observed probability"
    )

    plt.title(
        "Delta sweep: chance-constraint probability"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "delta_constraint_probability_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    return (
        trials,
        summary,
    )


# ================================================================
# 17. EXPERIMENT 5
#     CORRECTED PERSONALIZATION
# ================================================================
def run_personalization_method(
    X_all,
    P_all,
    uniforms,
    mode,
    epsilon,
    eta,
    rho,
):
    """
    Run either:

        mode="global"
        mode="personalized"

    on the SAME counterfactual environment.

    Experimental setup:
        - X_all[u, t, a] contains the same underlying
          request/context for all users.
        - P_all[u, t, a] differs across users because
          each user has a different true preference parameter.

    Global:
        one pooled theta_hat and V across all users.

    Personalized:
        one theta_hat and V per user.

    Both methods use:
        - one independent safety queue per user
        - the same counterfactual uniforms
        - the same contexts and candidate actions
    """

    n_users = X_all.shape[0]
    T_run = X_all.shape[1]

    if mode not in ("global", "personalized"):
        raise ValueError("Invalid mode.")

    # ------------------------------------------------------------
    # Learner state
    # ------------------------------------------------------------

    if mode == "global":

        V_global = BETA * np.eye(D)

        history_X_global = []
        history_y_global = []

    else:

        V_user = [
            BETA * np.eye(D)
            for _ in range(n_users)
        ]

        history_X_user = [
            []
            for _ in range(n_users)
        ]

        history_y_user = [
            []
            for _ in range(n_users)
        ]

    # ------------------------------------------------------------
    # Independent safety queue for each user
    # ------------------------------------------------------------

    lambdas = np.zeros(n_users)

    # ------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------

    costs = np.zeros(
        (n_users, T_run)
    )

    outcomes = np.zeros(
        (n_users, T_run),
        dtype=int,
    )

    fallback = np.zeros(
        (n_users, T_run)
    )

    selected_action = np.zeros(
        (n_users, T_run),
        dtype=int,
    )

    # ------------------------------------------------------------
    # Online round-robin interaction
    # ------------------------------------------------------------

    for t in range(T_run):

        for u in range(n_users):

            # ----------------------------------------------------
            # Current parameter estimate
            # ----------------------------------------------------

            if mode == "global":

                if len(history_y_global) > 0:

                    theta_hat = logistic_mle(
                        np.asarray(history_X_global),
                        np.asarray(history_y_global),
                    )

                else:

                    theta_hat = np.zeros(D)

                V_current = V_global

            else:

                if len(history_y_user[u]) > 0:

                    theta_hat = logistic_mle(
                        np.asarray(history_X_user[u]),
                        np.asarray(history_y_user[u]),
                    )

                else:

                    theta_hat = np.zeros(D)

                V_current = V_user[u]

            # ----------------------------------------------------
            # Confidence radius
            # ----------------------------------------------------

            Vinv = np.linalg.pinv(V_current)

            alpha = empirical_alpha(
                t + 1,
                FROZEN_ALPHA_SCALE,
            )

            # ----------------------------------------------------
            # Action scores
            # ----------------------------------------------------

            scores = np.zeros(K + 1)

            for a in range(K):

                x = X_all[
                    u,
                    t,
                    a,
                ]

                mean = float(
                    expit(
                        x @ theta_hat
                    )
                )

                uncertainty = math.sqrt(
                    max(
                        0.0,
                        x @ Vinv @ x,
                    )
                )

                ucb = (
                    mean
                    + alpha * uncertainty
                )

                scores[a] = (
                    COSTS[a]
                    - lambdas[u] * ucb
                )

            # ----------------------------------------------------
            # Perfect fallback
            # ----------------------------------------------------

            scores[FALLBACK] = (
                FALLBACK_COST
                - lambdas[u] * FALLBACK_UCB
            )

            action = int(
                np.argmin(scores)
            )

            selected_action[
                u,
                t,
            ] = action

            # ----------------------------------------------------
            # Observe selected action outcome
            # ----------------------------------------------------

            if action == FALLBACK:

                y = 1
                cost = FALLBACK_COST

                fallback[
                    u,
                    t,
                ] = 1.0

            else:

                p = P_all[
                    u,
                    t,
                    action,
                ]

                y = int(
                    uniforms[
                        u,
                        t,
                        action,
                    ] < p
                )

                cost = float(
                    COSTS[action]
                )

                x_selected = X_all[
                    u,
                    t,
                    action,
                ].copy()

                # ------------------------------------------------
                # Update appropriate learner
                # ------------------------------------------------

                if mode == "global":

                    history_X_global.append(
                        x_selected
                    )

                    history_y_global.append(
                        y
                    )

                    V_global += np.outer(
                        x_selected,
                        x_selected,
                    )

                else:

                    history_X_user[u].append(
                        x_selected
                    )

                    history_y_user[u].append(
                        y
                    )

                    V_user[u] += np.outer(
                        x_selected,
                        x_selected,
                    )

            costs[
                u,
                t,
            ] = cost

            outcomes[
                u,
                t,
            ] = y

            # ----------------------------------------------------
            # Update user-specific safety queue
            # ----------------------------------------------------

            lambdas[u] = max(
                0.0,
                lambdas[u]
                + eta * (
                    1.0
                    - epsilon
                    + rho
                    - y
                ),
            )

    return {
        "costs": costs,
        "outcomes": outcomes,
        "fallback": fallback,
        "actions": selected_action,
    }

def make_personalized_user_preferences(
    rng,
    model_dirs,
    n_users=5,
    alignment=0.7,
):
    """
    Create heterogeneous user preference vectors.

    User u's preference vector is centered around model_dirs[u],
    with additional random perturbation.

    alignment:
        Higher -> user preference is more strongly aligned with
        the corresponding model direction.

    Returns:
        theta_users: shape (n_users, D)
    """
    assert n_users <= len(model_dirs)

    theta_users = []

    for u in range(n_users):
        noise = unit(rng.normal(size=D))

        theta_user = unit(
            alignment * model_dirs[u]
            + (1.0 - alignment) * noise
        ) * S

        theta_users.append(theta_user)

    return np.array(theta_users)

    
def experiment_personalization():

    print(
        "\n" + "=" * 72
    )

    print(
        "EXPERIMENT 5: PERSONALIZATION"
    )

    print(
        "=" * 72
    )

    n_users = 5

    # ------------------------------------------------------------
    # User preference alignment with model directions
    #
    # User u is preferentially aligned with model u, while
    # retaining random variation.
    #
    # theta_u^* =
    #   S * unit(
    #       ALIGNMENT * d_u
    #       + (1 - ALIGNMENT) * z_u
    #   )
    #
    # where d_u is model u's direction and z_u is random noise.
    # ------------------------------------------------------------

    ALIGNMENT = 0.7

    eta = horizon_eta(
        BASE_T
    )

    rho = horizon_rho(
        BASE_T
    )

    print(
        f"T = {BASE_T}"
    )

    print(
        f"Users = {n_users}"
    )

    print(
        f"Trials = {N_PERSONALIZATION}"
    )

    print(
        f"User preference alignment = {ALIGNMENT:.2f}"
    )

    print(
        f"eta = {eta:.6f}"
    )

    print(
        f"rho = {rho:.6f}"
    )

    trial_rows = []

    for trial in range(
        N_PERSONALIZATION
    ):

        rng = np.random.default_rng(
            SEED
            + 50000
            + trial
        )

        # --------------------------------------------------------
        # Common context-generation parameter
        #
        # This is used only to generate the common request/context
        # sequence. It is NOT the preference parameter of any user.
        # --------------------------------------------------------

        theta_context = (
            unit(
                rng.normal(
                    size=D
                )
            )
            * S
        )

        # --------------------------------------------------------
        # Model directions
        #
        # Each model has its own region in context space.
        # --------------------------------------------------------

        model_dirs = (
            make_model_directions(
                rng
            )
        )

        # --------------------------------------------------------
        # One common context sequence
        #
        # Every user sees exactly the SAME X[t, a].
        #
        # This isolates personalization to differences in
        # user preferences rather than differences in workloads.
        # --------------------------------------------------------

        X_base = generate_contexts(
            rng=rng,
            theta_star=theta_context,
            model_dirs=model_dirs,
            xi=XI,
            user_dir=None,
            shared_mode=True,
            T_override=BASE_T,
        )

        # --------------------------------------------------------
        # Generate heterogeneous user preference parameters
        #
        # User u is centered around model_dirs[u].
        #
        # This creates meaningful preference heterogeneity:
        #
        #   User 1 -> prefers region of Model 1
        #   User 2 -> prefers region of Model 2
        #   ...
        #   User 5 -> prefers region of Model 5
        #
        # But the random component prevents this from being
        # completely deterministic.
        # --------------------------------------------------------

        theta_users = []

        for u in range(
            n_users
        ):

            noise = unit(
                rng.normal(
                    size=D
                )
            )

            theta_user = (
                unit(
                    ALIGNMENT * model_dirs[u]
                    + (
                        1.0
                        - ALIGNMENT
                    ) * noise
                )
                * S
            )

            theta_users.append(
                theta_user
            )

        theta_users = np.array(
            theta_users
        )

        # --------------------------------------------------------
        # Construct user-specific environments
        #
        # SAME X for every user.
        #
        # DIFFERENT theta_u^* for every user.
        #
        # Therefore:
        #
        #   p_{u,a,t}
        #       = sigmoid(
        #           x_{a,t}^T theta_u^*
        #         )
        #
        # differs across users even though the contexts are
        # identical.
        # --------------------------------------------------------

        X_all = np.zeros(
            (
                n_users,
                BASE_T,
                K,
                D,
            )
        )

        P_all = np.zeros(
            (
                n_users,
                BASE_T,
                K,
            )
        )

        for u in range(
            n_users
        ):

            theta_user = (
                theta_users[u]
            )

            X_all[u] = X_base

            P_all[u] = expit(
                np.einsum(
                    "tkd,d->tk",
                    X_base,
                    theta_user,
                )
            )

        # --------------------------------------------------------
        # Same counterfactual random numbers for both learners
        # --------------------------------------------------------

        uniforms = rng.random(
            (
                n_users,
                BASE_T,
                K,
            )
        )

        # --------------------------------------------------------
        # Global learner
        # --------------------------------------------------------

        global_result = (
            run_personalization_method(
                X_all=X_all,
                P_all=P_all,
                uniforms=uniforms,
                mode="global",
                epsilon=MAIN_EPSILON,
                eta=eta,
                rho=rho,
            )
        )

        # --------------------------------------------------------
        # Personalized learner
        # --------------------------------------------------------

        personalized_result = (
            run_personalization_method(
                X_all=X_all,
                P_all=P_all,
                uniforms=uniforms,
                mode="personalized",
                epsilon=MAIN_EPSILON,
                eta=eta,
                rho=rho,
            )
        )

        # --------------------------------------------------------
        # Store user-level results
        # --------------------------------------------------------

        for u in range(
            n_users
        ):

            global_cost = float(
                np.mean(
                    global_result[
                        "costs"
                    ][u]
                )
            )

            global_sat = float(
                np.mean(
                    global_result[
                        "outcomes"
                    ][u]
                )
            )

            global_fallback = float(
                np.mean(
                    global_result[
                        "fallback"
                    ][u]
                )
            )

            personalized_cost = float(
                np.mean(
                    personalized_result[
                        "costs"
                    ][u]
                )
            )

            personalized_sat = float(
                np.mean(
                    personalized_result[
                        "outcomes"
                    ][u]
                )
            )

            personalized_fallback = float(
                np.mean(
                    personalized_result[
                        "fallback"
                    ][u]
                )
            )

            trial_rows.append(
                {
                    "trial": trial,

                    "user": u + 1,

                    "global_cost":
                        global_cost,

                    "global_satisfaction":
                        global_sat,

                    "global_fallback_rate":
                        global_fallback,

                    "global_violation":
                        float(
                            global_sat
                            < (
                                1.0
                                - MAIN_EPSILON
                            )
                        ),

                    "personalized_cost":
                        personalized_cost,

                    "personalized_satisfaction":
                        personalized_sat,

                    "personalized_fallback_rate":
                        personalized_fallback,

                    "personalized_violation":
                        float(
                            personalized_sat
                            < (
                                1.0
                                - MAIN_EPSILON
                            )
                        ),

                    "cost_improvement":
                        (
                            global_cost
                            - personalized_cost
                        ),

                    "satisfaction_improvement":
                        (
                            personalized_sat
                            - global_sat
                        ),
                }
            )

        if (
            (trial + 1)
            % max(
                1,
                N_PERSONALIZATION
                // 10,
            )
            == 0
        ):

            print(
                f"Completed "
                f"{trial + 1}/"
                f"{N_PERSONALIZATION}"
            )

    trials = pd.DataFrame(
        trial_rows
    )

    # ------------------------------------------------------------
    # Save raw trial-level results
    # ------------------------------------------------------------

    trials.to_csv(
        ERRORBAR_OUT
        / "personalization_corrected_trials.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Per-user summary
    # ------------------------------------------------------------

    summary_rows = []

    for user_id in range(
        1,
        n_users + 1,
    ):

        d = trials[
            trials["user"]
            == user_id
        ]

        # Cost CIs
        pcost, pcost_lo, pcost_hi = (
            mean_ci95(
                d["personalized_cost"]
            )
        )

        gcost, gcost_lo, gcost_hi = (
            mean_ci95(
                d["global_cost"]
            )
        )

        # Satisfaction CIs
        psat, psat_lo, psat_hi = (
            mean_ci95(
                d["personalized_satisfaction"]
            )
        )

        gsat, gsat_lo, gsat_hi = (
            mean_ci95(
                d["global_satisfaction"]
            )
        )

        # Paired cost improvement CI
        imp, imp_lo, imp_hi = (
            mean_ci95(
                d["cost_improvement"]
            )
        )

        # --------------------------------------------------------
        # Constraint probability
        # --------------------------------------------------------

        p_success = int(
            np.sum(
                1.0
                - d[
                    "personalized_violation"
                ]
            )
        )

        g_success = int(
            np.sum(
                1.0
                - d[
                    "global_violation"
                ]
            )
        )

        pcp, pcp_lo, pcp_hi = (
            proportion_ci95_exact(
                p_success,
                len(d),
            )
        )

        gcp, gcp_lo, gcp_hi = (
            proportion_ci95_exact(
                g_success,
                len(d),
            )
        )

        summary_rows.append(
            {
                "user":
                    user_id,

                # Personalized cost
                "personalized_cost":
                    pcost,

                "personalized_cost_ci95_low":
                    pcost_lo,

                "personalized_cost_ci95_high":
                    pcost_hi,

                # Global cost
                "global_cost":
                    gcost,

                "global_cost_ci95_low":
                    gcost_lo,

                "global_cost_ci95_high":
                    gcost_hi,

                # Personalized satisfaction
                "personalized_satisfaction":
                    psat,

                "personalized_satisfaction_ci95_low":
                    psat_lo,

                "personalized_satisfaction_ci95_high":
                    psat_hi,

                # Global satisfaction
                "global_satisfaction":
                    gsat,

                "global_satisfaction_ci95_low":
                    gsat_lo,

                "global_satisfaction_ci95_high":
                    gsat_hi,

                # Personalized constraint
                "personalized_constraint_probability":
                    pcp,

                "personalized_constraint_ci95_low":
                    pcp_lo,

                "personalized_constraint_ci95_high":
                    pcp_hi,

                # Global constraint
                "global_constraint_probability":
                    gcp,

                "global_constraint_ci95_low":
                    gcp_lo,

                "global_constraint_ci95_high":
                    gcp_hi,

                # Cost improvement
                "mean_cost_improvement":
                    imp,

                "cost_improvement_ci95_low":
                    imp_lo,

                "cost_improvement_ci95_high":
                    imp_hi,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        ERRORBAR_OUT
        / "personalization_corrected_summary.csv",
        index=False,
    )

    print(
        "\nPERSONALIZATION RESULTS:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Overall paired comparison
    # ------------------------------------------------------------

    overall_personalized_cost, _, _ = (
        mean_ci95(
            trials[
                "personalized_cost"
            ]
        )
    )

    overall_global_cost, _, _ = (
        mean_ci95(
            trials[
                "global_cost"
            ]
        )
    )

    overall_improvement, overall_lo, overall_hi = (
        mean_ci95(
            trials[
                "cost_improvement"
            ]
        )
    )

    print(
        "\nOVERALL PERSONALIZATION:"
    )

    print(
        f"Global cost:        "
        f"{overall_global_cost:.6f}"
    )

    print(
        f"Personalized cost:  "
        f"{overall_personalized_cost:.6f}"
    )

    print(
        f"Cost improvement:   "
        f"{overall_improvement:.6f}"
    )

    print(
        f"95% CI:             "
        f"[{overall_lo:.6f}, "
        f"{overall_hi:.6f}]"
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    x = np.arange(
        n_users
    )

    width = 0.35

    # ------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 4.5)
    )

    py = summary[
        "personalized_cost"
    ].to_numpy()

    gy = summary[
        "global_cost"
    ].to_numpy()

    pyerr = np.vstack(
        [
            py
            - summary[
                "personalized_cost_ci95_low"
            ].to_numpy(),

            summary[
                "personalized_cost_ci95_high"
            ].to_numpy()
            - py,
        ]
    )

    gyerr = np.vstack(
        [
            gy
            - summary[
                "global_cost_ci95_low"
            ].to_numpy(),

            summary[
                "global_cost_ci95_high"
            ].to_numpy()
            - gy,
        ]
    )

    plt.bar(
        x - width / 2,
        py,
        width,
        yerr=pyerr,
        capsize=4,
        label="Personalized",
    )

    plt.bar(
        x + width / 2,
        gy,
        width,
        yerr=gyerr,
        capsize=4,
        label="Global",
    )

    plt.xticks(
        x,
        [
            f"User {i}"
            for i in range(
                1,
                n_users + 1,
            )
        ],
    )

    plt.ylabel(
        "Average cost"
    )

    plt.title(
        "Personalization: routing cost"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "personalization_cost_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    # ------------------------------------------------------------
    # Satisfaction
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 4.5)
    )

    py = summary[
        "personalized_satisfaction"
    ].to_numpy()

    gy = summary[
        "global_satisfaction"
    ].to_numpy()

    py_lo = summary[
        "personalized_satisfaction_ci95_low"
    ].to_numpy()

    py_hi = summary[
        "personalized_satisfaction_ci95_high"
    ].to_numpy()

    gy_lo = summary[
        "global_satisfaction_ci95_low"
    ].to_numpy()

    gy_hi = summary[
        "global_satisfaction_ci95_high"
    ].to_numpy()

    pyerr = np.vstack(
        [
            np.maximum(
                0.0,
                py - py_lo,
            ),
            np.maximum(
                0.0,
                py_hi - py,
            ),
        ]
    )

    gyerr = np.vstack(
        [
            np.maximum(
                0.0,
                gy - gy_lo,
            ),
            np.maximum(
                0.0,
                gy_hi - gy,
            ),
        ]
    )

    plt.bar(
        x - width / 2,
        py,
        width,
        yerr=pyerr,
        capsize=4,
        label="Personalized",
    )

    plt.bar(
        x + width / 2,
        gy,
        width,
        yerr=gyerr,
        capsize=4,
        label="Global",
    )

    plt.axhline(
        1.0 - MAIN_EPSILON,
        linestyle="--",
        label="Required satisfaction",
    )

    plt.xticks(
        x,
        [
            f"User {i}"
            for i in range(
                1,
                n_users + 1,
            )
        ],
    )

    plt.ylabel(
        "Mean satisfaction"
    )

    plt.title(
        "Personalization: satisfaction"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "personalization_satisfaction_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    # ------------------------------------------------------------
    # Cost improvement
    # ------------------------------------------------------------

    y = summary[
        "mean_cost_improvement"
    ].to_numpy()

    yerr = np.vstack(
        [
            y
            - summary[
                "cost_improvement_ci95_low"
            ].to_numpy(),

            summary[
                "cost_improvement_ci95_high"
            ].to_numpy()
            - y,
        ]
    )

    plt.figure(
        figsize=(8, 4.5)
    )

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=5,
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xticks(
        x,
        [
            f"User {i}"
            for i in range(
                1,
                n_users + 1,
            )
        ],
    )

    plt.ylabel(
        "Global cost − Personalized cost"
    )

    plt.title(
        "Personalization cost improvement"
    )

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "personalization_cost_improvement_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    return (
        trials,
        summary,
    )

# ================================================================
# 18. EXPERIMENT 6
#     ABLATIONS
# ================================================================

def experiment_ablation():

    print(
        "\n" + "=" * 72
    )

    print(
        "EXPERIMENT 6: ABLATIONS"
    )

    print(
        "=" * 72
    )

    methods = {

        "Ours":
            (
                True,
                True,
                True,
            ),

        "No-UCB":
            (
                False,
                True,
                True,
            ),

        "No-Queue":
            (
                True,
                False,
                True,
            ),

        "No-Rho":
            (
                True,
                True,
                False,
            ),
    }

    all_trials = []

    summaries = []

    for name, (
        use_ucb,
        use_queue,
        use_rho,
    ) in methods.items():

        print(
            f"\nRunning {name}..."
        )

        config = TrialConfig(
            use_ucb=use_ucb,
            use_queue=use_queue,
            use_rho=use_rho,
        )

        df = run_trials(
            config,
            N_ABLATION,
            seed0=SEED + 60000,
            T_run=BASE_T,
        )

        df["method"] = name

        all_trials.append(
            df
        )

        s = summarize_trials(
            df
        )

        s["method"] = name

        summaries.append(
            s
        )

    trials = pd.concat(
        all_trials,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summaries
    )

    trials.to_csv(
        ERRORBAR_OUT
        / "ablation_trials.csv",
        index=False,
    )

    summary.to_csv(
        ERRORBAR_OUT
        / "ablation_errorbars.csv",
        index=False,
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # Cost
    y = summary[
        "mean_cost"
    ].to_numpy()

    yerr = np.vstack(
        [
            y
            - summary[
                "cost_ci95_low"
            ].to_numpy(),

            summary[
                "cost_ci95_high"
            ].to_numpy()
            - y,
        ]
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.bar(
        summary["method"],
        y,
        yerr=yerr,
        capsize=5,
    )

    plt.ylabel(
        "Average cost"
    )

    plt.title(
        "Ablation: cost"
    )

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "ablation_cost_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    # Satisfaction
    y = summary[
        "mean_satisfaction"
    ].to_numpy()

    yerr = np.vstack(
        [
            y
            - summary[
                "satisfaction_ci95_low"
            ].to_numpy(),

            summary[
                "satisfaction_ci95_high"
            ].to_numpy()
            - y,
        ]
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.bar(
        summary["method"],
        y,
        yerr=yerr,
        capsize=5,
    )

    plt.axhline(
        1.0 - MAIN_EPSILON,
        linestyle="--",
    )

    plt.ylabel(
        "Mean satisfaction"
    )

    plt.title(
        "Ablation: satisfaction"
    )

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "ablation_satisfaction_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    # Constraint
    y = summary[
        "constraint_probability"
    ].to_numpy()

    yerr = np.vstack(
        [
            y
            - summary[
                "constraint_ci95_low"
            ].to_numpy(),

            summary[
                "constraint_ci95_high"
            ].to_numpy()
            - y,
        ]
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.bar(
        summary["method"],
        y,
        yerr=yerr,
        capsize=5,
    )

    plt.axhline(
        1.0 - MAIN_DELTA,
        linestyle="--",
    )

    plt.ylabel(
        "Constraint probability"
    )

    plt.title(
        "Ablation: chance constraint"
    )

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "ablation_constraint_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    return (
        trials,
        summary,
    )


# ================================================================
# 19. EXPERIMENT 7
#     CONTEXT ROBUSTNESS
# ================================================================

def experiment_robustness():

    print("\n" + "=" * 72)
    print("EXPERIMENT 7: CONTEXT ROBUSTNESS")
    print("=" * 72)

    # ------------------------------------------------
    # Fixed preference direction and theta*
    # ------------------------------------------------
    rng = np.random.default_rng(SEED + 70000)

    # Fixed reference direction
    theta_direction = rng.normal(size=D)
    theta_direction /= np.linalg.norm(theta_direction)

    # Very small perturbation so theta* is close to the
    # fixed reference direction
    perturbation = rng.normal(size=D)
    perturbation /= np.linalg.norm(perturbation)

    gamma = 0.02

    theta_star = theta_direction + gamma * perturbation
    theta_star /= np.linalg.norm(theta_star)

    # Report alignment
    alignment = np.dot(theta_star, theta_direction)
    angle_deg = np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))

    print(f"\nTheta* alignment with fixed direction: {alignment:.6f}")
    print(f"Theta* angle from fixed direction: {angle_deg:.3f} degrees")

    # ------------------------------------------------
    # Vary only the context noise
    # ------------------------------------------------
    xi_values = [
        0.0,
        0.20,
        0.35,
        0.50,
        0.75,
        1.0,
    ]

    all_trials = []
    summaries = []

    for xi in xi_values:

        print(f"\nRunning xi={xi:.2f}")

        config = TrialConfig(
            xi=xi,
            theta_star=theta_star,
        )

        df = run_trials(
            config,
            N_ROBUSTNESS,
            seed0=SEED + 70000,
            T_run=BASE_T,
        )

        df["xi"] = xi

        all_trials.append(df)

        s = summarize_trials(df)
        s["xi"] = xi
        summaries.append(s)

    trials = pd.concat(
        all_trials,
        ignore_index=True,
    )

    summary = (
        pd.DataFrame(summaries)
        .sort_values("xi")
    )

    trials.to_csv(
        ERRORBAR_OUT / "robustness_trials.csv",
        index=False,
    )

    summary.to_csv(
        ERRORBAR_OUT / "robustness_errorbars.csv",
        index=False,
    )

    print("\n" + summary.to_string(index=False))

    x = summary["xi"].to_numpy()

    # ------------------------------------------------
    # Cost
    # ------------------------------------------------
    y = summary["mean_cost"].to_numpy()

    yerr = np.vstack([
        y - summary["cost_ci95_low"].to_numpy(),
        summary["cost_ci95_high"].to_numpy() - y,
    ])

    plt.figure(figsize=(7, 4.5))

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=5,
    )

    plt.xlabel(r"Context noise level $\xi$")
    plt.ylabel("Average cost")

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT / "robustness_cost_95ci.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    # ------------------------------------------------
    # Satisfaction
    # ------------------------------------------------
    y = summary["mean_satisfaction"].to_numpy()

    yerr = np.vstack([
        y - summary["satisfaction_ci95_low"].to_numpy(),
        summary["satisfaction_ci95_high"].to_numpy() - y,
    ])

    plt.figure(figsize=(7, 4.5))

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=5,
    )

    plt.axhline(
        1.0 - MAIN_EPSILON,
        linestyle="--",
    )

    plt.xlabel(r"Context noise level $\xi$")
    plt.ylabel("Mean satisfaction")

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT / "robustness_satisfaction_95ci.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    # ------------------------------------------------
    # Chance constraint
    # ------------------------------------------------
    y = summary["constraint_probability"].to_numpy()

    yerr = np.vstack([
        y - summary["constraint_ci95_low"].to_numpy(),
        summary["constraint_ci95_high"].to_numpy() - y,
    ])

    plt.figure(figsize=(7, 4.5))

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=5,
    )

    plt.axhline(
        1.0 - MAIN_DELTA,
        linestyle="--",
    )

    plt.xlabel(r"Context noise level $\xi$")
    plt.ylabel("Constraint probability")

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT / "robustness_constraint_95ci.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    return (
        trials,
        summary,
    )


# ================================================================
# 20. STATIONARY ORACLE FOR EXPERIMENT 8
# ================================================================

def compute_stationary_oracle(
    P,
):
    """
    Compute the cheapest stationary randomized policy
    satisfying the expected satisfaction requirement.

    Given mean true satisfaction for each action:

        q_a = average_t p_{t,a}

    solve:

        minimize sum_a pi_a c_a

        subject to:
            sum_a pi_a q_a >= 1-epsilon
            sum_a pi_a = 1
            pi_a >= 0

    Fallback has q=1 and cost=1.
    """

    mean_p = np.mean(
        P,
        axis=0,
    )

    q = np.concatenate(
        [
            mean_p,
            np.array([1.0]),
        ]
    )

    costs = np.concatenate(
        [
            COSTS,
            np.array([FALLBACK_COST]),
        ]
    )

    n_actions = K + 1

    c = costs.copy()

    A_eq = np.ones(
        (
            1,
            n_actions,
        )
    )

    b_eq = np.array(
        [1.0]
    )

    A_ub = -q.reshape(
        1,
        -1,
    )

    b_ub = np.array(
        [
            -(
                1.0
                - MAIN_EPSILON
            )
        ]
    )

    result = minimize(
    lambda pi: np.dot(pi, c),
    np.ones(
        n_actions
    )
    / n_actions,
        method="SLSQP",
        bounds=[
            (
                0.0,
                1.0,
            )
            for _ in range(
                n_actions
            )
        ],
        constraints=[
            {
                "type":
                    "eq",

                "fun":
                    lambda pi:
                    np.sum(pi)
                    - 1.0,
            },
            {
                "type":
                    "ineq",

                "fun":
                    lambda pi:
                    np.dot(
                        pi,
                        q,
                    )
                    - (
                        1.0
                        - MAIN_EPSILON
                    ),
            },
        ],
        options={
            "maxiter": 500,
            "ftol": 1e-10,
        },
    )

    if not result.success:

        # --------------------------------------------------------
        # Robust fallback: choose cheapest action satisfying
        # expected constraint individually, otherwise fallback.
        # --------------------------------------------------------

        feasible = np.where(
            q
            >= (
                1.0
                - MAIN_EPSILON
            )
        )[0]

        if len(feasible) > 0:

            best = feasible[
                np.argmin(
                    costs[
                        feasible
                    ]
                )
            ]

            pi = np.zeros(
                n_actions
            )

            pi[best] = 1.0

        else:

            pi = np.zeros(
                n_actions
            )

            pi[FALLBACK] = 1.0

    else:

        pi = result.x

        pi = np.maximum(
            pi,
            0.0,
        )

        total = np.sum(pi)

        if total <= 1e-12:

            pi = np.zeros(
                n_actions
            )

            pi[FALLBACK] = 1.0

        else:

            pi /= total

    expected_cost = float(
        np.dot(
            pi,
            costs,
        )
    )

    expected_satisfaction = float(
        np.dot(
            pi,
            q,
        )
    )

    return (
        pi,
        expected_cost,
        expected_satisfaction,
        q,
    )


# ================================================================
# 21. EXPERIMENT 8
#     FINITE-HORIZON COST-GAP SCALING
# ================================================================

def experiment_regret():

    print(
        "\n" + "=" * 72
    )

    print(
        "EXPERIMENT 8: FINITE-HORIZON COST-GAP SCALING"
    )

    print(
        "=" * 72
    )

    all_rows = []

    oracle_composition_rows = []

    for T_value in HORIZONS:

        print(
            f"\nRunning T={T_value}"
        )

        eta = horizon_eta(
            T_value
        )

        rho = horizon_rho(
            T_value
        )

        for trial in range(
            N_REGRET
        ):

            seed = (
                SEED
                + 80000
                + T_value
                * 10000
                + trial
            )

            rng = np.random.default_rng(
                seed
            )

            # ----------------------------------------------------
            # Generate one fixed environment
            # ----------------------------------------------------

            theta_star = (
                unit(
                    rng.normal(
                        size=D
                    )
                )
                * S
            )

            model_dirs = (
                make_model_directions(
                    rng
                )
            )

            X = generate_contexts(
                rng=rng,
                theta_star=theta_star,
                model_dirs=model_dirs,
                xi=XI,
                user_dir=unit(
                    theta_star
                ),
                shared_mode=False,
                T_override=T_value,
            )

            P = expit(
                np.einsum(
                    "tkd,d->tk",
                    X,
                    theta_star,
                )
            )

            # ----------------------------------------------------
            # Oracle
            # ----------------------------------------------------

            (
                pi,
                oracle_cost,
                oracle_sat,
                q,
            ) = compute_stationary_oracle(
                P
            )

            oracle_composition_rows.append(
                {
                    "T":
                        T_value,

                    "trial":
                        trial,

                    **{
                        f"oracle_q_action_{a}":
                            pi[a]
                        for a in range(K)
                    },

                    "oracle_q_fallback":
                        pi[FALLBACK],
                }
            )

            # ----------------------------------------------------
            # Algorithm
            # ----------------------------------------------------

            config = TrialConfig(
                epsilon=MAIN_EPSILON,
                delta=MAIN_DELTA,
                xi=XI,
                eta=eta,
                rho=rho,
            )

            result = run_trial(
                config=config,
                seed=seed + 1,
                T_run=T_value,
            )

            alg_cost = (
                result[
                    "avg_cost"
                ]
            )

            alg_sat = (
                result[
                    "satisfaction"
                ]
            )

            gap_per_round = (
                alg_cost
                - oracle_cost
            )

            cumulative_gap = (
                T_value
                * gap_per_round
            )

            all_rows.append(
                {
                    "T":
                        T_value,

                    "trial":
                        trial,

                    "algorithm_cost":
                        alg_cost,

                    "oracle_cost":
                        oracle_cost,

                    "cost_gap_per_round":
                        gap_per_round,

                    "cumulative_cost_gap":
                        cumulative_gap,

                    "satisfaction":
                        alg_sat,

                    "violation":
                        float(
                            alg_sat
                            < (
                                1.0
                                - MAIN_EPSILON
                            )
                        ),

                    "eta":
                        eta,

                    "rho":
                        rho,

                    "oracle_satisfaction":
                        oracle_sat,

                    "oracle_expected_cost":
                        oracle_cost,
                }
            )

        print(
            f"Finished T={T_value}"
        )

    trials = pd.DataFrame(
        all_rows
    )

    oracle_df = pd.DataFrame(
        oracle_composition_rows
    )

    trials.to_csv(
        OUT
        / "regret_scaling.csv",
        index=False,
    )

    oracle_df.to_csv(
        OUT
        / "oracle_policy_composition_trials.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Summary + CIs
    # ------------------------------------------------------------

    rows = []

    for T_value, group in trials.groupby(
        "T"
    ):

        ac, acl, ach = mean_ci95(
            group[
                "algorithm_cost"
            ]
        )

        oc, ocl, och = mean_ci95(
            group[
                "oracle_cost"
            ]
        )

        gap, gl, gh = mean_ci95(
            group[
                "cost_gap_per_round"
            ]
        )

        cum, cml, cmh = mean_ci95(
            group[
                "cumulative_cost_gap"
            ]
        )

        sat, sl, sh = mean_ci95(
            group[
                "satisfaction"
            ]
        )

        successes = int(
            np.sum(
                1.0
                - group["violation"]
            )
        )

        cp, cpl, cph = (
            proportion_ci95_exact(
                successes,
                len(group),
            )
        )

        rows.append(
            {
                "T":
                    T_value,

                "algorithm_cost":
                    ac,

                "algorithm_cost_ci95_low":
                    acl,

                "algorithm_cost_ci95_high":
                    ach,

                "oracle_cost":
                    oc,

                "oracle_cost_ci95_low":
                    ocl,

                "oracle_cost_ci95_high":
                    och,

                "cost_gap_per_round":
                    gap,

                "cost_gap_ci95_low":
                    gl,

                "cost_gap_ci95_high":
                    gh,

                "cumulative_cost_gap":
                    cum,

                "cumulative_gap_ci95_low":
                    cml,

                "cumulative_gap_ci95_high":
                    cmh,

                "satisfaction":
                    sat,

                "satisfaction_ci95_low":
                    sl,

                "satisfaction_ci95_high":
                    sh,

                "constraint_probability":
                    cp,

                "constraint_ci95_low":
                    cpl,

                "constraint_ci95_high":
                    cph,

            }
        )

    summary = pd.DataFrame(
        rows
    ).sort_values(
        "T"
    )

    summary.to_csv(
        OUT
        / "regret_scaling_errorbars.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Empirical scaling exponent
    # ------------------------------------------------------------

    positive = (
        summary[
            "cumulative_cost_gap"
        ]
        > 0
    )

    if positive.sum() >= 2:

        xlog = np.log(
            summary.loc[
                positive,
                "T",
            ].to_numpy()
        )

        ylog = np.log(
            summary.loc[
                positive,
                "cumulative_cost_gap",
            ].to_numpy()
        )

        slope, intercept = np.polyfit(
            xlog,
            ylog,
            1,
        )

    else:

        slope = np.nan
        intercept = np.nan

    print(
        "\nREGRET / COST-GAP SUMMARY:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nEmpirical cumulative-gap exponent: "
        f"{slope:.6f}"
    )

    # ------------------------------------------------------------
    # Algorithm vs oracle
    # ------------------------------------------------------------

    x = summary[
        "T"
    ].to_numpy()

    ya = summary[
        "algorithm_cost"
    ].to_numpy()

    yo = summary[
        "oracle_cost"
    ].to_numpy()

    yaerr = np.vstack(
        [
            ya
            - summary[
                "algorithm_cost_ci95_low"
            ].to_numpy(),

            summary[
                "algorithm_cost_ci95_high"
            ].to_numpy()
            - ya,
        ]
    )

    yoerr = np.vstack(
        [
            yo
            - summary[
                "oracle_cost_ci95_low"
            ].to_numpy(),

            summary[
                "oracle_cost_ci95_high"
            ].to_numpy()
            - yo,
        ]
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.errorbar(
        x,
        ya,
        yerr=yaerr,
        marker="o",
        capsize=4,
        label="Algorithm",
    )

    plt.errorbar(
        x,
        yo,
        yerr=yoerr,
        marker="o",
        capsize=4,
        label="Stationary oracle",
    )

    plt.xlabel(
        "Horizon T"
    )

    plt.ylabel(
        "Average cost"
    )

    plt.title(
        "Algorithm vs stationary oracle"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "regret_algorithm_vs_oracle_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    # ------------------------------------------------------------
    # Per-round gap
    # ------------------------------------------------------------

    y = summary[
        "cost_gap_per_round"
    ].to_numpy()

    yerr = np.vstack(
        [
            y
            - summary[
                "cost_gap_ci95_low"
            ].to_numpy(),

            summary[
                "cost_gap_ci95_high"
            ].to_numpy()
            - y,
        ]
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=4,
    )

    plt.xlabel(
        "Horizon T"
    )

    plt.ylabel(
        "Cost gap per round"
    )

    plt.title(
        "Per-round cost gap vs horizon"
    )

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "regret_cost_gap_per_round_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    # ------------------------------------------------------------
    # Cumulative gap
    # ------------------------------------------------------------

    positive = (
        (
            summary[
                "cumulative_cost_gap"
            ]
            > 0
        )
        &
        (
            summary[
                "cumulative_gap_ci95_low"
            ]
            > 0
        )
    )

    if positive.any():

        s = summary.loc[
            positive
        ]

        x2 = s[
            "T"
        ].to_numpy()

        y2 = s[
            "cumulative_cost_gap"
        ].to_numpy()

        err2 = np.vstack(
            [
                y2
                - s[
                    "cumulative_gap_ci95_low"
                ].to_numpy(),

                s[
                    "cumulative_gap_ci95_high"
                ].to_numpy()
                - y2,
            ]
        )

        plt.figure(
            figsize=(7, 4.5)
        )

        plt.errorbar(
            x2,
            y2,
            yerr=err2,
            marker="o",
            capsize=4,
        )

        plt.xscale(
            "log"
        )

        plt.yscale(
            "log"
        )

        plt.xlabel(
            "Horizon T"
        )

        plt.ylabel(
            "Cumulative cost gap"
        )

        plt.title(
            "Cumulative cost-gap scaling"
        )

        plt.tight_layout()

        plt.savefig(
            ERRORBAR_OUT
            / "regret_cumulative_gap_loglog_95ci.png",
            dpi=200,
        )

        plt.show()

        plt.close()

    # ------------------------------------------------------------
    # Satisfaction
    # ------------------------------------------------------------

    y = summary[
        "satisfaction"
    ].to_numpy()

    yerr = np.vstack(
        [
            y
            - summary[
                "satisfaction_ci95_low"
            ].to_numpy(),

            summary[
                "satisfaction_ci95_high"
            ].to_numpy()
            - y,
        ]
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=4,
    )

    plt.axhline(
        1.0 - MAIN_EPSILON,
        linestyle="--",
    )

    plt.xlabel(
        "Horizon T"
    )

    plt.ylabel(
        "Mean satisfaction"
    )

    plt.title(
        "Satisfaction vs horizon"
    )

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "regret_satisfaction_vs_T_95ci.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    # ------------------------------------------------------------
    # Oracle policy composition
    # ------------------------------------------------------------

    oracle_cols = [
        f"oracle_q_action_{a}"
        for a in range(K)
    ]

    oracle_cols.append(
        "oracle_q_fallback"
    )

    oracle_policy = (
        oracle_df
        .groupby("T")[
            oracle_cols
        ]
        .mean()
    )

    bottom = np.zeros(
        len(
            oracle_policy
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    labels = [
        "Model 1",
        "Model 2",
        "Model 3",
        "Model 4",
        "Model 5",
        "Fallback",
    ]

    for col, label in zip(
        oracle_cols,
        labels,
    ):

        values = (
            oracle_policy[
                col
            ].to_numpy()
        )

        plt.bar(
            oracle_policy.index,
            values,
            bottom=bottom,
            label=label,
        )

        bottom += values

    plt.xlabel(
        "Horizon T"
    )

    plt.ylabel(
        "Oracle policy probability"
    )

    plt.title(
        "Stationary oracle policy composition"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ERRORBAR_OUT
        / "oracle_policy_composition.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    return (
        trials,
        summary,
        slope,
    )


# ================================================================
# 22. EXPERIMENT 9
#     CONFIDENCE CALIBRATION
# ================================================================

def experiment_calibration():

    print(
        "\n" + "=" * 72
    )

    print(
        "EXPERIMENT 9: CONFIDENCE CALIBRATION"
    )

    print(
        "=" * 72
    )

    all_errors = []

    all_theory_widths = []

    all_calibrated_widths = []

    for trial in range(
        N_CALIBRATION
    ):

        rng = np.random.default_rng(
            SEED
            + 90000
            + trial
        )

        theta_star = (
            unit(
                rng.normal(
                    size=D
                )
            )
            * S
        )

        model_dirs = (
            make_model_directions(
                rng
            )
        )

        X = generate_contexts(
            rng=rng,
            theta_star=theta_star,
            model_dirs=model_dirs,
            xi=XI,
            user_dir=unit(
                theta_star
            ),
            shared_mode=False,
            T_override=BASE_T,
        )

        P = expit(
            np.einsum(
                "tkd,d->tk",
                X,
                theta_star,
            )
        )

        V = (
            BETA
            * np.eye(D)
        )

        history_X = []

        history_y = []

        for t in range(
            BASE_T
        ):

            if len(
                history_y
            ) > 0:

                theta_hat = logistic_mle(
                    np.asarray(
                        history_X
                    ),
                    np.asarray(
                        history_y
                    ),
                )

            else:

                theta_hat = np.zeros(D)

            Vinv = np.linalg.pinv(
                V
            )

            theory_alpha = theoretical_alpha(
                t + 1
            )

            calibrated_alpha = empirical_alpha(
                t + 1,
                FROZEN_ALPHA_SCALE,
            )

            # Evaluate all learned actions.
            for a in range(K):

                x = X[t, a]

                estimate = float(
                    expit(
                        x @ theta_hat
                    )
                )

                uncertainty = math.sqrt(
                    max(
                        0.0,
                        x
                        @ Vinv
                        @ x,
                    )
                )

                theory_width = (
                    theory_alpha
                    * uncertainty
                )

                calibrated_width = (
                    calibrated_alpha
                    * uncertainty
                )

                error = abs(
                    P[t, a]
                    - estimate
                )

                all_errors.append(
                    error
                )

                all_theory_widths.append(
                    theory_width
                )

                all_calibrated_widths.append(
                    calibrated_width
                )

            # Sample an action only to generate learning data.
            action = rng.integers(
                0,
                K,
            )

            y = int(
                rng.random()
                < P[t, action]
            )

            history_X.append(
                X[t, action].copy()
            )

            history_y.append(
                y
            )

            V += np.outer(
                X[t, action],
                X[t, action],
            )

    errors = np.asarray(
        all_errors
    )

    theory_widths = np.asarray(
        all_theory_widths
    )

    calibrated_widths = np.asarray(
        all_calibrated_widths
    )

    theory_ratio = (
        errors
        / np.maximum(
            theory_widths,
            1e-12,
        )
    )

    calibrated_ratio = (
        errors
        / np.maximum(
            calibrated_widths,
            1e-12,
        )
    )

    summary = pd.DataFrame(
        [
            {
                "T":
                    BASE_T,

                "mean_error":
                    np.mean(errors),

                "median_error":
                    np.median(errors),

                "q95_error":
                    np.quantile(
                        errors,
                        0.95,
                    ),

                "mean_theory_width":
                    np.mean(
                        theory_widths
                    ),

                "mean_calibrated_width":
                    np.mean(
                        calibrated_widths
                    ),

                "median_error_over_theory_width":
                    np.median(
                        theory_ratio
                    ),

                "median_error_over_calibrated_width":
                    np.median(
                        calibrated_ratio
                    ),

                "q95_error_over_theory_width":
                    np.quantile(
                        theory_ratio,
                        0.95,
                    ),

                "q95_error_over_calibrated_width":
                    np.quantile(
                        calibrated_ratio,
                        0.95,
                    ),

                "fraction_error_below_theory_width":
                    np.mean(
                        errors
                        <= theory_widths
                    ),

                "fraction_error_below_calibrated_width":
                    np.mean(
                        errors
                        <= calibrated_widths
                    ),

                "alpha_scale_frozen":
                    FROZEN_ALPHA_SCALE,
            }
        ]
    )

    summary.to_csv(
        OUT
        / "confidence_calibration.csv",
        index=False,
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Histogram of normalized errors
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 4.5)
    )

    plt.hist(
        theory_ratio,
        bins=100,
        alpha=0.65,
        label="Error / theoretical width",
    )

    plt.hist(
        calibrated_ratio,
        bins=100,
        alpha=0.65,
        label="Error / calibrated width",
    )

    plt.axvline(
        1.0,
        linestyle="--",
        label="Ratio = 1",
    )

    plt.xlabel(
        "Absolute error / confidence width"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Confidence-radius calibration"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT
        / "confidence_calibration.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    # ------------------------------------------------------------
    # Error vs calibrated width
    # ------------------------------------------------------------

    # Subsample only for visualization if enormous.
    rng_plot = np.random.default_rng(
        SEED
    )

    n_plot = min(
        100000,
        len(errors),
    )

    indices = rng_plot.choice(
        len(errors),
        size=n_plot,
        replace=False,
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.scatter(
        calibrated_widths[
            indices
        ],
        errors[
            indices
        ],
        s=4,
        alpha=0.15,
    )

    max_value = max(
        np.max(
            calibrated_widths[
                indices
            ]
        ),
        np.max(
            errors[
                indices
            ]
        ),
    )

    plt.plot(
        [
            0,
            max_value,
        ],
        [
            0,
            max_value,
        ],
        linestyle="--",
    )

    plt.xlabel(
        "Calibrated confidence width"
    )

    plt.ylabel(
        "Absolute prediction error"
    )

    plt.title(
        "Prediction error vs calibrated width"
    )

    plt.tight_layout()

    plt.savefig(
        OUT
        / "confidence_error_vs_width.png",
        dpi=200,
    )

    plt.show()

    plt.close()

    return summary


# ================================================================
# 23. RUN COMPLETE SUITE
# ================================================================

print(
    "\n"
    + "=" * 72
)

print(
    "STARTING COMPLETE EXPERIMENT SUITE"
)

print(
    "=" * 72
)


# ------------------------------------------------
# Experiment 1
# ------------------------------------------------

main_trials, main_summary = (
    experiment_main()
)


# ------------------------------------------------
# Experiment 2
# ------------------------------------------------

chance_trials, chance_summary = (
    experiment_chance_constraint()
)


# ------------------------------------------------
# Experiment 3
# ------------------------------------------------

epsilon_trials, epsilon_summary = (
    experiment_epsilon()
)


# ------------------------------------------------
# Experiment 4
# ------------------------------------------------

delta_trials, delta_summary = (
    experiment_delta()
)


# ------------------------------------------------ex
# Experiment 5
# ------------------------------------------------

personalization_trials, personalization_summary = (
    experiment_personalization()
)


# ------------------------------------------------
# Experiment 6
# ------------------------------------------------

ablation_trials, ablation_summary = (
    experiment_ablation()
)


# ------------------------------------------------
# Experiment 7
# ------------------------------------------------

robustness_trials, robustness_summary = (
    experiment_robustness()
)


# ------------------------------------------------
# Experiment 8
# ------------------------------------------------

regret_trials, regret_summary, regret_exponent = (
    experiment_regret()
)


# ------------------------------------------------
# Experiment 9
# ------------------------------------------------

calibration_summary = (
    experiment_calibration()
)


# ================================================================
# 24. FINAL SUMMARY
# ================================================================

print(
    "\n"
    + "=" * 72
)

print(
    "COMPLETE EXPERIMENT SUITE FINISHED"
)

print(
    "=" * 72
)

print(
    "\nAll results were written to:"
)

print(
    OUT.resolve()
)

print(
    "\nError-bar results:"
)

print(
    ERRORBAR_OUT.resolve()
)

print(
    "\nReference protocol:"
)

print(
    f"Reference T       = {REFERENCE_T}"
)

print(
    f"Reference eta     = {REFERENCE_ETA}"
)

print(
    f"Reference rho     = {REFERENCE_RHO}"
)

print(
    f"Frozen alpha      = {FROZEN_ALPHA_SCALE}"
)

print(
    "\nAt T=1000:"
)

print(
    f"eta = {horizon_eta(1000):.6f}"
)

print(
    f"rho = {horizon_rho(1000):.6f}"
)

print(
    "\nAt T=100:"
)

print(
    f"eta = {horizon_eta(100):.6f}"
)

print(
    f"rho = {horizon_rho(100):.6f}"
)

print(
    "\nTheoretical rho_max* at T=100:"
)

print(
    f"{rho_max_star(100, MAIN_EPSILON, MAIN_DELTA):.9f}"
)

print(
    "\nTheoretical rho_max* at T=1000:"
)

print(
    f"{rho_max_star(1000, MAIN_EPSILON, MAIN_DELTA):.9f}"
)

print(
    "\nEmpirical Experiment 8 exponent:"
)

print(
    f"{regret_exponent:.6f}"
)

print(
    "\nIMPORTANT INTERPRETATION:"
)

print(
    "The Experiment 8 exponent is an empirical scaling estimate;"
)

print(
    "it should NOT be presented as proof of a theoretical O(sqrt(T)) bound."
)

print(
    "\nExperiment 9's calibrated alpha_scale=0.005 is an empirical"
)

print(
    "calibration, not a theoretically valid 95% confidence guarantee."
)

print(
    "\nExperiment 5 is now a genuine global-vs-personalized learner"
)

print(
    "comparison using the same counterfactual environments."
)

print(
    "\nDONE."
)
