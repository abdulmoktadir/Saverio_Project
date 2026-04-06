import io
import math
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import pulp
except Exception:
    pulp = None


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="m-ITARA + Fuzzy OPA + Fuzzy Bonferroni CoCoSo",
    page_icon="📊",
    layout="wide",
)

st.title("📊 m-ITARA + Fuzzy OPA + Fuzzy Bonferroni CoCoSo")
st.caption(
    "Integrated app: m-ITARA for objective weight determination, "
    "fuzzy OPA for subjective weight determination, "
    "and fuzzy Bonferroni CoCoSo for performance evaluation."
)

EPS = 1e-12


# ============================================================
# TRIANGULAR FUZZY SCALE
# ============================================================
LINGUISTIC_SCALE = {
    "AL": (1.00, 1.50, 2.50),
    "VL": (1.50, 2.50, 3.50),
    "L":  (2.50, 3.50, 4.50),
    "ML": (3.50, 4.50, 5.50),
    "E":  (4.50, 5.50, 6.50),
    "MH": (5.50, 6.50, 7.50),
    "H":  (6.50, 7.50, 8.50),
    "VH": (7.50, 8.50, 9.50),
    "AH": (8.50, 9.00, 10.00),
}
LINGUISTIC_OPTIONS = list(LINGUISTIC_SCALE.keys())


# ============================================================
# HELPERS
# ============================================================
def parse_number(x):
    """Convert strings like '7,400,000.00' to float; keep floats as-is."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace(",", "")
    if re.match(r"^\(\s*[-+]?\d+(\.\d+)?\s*\)$", s):
        s = "-" + s.strip("()").strip()
    return float(s)


def safe_normalize_to_1(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = np.nansum(v)
    if len(v) == 0:
        raise ValueError("Vector length cannot be zero.")
    if s <= 0 or np.isclose(s, 0.0) or np.isnan(s):
        return np.ones_like(v) / len(v)
    return v / s


def ensure_2d(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a


def safe_pos(x: float, eps: float = EPS) -> float:
    return max(float(x), eps)


def defuzz_tfn(tfn):
    """Graded mean integration representation for TFN."""
    return (tfn[0] + 4 * tfn[1] + tfn[2]) / 6.0


def to_bc_label(x: str) -> str:
    s = str(x).strip().upper()
    if s in {"B", "BENEFIT", "MAX"}:
        return "B"
    return "C"


def validate_linguistic_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].astype(str).str.strip().str.upper()
        out[c] = out[c].where(out[c].isin(LINGUISTIC_OPTIONS), "E")
    return out


def check_missing_df(df: pd.DataFrame, name: str):
    if df.isna().any().any():
        raise ValueError(f"{name} contains blank or invalid cells.")


def check_missing_array(arr, name: str):
    arr = np.asarray(arr, dtype=float)
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains blank or invalid values.")
    return arr


# ============================================================
# OBJECTIVE PART: SAME AS YOUR STANDALONE m-ITARA APP
# ============================================================

# ----------------------------
# Stage I: Modified ITARA (w1)
# ----------------------------
def mitara_stage1_independent_weights(D, AL, IT):
    """
    D: m x n matrix
    AL: n vector (aspiration)
    IT: n vector (threshold)
    Implements:
      - Append AL as (m+1)th row
      - Normalize by column sum
      - Sort (beta), differences (gamma)
      - delta = max(gamma - NIT, 0)
      - nu_j = sqrt(sum delta^2)
      - w1 = nu / sum(nu)
    """
    D = ensure_2d(D)
    AL = np.asarray(AL, dtype=float).reshape(1, -1)
    IT = np.asarray(IT, dtype=float).ravel()

    m, n = D.shape
    if AL.shape[1] != n or IT.shape[0] != n:
        raise ValueError("Dimension mismatch in Stage I inputs.")

    D_ext = np.vstack([D, AL])  # (m+1) x n
    col_sum = np.sum(D_ext, axis=0)
    if np.any(np.isclose(col_sum, 0.0)):
        raise ValueError("Some criterion column sums are 0 (cannot normalize).")

    A = D_ext / col_sum
    NIT = IT / col_sum

    beta = np.sort(A, axis=0)                  # (m+1) x n
    gamma = beta[1:, :] - beta[:-1, :]         # m x n
    delta = np.maximum(gamma - NIT.reshape(1, -1), 0.0)
    nu = np.sqrt(np.sum(delta**2, axis=0))     # n
    w1 = safe_normalize_to_1(nu)

    return {
        "D_ext": D_ext,
        "A": A,
        "NIT": NIT,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "nu": nu,
        "w1": w1,
    }


# ----------------------------
# Stage II: Dependent weights (Correlation) (w2)
# ----------------------------
def stage2_dependent_weights_corr(D, types_bc):
    """
    Normalize each criterion column across alternatives to [0,1]:
      Benefit: (x-min)/(max-min)
      Cost:    (max-x)/(max-min)
    Then Pearson correlation matrix R across criteria columns
    phi_j = sum_{j'} (1 - r_jj')
    w2 = phi / sum(phi)
    """
    D = ensure_2d(D)
    types_bc = [str(t).strip().upper() for t in types_bc]

    m, n = D.shape
    if len(types_bc) != n:
        raise ValueError("types_bc length must equal number of criteria (n).")

    X = np.zeros_like(D, dtype=float)

    for j in range(n):
        col = D[:, j].astype(float)
        mn, mx = np.nanmin(col), np.nanmax(col)
        if np.isclose(mx - mn, 0.0) or np.isnan(mx - mn):
            X[:, j] = 0.0
            continue

        if types_bc[j] == "B":
            X[:, j] = (col - mn) / (mx - mn)
        elif types_bc[j] == "C":
            X[:, j] = (mx - col) / (mx - mn)
        else:
            raise ValueError(f"Invalid type at criterion {j+1}: {types_bc[j]} (use B or C)")

    R = np.corrcoef(X, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)

    phi = np.sum(1.0 - R, axis=1)
    w2 = safe_normalize_to_1(phi)

    return {"X": X, "R": R, "phi": phi, "w2": w2}


# ----------------------------
# Stage III: EWCS fusion
# ----------------------------
def ewcs_fusion(w1, w2):
    """
    Solve:
      [w1·w1  w1·w2] [p1] = [w1·w1]
      [w2·w1  w2·w2] [p2]   [w2·w2]
    Then p* = |p|/sum(|p|), wagg = p1*w1 + p2*w2, normalize to sum=1
    """
    w1 = np.asarray(w1, dtype=float).ravel()
    w2 = np.asarray(w2, dtype=float).ravel()
    if w1.shape != w2.shape:
        raise ValueError("w1 and w2 must have the same length.")

    a11 = float(np.dot(w1, w1))
    a12 = float(np.dot(w1, w2))
    a22 = float(np.dot(w2, w2))

    A = np.array([[a11, a12], [a12, a22]], dtype=float)
    b = np.array([a11, a22], dtype=float)

    det = np.linalg.det(A)
    if np.isclose(det, 0.0):
        p_raw = np.array([1.0, 1.0])
    else:
        p_raw = np.linalg.solve(A, b)

    p_star = np.abs(p_raw)
    p_star = p_star / np.sum(p_star)

    wagg = p_star[0] * w1 + p_star[1] * w2
    wagg = safe_normalize_to_1(wagg)

    return {"p_raw": p_raw, "p_star": p_star, "wagg": wagg}


# ============================================================
# FUZZY OPA: MATCHED TO YOUR STANDALONE OPA MODEL
# ============================================================
def trig_geom_component(values, weights):
    """
    Trigonometric aggregation component matching your standalone model:
    aggregated = sum(values) * (2/pi) * acos( product( cos(pi * v/sum(values) / 2) ** w ) )
    """
    s = sum(values)
    if s == 0:
        return 0.0

    prod = 1.0
    for v, w in zip(values, weights):
        ratio = v / s
        term = np.cos(np.pi * ratio / 2) ** w
        prod *= term

    prod = np.clip(prod, -1.0, 1.0)
    return s * (2 / np.pi) * np.arccos(prod)


def aggregate_tfn(tfn_list, weights):
    ls = [t[0] for t in tfn_list]
    ms = [t[1] for t in tfn_list]
    us = [t[2] for t in tfn_list]
    l = trig_geom_component(ls, weights)
    m = trig_geom_component(ms, weights)
    u = trig_geom_component(us, weights)
    return (l, m, u)


def solve_fuzzy_opa(coeff_list, n):
    """
    Exact LP structure from your standalone fuzzy OPA model.
    """
    if pulp is None:
        raise RuntimeError("PuLP is not installed. Please install pulp to run the fuzzy OPA solver.")

    prob = pulp.LpProblem("Triangular_Fuzzy_OPA", pulp.LpMaximize)

    w_l = [pulp.LpVariable(f"w_l_{i}", lowBound=0) for i in range(n)]
    w_m = [pulp.LpVariable(f"w_m_{i}", lowBound=0) for i in range(n)]
    w_u = [pulp.LpVariable(f"w_u_{i}", lowBound=0) for i in range(n)]

    psi_l = pulp.LpVariable("psi_l", lowBound=0)
    psi_m = pulp.LpVariable("psi_m", lowBound=0)
    psi_u = pulp.LpVariable("psi_u", lowBound=0)

    prob += (psi_l + 2 * psi_m + psi_u) / 4

    for i in range(n):
        prob += w_l[i] <= w_m[i]
        prob += w_m[i] <= w_u[i]

    prob += pulp.lpSum(w_l) == 0.8
    prob += pulp.lpSum(w_m) == 1.0
    prob += pulp.lpSum(w_u) == 1.2

    for a in range(n - 1):
        prob += coeff_list[a][0] * (w_l[a] - w_u[a + 1]) >= psi_l
        prob += coeff_list[a][1] * (w_m[a] - w_m[a + 1]) >= psi_m
        prob += coeff_list[a][2] * (w_u[a] - w_l[a + 1]) >= psi_u

    prob += coeff_list[n - 1][0] * w_l[n - 1] >= psi_l
    prob += coeff_list[n - 1][1] * w_m[n - 1] >= psi_m
    prob += coeff_list[n - 1][2] * w_u[n - 1] >= psi_u

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if status != 1:
        return None, None

    weights = []
    for i in range(n):
        weights.append((
            max(0, pulp.value(w_l[i])),
            max(0, pulp.value(w_m[i])),
            max(0, pulp.value(w_u[i]))
        ))

    psi = (
        max(0, pulp.value(psi_l)),
        max(0, pulp.value(psi_m)),
        max(0, pulp.value(psi_u))
    )
    return weights, psi


def run_fuzzy_opa(criteria_names, linguistic_df, expert_weights):
    """
    Full fuzzy OPA flow matched to your standalone model:
    1. Aggregate expert linguistic TFNs
    2. Defuzzify theta
    3. Build coefficients
    4. Rank criteria by defuzzified theta
    5. Solve LP in sorted order
    6. Map results back to original criterion order
    """
    n = len(criteria_names)

    theta = []
    defuzz_values = []
    for crit in criteria_names:
        tfn_list = [LINGUISTIC_SCALE[linguistic_df.loc[crit, e]] for e in linguistic_df.columns]
        aggregated = aggregate_tfn(tfn_list, expert_weights)
        theta.append(aggregated)
        defuzz_values.append(defuzz_tfn(aggregated))

    min_l = min(t[0] for t in theta if t[0] > 0) if any(t[0] > 0 for t in theta) else 1.0

    coeff = []
    for t in theta:
        if min(t) <= 0:
            coeff.append((0.0, 0.0, 0.0))
        else:
            coeff.append((min_l / t[2], min_l / t[1], min_l / t[0]))

    sorted_indices = np.argsort(defuzz_values)[::-1]
    coeff_sorted = [coeff[idx] for idx in sorted_indices]
    theta_sorted = [theta[idx] for idx in sorted_indices]
    criteria_sorted = [criteria_names[idx] for idx in sorted_indices]

    weights_sorted, psi = solve_fuzzy_opa(coeff_sorted, n)
    if weights_sorted is None:
        raise ValueError("Fuzzy OPA optimization failed. Please check your linguistic inputs and expert weights.")

    weights = [None] * n
    for rank, idx in enumerate(sorted_indices):
        weights[idx] = weights_sorted[rank]

    ranked_criteria = [(sorted_indices[k], weights_sorted[k]) for k in range(n)]

    theta_df = pd.DataFrame({
        "Criterion": criteria_names,
        "l": [t[0] for t in theta],
        "m": [t[1] for t in theta],
        "u": [t[2] for t in theta],
        "Defuzzified": defuzz_values,
    })

    coeff_df = pd.DataFrame({
        "Criterion": criteria_names,
        "Coeff l": [c[0] for c in coeff],
        "Coeff m": [c[1] for c in coeff],
        "Coeff u": [c[2] for c in coeff],
    })

    weights_df = pd.DataFrame({
        "Criterion": criteria_names,
        "l": [w[0] for w in weights],
        "m": [w[1] for w in weights],
        "u": [w[2] for w in weights],
        "Defuzzified": [defuzz_tfn(w) for w in weights],
    })

    ranked_df = pd.DataFrame({
        "Rank": list(range(1, n + 1)),
        "Criterion": [criteria_names[idx] for idx, _ in ranked_criteria],
        "Weight l": [w[0] for _, w in ranked_criteria],
        "Weight m": [w[1] for _, w in ranked_criteria],
        "Weight u": [w[2] for _, w in ranked_criteria],
        "Defuzzified": [defuzz_tfn(w) for _, w in ranked_criteria],
    })

    return {
        "theta": theta,
        "defuzz_values": defuzz_values,
        "coeff": coeff,
        "sorted_indices": sorted_indices,
        "criteria_sorted": criteria_sorted,
        "theta_sorted": theta_sorted,
        "coeff_sorted": coeff_sorted,
        "weights": weights,
        "weights_sorted": weights_sorted,
        "psi": psi,
        "theta_df": theta_df,
        "coeff_df": coeff_df,
        "weights_df": weights_df,
        "ranked_df": ranked_df,
    }


# ============================================================
# FUZZY INPUT CONVERSION
# ============================================================
def build_fuzzy_df_from_crisp_percent(raw_df: pd.DataFrame, epsilon: float = 0.10) -> pd.DataFrame:
    cols = []
    rows = []

    for c in raw_df.columns:
        cols.extend([f"{c}_l", f"{c}_m", f"{c}_u"])

    for _, r in raw_df.iterrows():
        row_vals = []
        for c in raw_df.columns:
            x = float(r[c])
            l = x * (1.0 - epsilon)
            m = x
            u = x * (1.0 + epsilon)
            trip = sorted([l, m, u])
            row_vals.extend(trip)
        rows.append(row_vals)

    return pd.DataFrame(rows, index=raw_df.index, columns=cols)


def sanitize_fuzzy_df(fuzzy_df: pd.DataFrame, criteria, alternatives) -> pd.DataFrame:
    out = fuzzy_df.copy().astype(float)
    for c in criteria:
        trip = out[[f"{c}_l", f"{c}_m", f"{c}_u"]].values
        trip = np.sort(trip, axis=1)
        out[[f"{c}_l", f"{c}_m", f"{c}_u"]] = trip
    return out.reindex(index=alternatives)


def fuzzy_df_to_nested_matrix(fuzzy_df: pd.DataFrame, criteria, alternatives):
    matrix = []
    for alt in alternatives:
        row = []
        for c in criteria:
            trip = (
                float(fuzzy_df.loc[alt, f"{c}_l"]),
                float(fuzzy_df.loc[alt, f"{c}_m"]),
                float(fuzzy_df.loc[alt, f"{c}_u"]),
            )
            row.append(tuple(sorted(trip)))
        matrix.append(row)
    return matrix


# ============================================================
# FUZZY BONFERRONI CoCoSo
# REVISED: USE DEFUZZIFIED SCoB AND PCoB FOR KIA/KIB/KIC/RANKING
# ============================================================
def normalize_cocoso_bonferroni(decision, types_bc):
    n_alt = len(decision)
    n_crit = len(types_bc)
    norm = [[(0.0, 0.0, 0.0) for _ in range(n_crit)] for _ in range(n_alt)]

    for j in range(n_crit):
        typ = to_bc_label(types_bc[j])

        if typ == "B":
            max_u = max(decision[i][j][2] for i in range(n_alt))
            max_u = safe_pos(max_u)
            for i in range(n_alt):
                l, m, u = decision[i][j]
                norm[i][j] = (l / max_u, m / max_u, u / max_u)
        else:
            min_l = min(decision[i][j][0] for i in range(n_alt))
            min_l = safe_pos(min_l)
            for i in range(n_alt):
                l, m, u = decision[i][j]
                l = safe_pos(l)
                m = safe_pos(m)
                u = safe_pos(u)
                norm[i][j] = (min_l / u, min_l / m, min_l / l)

    return norm


def compute_bonferroni(norm_matrix, weights, phi1=1.0, phi2=1.0):
    """
    SCoB uses outer power 1 / (phi1 + phi2)
    PCoB uses product form and then divides by (phi1 + phi2)
    """
    weights = safe_normalize_to_1(pd.Series(weights).astype(float).values)

    n_alt = len(norm_matrix)
    n_crit = len(weights)

    if n_crit < 2:
        raise ValueError("At least two criteria are required for fuzzy Bonferroni CoCoSo.")

    scob = []
    pcob = []
    exp_term = 1.0 / safe_pos(phi1 + phi2)

    for a in range(n_alt):
        s_l = 0.0
        s_m = 0.0
        s_u = 0.0

        log_p_l = 0.0
        log_p_m = 0.0
        log_p_u = 0.0

        for i in range(n_crit):
            wi = min(max(weights[i], EPS), 1.0 - EPS)
            denom = 1.0 - wi

            for j in range(n_crit):
                if i == j:
                    continue

                wj = weights[j]
                term = (wi * wj) / denom

                gi_l, gi_m, gi_u = norm_matrix[a][i]
                gj_l, gj_m, gj_u = norm_matrix[a][j]

                s_l += term * (safe_pos(gi_l) ** phi1) * (safe_pos(gj_l) ** phi2)
                s_m += term * (safe_pos(gi_m) ** phi1) * (safe_pos(gj_m) ** phi2)
                s_u += term * (safe_pos(gi_u) ** phi1) * (safe_pos(gj_u) ** phi2)

                base_l = safe_pos(phi1 * gi_l + phi2 * gj_l)
                base_m = safe_pos(phi1 * gi_m + phi2 * gj_m)
                base_u = safe_pos(phi1 * gi_u + phi2 * gj_u)

                log_p_l += term * math.log(base_l)
                log_p_m += term * math.log(base_m)
                log_p_u += term * math.log(base_u)

        s_l = safe_pos(s_l) ** exp_term
        s_m = safe_pos(s_m) ** exp_term
        s_u = safe_pos(s_u) ** exp_term
        scob.append((s_l, s_m, s_u))

        p_l = math.exp(log_p_l) / safe_pos(phi1 + phi2)
        p_m = math.exp(log_p_m) / safe_pos(phi1 + phi2)
        p_u = math.exp(log_p_u) / safe_pos(phi1 + phi2)
        pcob.append((p_l, p_m, p_u))

    return scob, pcob


def relative_significance(scob, pcob, pi=0.5):
    """
    Revised Excel-style flow:
    1. Defuzzify SCoB and PCoB using SAME TFN formula:
         Crisp = (l + 4m + u) / 6
    2. Compute Kia, Kib, Kic from these crisp values.
    """
    scob_crisp = np.array([defuzz_tfn(s) for s in scob], dtype=float)
    pcob_crisp = np.array([defuzz_tfn(p) for p in pcob], dtype=float)

    sum_scob = safe_pos(np.sum(scob_crisp))
    sum_pcob = safe_pos(np.sum(pcob_crisp))

    min_scob = safe_pos(np.min(scob_crisp))
    min_pcob = safe_pos(np.min(pcob_crisp))

    max_scob = safe_pos(np.max(scob_crisp))
    max_pcob = safe_pos(np.max(pcob_crisp))

    kia = (scob_crisp + pcob_crisp) / safe_pos(sum_scob + sum_pcob)
    kib = (scob_crisp / min_scob) + (pcob_crisp / min_pcob)
    kic = (pi * scob_crisp + (1.0 - pi) * pcob_crisp) / safe_pos(
        pi * max_scob + (1.0 - pi) * max_pcob
    )

    audit_df = pd.DataFrame({
        "sum_Crisp_SCoB": [sum_scob],
        "sum_Crisp_PCoB": [sum_pcob],
        "min_Crisp_SCoB": [min_scob],
        "min_Crisp_PCoB": [min_pcob],
        "max_Crisp_SCoB": [max_scob],
        "max_Crisp_PCoB": [max_pcob],
        "pi": [pi],
    })

    return scob_crisp, pcob_crisp, kia, kib, kic, audit_df


def final_scores_bonferroni(
    scob,
    pcob,
    scob_crisp,
    pcob_crisp,
    kia,
    kib,
    kic,
    alternative_names=None,
):
    n_alt = len(kia)
    if alternative_names is None:
        alternative_names = [f"A{i+1}" for i in range(n_alt)]

    rows = []
    for i in range(n_alt):
        K = (
            (safe_pos(kia[i]) * safe_pos(kib[i]) * safe_pos(kic[i])) ** (1.0 / 3.0)
            + (kia[i] + kib[i] + kic[i]) / 3.0
        )

        rows.append([
            alternative_names[i],
            scob[i][0], scob[i][1], scob[i][2],
            pcob[i][0], pcob[i][1], pcob[i][2],
            scob_crisp[i],
            pcob_crisp[i],
            kia[i],
            kib[i],
            kic[i],
            K,
            K,   # keep compatibility with existing plot/export expecting "Crisp"
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "Alternative",
            "SCoB_l", "SCoB_m", "SCoB_u",
            "PCoB_l", "PCoB_m", "PCoB_u",
            "Crisp_SCoB",
            "Crisp_PCoB",
            "Kia",
            "Kib",
            "Kic",
            "K",
            "Crisp",
        ],
    )
    df["Rank"] = df["K"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["K", "Alternative"], ascending=[False, True]).reset_index(drop=True)


def cocoso_bonferroni_from_app(fuzzy_df, types_bc, final_weights, phi1=1.0, phi2=1.0, pi=0.5):
    criteria = [c[:-2] for c in fuzzy_df.columns if c.endswith("_l")]
    alternatives = fuzzy_df.index.astype(str).tolist()

    decision = fuzzy_df_to_nested_matrix(fuzzy_df, criteria, alternatives)
    norm_matrix = normalize_cocoso_bonferroni(decision, types_bc)
    weights = pd.Series(final_weights, index=criteria).astype(float)

    scob, pcob = compute_bonferroni(norm_matrix, weights, phi1=phi1, phi2=phi2)
    scob_crisp, pcob_crisp, kia, kib, kic, audit_df = relative_significance(scob, pcob, pi=pi)

    ranking_df = final_scores_bonferroni(
        scob=scob,
        pcob=pcob,
        scob_crisp=scob_crisp,
        pcob_crisp=pcob_crisp,
        kia=kia,
        kib=kib,
        kic=kic,
        alternative_names=alternatives,
    )

    norm_rows = []
    for i, alt in enumerate(alternatives):
        row = {"Alternative": alt}
        for j, c in enumerate(criteria):
            row[f"{c}_l"] = norm_matrix[i][j][0]
            row[f"{c}_m"] = norm_matrix[i][j][1]
            row[f"{c}_u"] = norm_matrix[i][j][2]
        norm_rows.append(row)
    norm_df = pd.DataFrame(norm_rows).set_index("Alternative")

    scob_df = pd.DataFrame(scob, columns=["SCoB_l", "SCoB_m", "SCoB_u"], index=alternatives)
    scob_df["Crisp_SCoB"] = scob_crisp

    pcob_df = pd.DataFrame(pcob, columns=["PCoB_l", "PCoB_m", "PCoB_u"], index=alternatives)
    pcob_df["Crisp_PCoB"] = pcob_crisp

    psi_a_df = pd.DataFrame({"Kia": kia}, index=alternatives)
    psi_b_df = pd.DataFrame({"Kib": kib}, index=alternatives)
    psi_c_df = pd.DataFrame({"Kic": kic}, index=alternatives)

    return ranking_df, {"phi1": phi1, "phi2": phi2, "pi": pi}, norm_df, scob_df, pcob_df, psi_a_df, psi_b_df, psi_c_df, audit_df


# ============================================================
# EXPORT
# ============================================================
def build_excel_bytes(sheets):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31])
    out.seek(0)
    return out.getvalue()


# ============================================================
# UI
# ============================================================
with st.sidebar:
    st.header("Input method")
    mode = st.radio("Choose input method", ["Manual entry", "Upload file (CSV/Excel)"], index=0)

    st.divider()
    st.header("Problem size")
    m = st.number_input("Number of alternatives (m)", min_value=2, max_value=50, value=3, step=1)
    n = st.number_input("Number of criteria (n)", min_value=2, max_value=30, value=11, step=1)
    k = st.number_input("Number of experts (k)", min_value=1, max_value=20, value=7, step=1)

    st.divider()
    st.header("Bonferroni parameters")
    phi1 = st.number_input("ϕ1", min_value=0.1, value=1.0, step=0.1)
    phi2 = st.number_input("ϕ2", min_value=0.1, value=1.0, step=0.1)
    pi_coef = st.slider("π", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.divider()
    epsilon = st.number_input(
        "ε for fuzzy conversion",
        min_value=0.00,
        max_value=1.00,
        value=0.10,
        step=0.01,
        help="Automatic fuzzy conversion: ((1-ε)x, x, (1+ε)x)",
    )

    st.divider()
    st.write("- Type: **B** = Benefit, **C** = Cost")
    st.write("- Objective block follows real data/crisp data.")
    st.write("- Fuzzy OPA use PuLP")
    st.write("- Fuzzy Bonferroni CoCoSo uses defuzzified SCoB and PCoB for Kia/Kib/Kic.")
    if pulp is None:
        st.warning("PuLP is not installed. Please install it: pip install pulp")

alt_names = [f"A{i+1}" for i in range(int(m))]
crit_names = [f"C{j+1}" for j in range(int(n))]
expert_names = [f"E{i+1}" for i in range(int(k))]

D_df = None
types = None
AL = None
WL = None
IT = None

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1) Inputs", "2) m-ITARA", "3) Fuzzy OPA", "4) Final weights", "5) Fuzzy Bonferroni CoCoSo"]
)

with tab1:
    if mode == "Upload file (CSV/Excel)":
        st.subheader("Upload your data")
        up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

        st.info(
            "Expected structure:\n"
            "- A sheet (or CSV) with columns: Alt, C1..Cn\n"
            "- Types/AL/WL/IT can be entered below."
        )

        if up is not None:
            if up.name.lower().endswith(".csv"):
                raw = pd.read_csv(up)
                if "Alt" in raw.columns:
                    raw = raw.set_index("Alt")
                D_candidate = raw.copy()
            else:
                xls = pd.ExcelFile(up)
                sheet = st.selectbox("Select sheet containing Decision Matrix", xls.sheet_names)
                raw = pd.read_excel(xls, sheet_name=sheet)
                if "Alt" in raw.columns:
                    raw = raw.set_index("Alt")
                D_candidate = raw.copy()

            cols_named = [c for c in D_candidate.columns if str(c).strip().upper() in [cn.upper() for cn in crit_names]]
            if len(cols_named) >= int(n):
                D_candidate = D_candidate[crit_names]
            else:
                D_candidate = D_candidate.iloc[:, : int(n)]
                D_candidate.columns = crit_names

            D_candidate = D_candidate.iloc[: int(m), :]
            D_candidate.index = alt_names
            D_df = D_candidate.map(parse_number)

            st.success("Decision matrix loaded.")
            st.dataframe(D_df, use_container_width=True)

        st.subheader("Types (B/C), AL, WL, IT")
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

        with col1:
            types_df = st.data_editor(
                pd.DataFrame({"Type (B/C)": ["B"] * int(n)}, index=crit_names),
                use_container_width=True,
                height=380,
                key="types_upload",
            )
        with col2:
            al_df = st.data_editor(
                pd.DataFrame({"AL": [0.0] * int(n)}, index=crit_names),
                use_container_width=True,
                height=380,
                key="al_upload",
            )
        with col3:
            wl_df = st.data_editor(
                pd.DataFrame({"WL": [0.0] * int(n)}, index=crit_names),
                use_container_width=True,
                height=380,
                key="wl_upload",
            )
        with col4:
            it_df = st.data_editor(
                pd.DataFrame({"IT": [0.0] * int(n)}, index=crit_names),
                use_container_width=True,
                height=380,
                key="it_upload",
            )

        types = types_df["Type (B/C)"].astype(str).tolist()
        AL = al_df["AL"].apply(parse_number).to_numpy(dtype=float)
        WL = wl_df["WL"].apply(parse_number).to_numpy(dtype=float)
        IT = it_df["IT"].apply(parse_number).to_numpy(dtype=float)

    else:
        st.subheader("Enter your data (manual)")

        # Your current dataset default when m=3, n=11, k=7
        if int(m) == 3 and int(n) == 11 and int(k) == 7:
            default_case1 = pd.DataFrame(
                [
                    [21.623, 4274.163, 0.929, 2.896, 31.119, 0.00019, 103994.797, 0.000144, 8.821, 0.094, 2959.053],
                    [27.142, 5113.711, 1.862, 3.652, 38.375, 0.00021, 130388.771, 0.000163, 10.748, 0.089, 3516.777],
                    [11993005.999, 600014.237, 359.863, 6302642.068, 69021242.718, 0.08799, 8324418.205, 25.971, 16203294.671, 52.214, 15667781.034],
                ],
                index=alt_names,
                columns=crit_names,
            )
            default_types = ["C"] * 11
            default_AL = np.array([10.812, 2137.081, 0.464, 1.448, 15.560, 0.00009, 51997.398, 0.000, 4.411, 0.047, 1479.527], dtype=float)
            default_WL = np.array([14991257.499, 750017.796, 449.828, 7878302.585, 86276553.398, 0.110, 10405522.756, 32.464, 20254118.338, 65.268, 19584726.293], dtype=float)
            default_IT = np.array([2826772.876, 140318.445, 84.492, 1485546.210, 16268454.734, 0.021, 1934469.322, 6.121, 3819150.874, 12.285, 3692168.223], dtype=float)
        elif int(m) == 4 and int(n) == 11:
            default_case1 = pd.DataFrame(
                [
                    ["7,400,000.00", 98.40, 73.10, "50,800.00", 4.10, 90.10, 2.40, 20.40, 393.00, 10.60, 3.50],
                    ["33,500,000.00", 78.20, 66.50, "144,600.00", 4.10, 84.00, 6.40, 10.10, 347.00, 8.00, 1.83],
                    ["10,850,000.00", 96.00, 61.40, "255,000.00", 4.00, 75.60, 1.10, 10.40, 275.00, 6.30, 7.52],
                    ["29,200.00", 86.60, 82.30, "657,000.00", 5.60, 89.30, 4.60, 21.50, 150.00, 12.90, 3.99],
                ],
                index=alt_names,
                columns=crit_names,
            )
            default_types = ["B", "B", "B", "B", "B", "B", "C", "B", "B", "B", "B"]
            default_AL = np.array([35000000, 100, 90, 660000, 6, 95, 1, 25, 400, 15, 10], dtype=float)
            default_WL = np.array([25000, 75, 55, 50000, 4, 75, 7, 10, 100, 5, 1], dtype=float)
            default_IT = np.array([1000000, 10, 10, 50000, 0, 0, 2, 7, 50, 2, 3], dtype=float)
        else:
            default_case1 = pd.DataFrame(np.nan, index=alt_names, columns=crit_names)
            default_types = ["B"] * int(n)
            default_AL = np.zeros(int(n), dtype=float)
            default_WL = np.zeros(int(n), dtype=float)
            default_IT = np.zeros(int(n), dtype=float)

        st.markdown("**Decision matrix (A × C)**")
        D_edit = st.data_editor(
            default_case1,
            use_container_width=True,
            height=260,
            key="D_manual",
        )
        D_df = D_edit.map(parse_number)

        st.markdown("**Types (B/C), AL, WL, IT (per criterion)**")
        meta_default = pd.DataFrame(
            {
                "Type (B/C)": default_types,
                "AL": default_AL,
                "WL": default_WL,
                "IT": default_IT,
            },
            index=crit_names,
        )
        meta_edit = st.data_editor(meta_default, use_container_width=True, height=320, key="meta_manual")

        types = meta_edit["Type (B/C)"].astype(str).tolist()
        AL = meta_edit["AL"].apply(parse_number).to_numpy(dtype=float)
        WL = meta_edit["WL"].apply(parse_number).to_numpy(dtype=float)
        IT = meta_edit["IT"].apply(parse_number).to_numpy(dtype=float)

    st.subheader("Expert weights")
    expert_weight_default = pd.DataFrame(
        {"Weight": [1.0 / int(k)] * int(k)},
        index=expert_names,
    )
    expert_weight_df = st.data_editor(
        expert_weight_default,
        use_container_width=True,
        key="expert_weights",
    )
    expert_weights = expert_weight_df["Weight"].astype(float).to_list()
    sum_expert_weights = sum(expert_weights)
    st.write(f"**Sum of expert weights: {sum_expert_weights:.6f}**")

    st.subheader("Linguistic judgments for criteria")

    if int(m) == 3 and int(n) == 11 and int(k) == 7 and mode == "Manual entry":
        linguistic_default = pd.DataFrame(
            {
                "E1": ["ML", "AH", "AH", "AH", "ML", "VH", "VH", "AH", "VH", "AH", "AH"],
                "E2": ["MH", "VH", "MH", "MH", "MH", "H", "VH", "H", "H", "H", "MH"],
                "E3": ["VH", "VH", "L", "L", "L", "E", "AH", "AH", "E", "E", "E"],
                "E4": ["L", "VH", "H", "ML", "E", "H", "VH", "VH", "AH", "VH", "H"],
                "E5": ["AL", "AH", "L", "L", "L", "VL", "H", "ML", "L", "H", "H"],
                "E6": ["H", "AH", "VH", "H", "H", "E", "H", "VH", "H", "H", "H"],
                "E7": ["H", "VH", "ML", "E", "ML", "H", "VH", "H", "H", "ML", "H"],
            },
            index=crit_names,
        )
    else:
        linguistic_default = pd.DataFrame("E", index=crit_names, columns=expert_names)

    linguistic_df = st.data_editor(
        linguistic_default,
        use_container_width=True,
        height=280,
        key="linguistic_input",
    )
    linguistic_df = validate_linguistic_df(linguistic_df)

    st.subheader("Fuzzy matrix for Bonferroni CoCoSo")
    auto_fuzzy = st.checkbox(
        "Automatically convert raw data into TFNs using ((1-ε)x, x, (1+ε)x)",
        value=True,
    )

    expected_cols = []
    for c in crit_names:
        expected_cols.extend([f"{c}_l", f"{c}_m", f"{c}_u"])

    if D_df is None:
        D_df = pd.DataFrame(np.nan, index=alt_names, columns=crit_names)

    if auto_fuzzy:
        D_num_preview = D_df.fillna(0.0).astype(float)
        fuzzy_df = build_fuzzy_df_from_crisp_percent(D_num_preview, epsilon=epsilon)
        st.caption(f"Automatic fuzzy conversion is active with ε = {epsilon:.2f}")
        st.dataframe(fuzzy_df, use_container_width=True)
    else:
        manual_fuzzy_default = pd.DataFrame(
            np.ones((len(alt_names), len(expected_cols))),
            index=alt_names,
            columns=expected_cols,
        )
        fuzzy_df = st.data_editor(
            manual_fuzzy_default,
            use_container_width=True,
            height=280,
            key="manual_fuzzy",
        )
        fuzzy_df = sanitize_fuzzy_df(fuzzy_df, criteria=crit_names, alternatives=alt_names)

st.divider()
st.subheader("Compute integrated model")
run = st.button("▶️ Run m-ITARA + Fuzzy OPA + Fuzzy Bonferroni CoCoSo", type="primary")

if run:
    try:
        if D_df is None or D_df.shape != (int(m), int(n)):
            raise ValueError("Decision matrix shape mismatch. Please ensure it is m × n.")

        if any(str(t).strip().upper() not in ["B", "C"] for t in types):
            raise ValueError("Types must be B or C for all criteria.")

        check_missing_df(D_df, "Decision matrix")
        AL = check_missing_array(AL, "AL")
        WL = check_missing_array(WL, "WL")
        IT = check_missing_array(IT, "IT")

        if abs(sum_expert_weights - 1.0) > 1e-6:
            raise ValueError("Expert weights must sum to 1.0 to match the standalone fuzzy OPA model.")

        D_num = D_df.to_numpy(dtype=float)

        sigma = D_num.std(axis=0, ddof=0)
        it_ok = IT < sigma

        stage1 = mitara_stage1_independent_weights(D_num, AL, IT)
        stage2 = stage2_dependent_weights_corr(D_num, types)
        stage3_obj = ewcs_fusion(stage1["w1"], stage2["w2"])

        opa_result = run_fuzzy_opa(crit_names, linguistic_df, expert_weights)
        subjective_crisp = opa_result["weights_df"]["Defuzzified"].to_numpy(dtype=float)

        final_stage = ewcs_fusion(stage3_obj["wagg"], subjective_crisp)
        final_weights = pd.Series(final_stage["wagg"], index=crit_names, name="final_weight")

        ranking_df, cocoso_meta, norm_df, scob_df, pcob_df, psi_a_df, psi_b_df, psi_c_df, cocoso_audit_df = cocoso_bonferroni_from_app(
            fuzzy_df=fuzzy_df,
            types_bc=types,
            final_weights=final_weights.values,
            phi1=float(phi1),
            phi2=float(phi2),
            pi=float(pi_coef),
        )

        # ----------------------------------------------------
        # TAB 2: m-ITARA
        # ----------------------------------------------------
        with tab2:
            summary = pd.DataFrame(
                {
                    "Criterion": crit_names,
                    "Type": [str(t).strip().upper() for t in types],
                    "sigma (STDEV.P)": sigma,
                    "IT": IT,
                    "IT < sigma?": it_ok,
                    "w1 (independent)": stage1["w1"],
                    "w2 (dependent)": stage2["w2"],
                    "wagg (EWCS)": stage3_obj["wagg"],
                }
            )
            summary["Rank (wagg)"] = summary["wagg (EWCS)"].rank(ascending=False, method="dense").astype(int)
            summary = summary.sort_values("Rank (wagg)").reset_index(drop=True)

            st.subheader("Objective weights & ranking")
            st.dataframe(summary, use_container_width=True)

            c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
            with c1:
                st.metric("EWCS p*(w1)", float(stage3_obj["p_star"][0]))
            with c2:
                st.metric("EWCS p*(w2)", float(stage3_obj["p_star"][1]))
            with c3:
                st.write("p_raw =", np.array2string(stage3_obj["p_raw"], precision=6))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Stage I: A**")
                st.dataframe(
                    pd.DataFrame(stage1["A"], index=alt_names + ["AL"], columns=crit_names),
                    use_container_width=True,
                )
                st.markdown("**Stage I: beta**")
                st.dataframe(pd.DataFrame(stage1["beta"], columns=crit_names), use_container_width=True)
                st.markdown("**Stage I: NIT**")
                st.dataframe(pd.DataFrame({"NIT": stage1["NIT"]}, index=crit_names), use_container_width=True)

            with col2:
                st.markdown("**Stage I: gamma**")
                st.dataframe(pd.DataFrame(stage1["gamma"], columns=crit_names), use_container_width=True)
                st.markdown("**Stage I: delta**")
                st.dataframe(pd.DataFrame(stage1["delta"], columns=crit_names), use_container_width=True)
                st.markdown("**Stage I: nu**")
                st.dataframe(pd.DataFrame({"nu": stage1["nu"]}, index=crit_names), use_container_width=True)

            st.subheader("Stage II: correlation-based dependent weights")
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Normalized matrix X**")
                st.dataframe(
                    pd.DataFrame(stage2["X"], index=alt_names, columns=crit_names),
                    use_container_width=True,
                )
            with col4:
                st.markdown("**Correlation matrix R**")
                st.dataframe(
                    pd.DataFrame(stage2["R"], index=crit_names, columns=crit_names),
                    use_container_width=True,
                )
                st.markdown("**phi and w2**")
                st.dataframe(
                    pd.DataFrame({"phi": stage2["phi"], "w2": stage2["w2"]}, index=crit_names),
                    use_container_width=True,
                )

            bad = summary[summary["IT < sigma?"] == False]
            if len(bad) > 0:
                st.warning(
                    "Some criteria have IT ≥ sigma. Consider reducing IT for these criteria: "
                    + ", ".join(bad["Criterion"].tolist())
                )
            else:
                st.success("All criteria satisfy IT < sigma (recommended check).")

        # ----------------------------------------------------
        # TAB 3: Fuzzy OPA
        # ----------------------------------------------------
        with tab3:
            st.subheader("Normalized expert weights")
            st.dataframe(
                pd.DataFrame({
                    "Expert": expert_names,
                    "Weight": expert_weights,
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Aggregated triangular fuzzy importance (Theta)")
            st.dataframe(opa_result["theta_df"], use_container_width=True, hide_index=True)

            st.subheader("Coefficients for Fuzzy OPA")
            st.dataframe(opa_result["coeff_df"], use_container_width=True, hide_index=True)

            st.subheader("Triangular fuzzy weights")
            st.dataframe(opa_result["weights_df"], use_container_width=True, hide_index=True)

            st.subheader("Ranked criteria and weights")
            st.dataframe(opa_result["ranked_df"], use_container_width=True, hide_index=True)

            psi = opa_result["psi"]
            st.markdown(
                f"""
                **Psi (l, m, u)** = ({psi[0]:.6f}, {psi[1]:.6f}, {psi[2]:.6f})  
                **Defuzzified Psi** = {defuzz_tfn(psi):.6f}
                """
            )

            fig_subj = px.bar(
                opa_result["weights_df"].sort_values("Defuzzified", ascending=False),
                x="Criterion",
                y="Defuzzified",
                text="Defuzzified",
                title="Fuzzy OPA subjective crisp weights",
            )
            fig_subj.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_subj.update_layout(height=420)
            st.plotly_chart(fig_subj, use_container_width=True)

        # ----------------------------------------------------
        # TAB 4: Final weights
        # ----------------------------------------------------
        with tab4:
            final_df = pd.DataFrame(
                {
                    "Criterion": crit_names,
                    "Objective_weight": stage3_obj["wagg"],
                    "Subjective_weight": subjective_crisp,
                    "Final_weight": final_weights.values,
                }
            )
            final_df["Rank"] = final_df["Final_weight"].rank(ascending=False, method="min").astype(int)
            final_df = final_df.sort_values("Final_weight", ascending=False)

            st.subheader("Integrated final criterion weights")
            st.dataframe(final_df, use_container_width=True)

            c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
            with c1:
                st.metric("Final EWCS p*(objective)", float(final_stage["p_star"][0]))
            with c2:
                st.metric("Final EWCS p*(subjective)", float(final_stage["p_star"][1]))
            with c3:
                st.write("Final p_raw =", np.array2string(final_stage["p_raw"], precision=6))

            fig_final = px.bar(
                final_df,
                x="Criterion",
                y="Final_weight",
                text="Final_weight",
                title="Composite final criterion weights",
            )
            fig_final.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_final.update_layout(height=420)
            st.plotly_chart(fig_final, use_container_width=True)

        # ----------------------------------------------------
        # TAB 5: Fuzzy Bonferroni CoCoSo
        # ----------------------------------------------------
        with tab5:
            st.subheader("Normalized fuzzy decision matrix")
            st.dataframe(norm_df, use_container_width=True)

            st.subheader("Weighted Bonferroni sequences")
            x1, x2 = st.columns(2)
            with x1:
                st.markdown("**SCoB**")
                st.dataframe(scob_df, use_container_width=True)
            with x2:
                st.markdown("**PCoB**")
                st.dataframe(pcob_df, use_container_width=True)

            st.subheader("Relative significance from defuzzified SCoB and PCoB")
            x3, x4, x5 = st.columns(3)
            with x3:
                st.markdown("**Kia / Ki1**")
                st.dataframe(psi_a_df, use_container_width=True)
            with x4:
                st.markdown("**Kib / Ki2**")
                st.dataframe(psi_b_df, use_container_width=True)
            with x5:
                st.markdown("**Kic / Ki3**")
                st.dataframe(psi_c_df, use_container_width=True)

            st.subheader("Ki reference values (Excel-style audit)")
            st.dataframe(cocoso_audit_df, use_container_width=True)

            st.subheader("Final ranking")
            st.dataframe(ranking_df, use_container_width=True)

            fig_rank = px.bar(
                ranking_df,
                x="Alternative",
                y="K",
                text="K",
                title="Final fuzzy Bonferroni CoCoSo ranking scores",
            )
            fig_rank.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_rank.update_layout(height=420)
            st.plotly_chart(fig_rank, use_container_width=True)

            st.markdown(
                f"""
                **Parameters**  
                ϕ1 = `{cocoso_meta["phi1"]:.4f}`  
                ϕ2 = `{cocoso_meta["phi2"]:.4f}`  
                π = `{cocoso_meta["pi"]:.4f}`
                """
            )

            export_sheets = {
                "DecisionMatrix": D_df,
                "AL_WL_IT_sigma": pd.DataFrame(
                    {
                        "Type": [str(t).strip().upper() for t in types],
                        "AL": AL,
                        "WL": WL,
                        "IT": IT,
                        "sigma": sigma,
                        "IT<sigma": it_ok,
                    },
                    index=crit_names,
                ),
                "WeightsSummary_Objective": summary.set_index("Criterion"),
                "StageI_A": pd.DataFrame(stage1["A"], index=alt_names + ["AL"], columns=crit_names),
                "StageI_beta": pd.DataFrame(stage1["beta"], columns=crit_names),
                "StageI_gamma": pd.DataFrame(stage1["gamma"], columns=crit_names),
                "StageI_delta": pd.DataFrame(stage1["delta"], columns=crit_names),
                "StageI_NIT": pd.DataFrame({"NIT": stage1["NIT"]}, index=crit_names),
                "StageI_nu": pd.DataFrame({"nu": stage1["nu"]}, index=crit_names),
                "StageII_X": pd.DataFrame(stage2["X"], index=alt_names, columns=crit_names),
                "StageII_R": pd.DataFrame(stage2["R"], index=crit_names, columns=crit_names),
                "StageIII_Objective_EWCS": pd.DataFrame(
                    {"p_raw": stage3_obj["p_raw"], "p_star": stage3_obj["p_star"]},
                    index=["w1", "w2"],
                ),
                "OPA_Theta": opa_result["theta_df"].set_index("Criterion"),
                "OPA_Coefficients": opa_result["coeff_df"].set_index("Criterion"),
                "OPA_Weights": opa_result["weights_df"].set_index("Criterion"),
                "OPA_Ranked": opa_result["ranked_df"].set_index("Rank"),
                "Final_Weights": final_df.set_index("Criterion"),
                "Final_EWCS": pd.DataFrame(
                    {"p_raw": final_stage["p_raw"], "p_star": final_stage["p_star"]},
                    index=["objective", "subjective"],
                ),
                "Fuzzy_Input": fuzzy_df,
                "Bonferroni_Normalized": norm_df,
                "SCoB": scob_df,
                "PCoB": pcob_df,
                "psi_a": psi_a_df,
                "psi_b": psi_b_df,
                "psi_c": psi_c_df,
                "CoCoSo_Audit": cocoso_audit_df,
                "Ranking": ranking_df,
            }

            st.download_button(
                "⬇️ Download results (Excel)",
                data=build_excel_bytes(export_sheets),
                file_name="m_ITARA_FuzzyOPA_FuzzyBonferroniCoCoSo_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

with st.expander("Implementation notes"):
    st.write(
        "- The objective block use m-ITARA.\n"
        "- The fuzzy OPA block uses the aggregation, coefficient construction, LP objective, and last-criterion constraint.\n"
        "- Expert inputs for fuzzy OPA are direct expert weights, not expert ranks.\n"
        "- Fuzzy Bonferroni CoCoSo now uses defuzzified SCoB and defuzzified PCoB for Kia/Kib/Kic and final ranking."
    )
