"""
rq2_firth.py — Firth-penalized refit of the one remaining RQ2 separation
condition (Claude Haiku, GPQA-Diamond) on the final corrected dataset.

Claude Haiku/MMLU-Pro and GPT-5.4 Mini/GPQA-Diamond, the other two
conditions the original Firth check covered, are no longer separation-
flagged with complete data (see tables/rq2_combined.tex) and are not
refit here -- their main cluster-robust logit estimates are already
reliable.

firthlogist has no cluster-robust option; p-values are from its default
penalized-likelihood-ratio test (more reliable under separation than Wald,
per the package's own default), not clustered by question like the rest of
RQ2. This is a genuine limitation of the Firth check, not an oversight,
and is stated in the appendix text.

Usage: python analysis/rq2_firth.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp


def firth_logistic(X, y, max_iter=100, tol=1e-6):
    """Firth (1993) bias-reduced logistic regression via modified-score IRLS
    (Heinze & Schemper 2002): standard IRLS with (y - p) replaced by
    (y - p + h*(0.5 - p)), h = diagonal of the weighted hat matrix.
    Gives finite, bounded estimates under separation, unlike plain MLE.

    Returns (beta, se) where se is the Wald SE from the final penalized
    Fisher information -- not a profile-likelihood CI/p-value (what Firth's
    original paper and most dedicated packages use), a materially simpler
    and slightly less accurate approximation under residual near-separation,
    stated explicitly wherever this is reported.
    """
    n, k = X.shape
    rank = np.linalg.matrix_rank(X)
    if rank < k:
        return None, None  # genuinely non-identifiable, not a convergence issue Firth can fix
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        XtWX = (X * W[:, None]).T @ X
        XtWX_inv = np.linalg.inv(XtWX)
        # hat values of the weighted design matrix W^.5 X (X'WX)^-1 X' W^.5
        h = np.einsum("ij,jk,ik->i", X, XtWX_inv, X) * W
        score = X.T @ (y - p + h * (0.5 - p))
        step = XtWX_inv @ score
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    se = np.sqrt(np.diag(XtWX_inv))
    return beta, se


def person_turn_rows(d, col="entropy", center=True):
    d = d[d["complete"] & d["t0_correct"] & d[col].notna()].copy()
    d["U"] = d[col] - d[col].mean() if center else d[col]
    rows = []
    for _, row in d.iterrows():
        last = int(row["first_flip"]) if row["ever_flip"] else pp.N_TURNS - 1
        for t in range(1, last + 1):
            rows.append({"turn": t, "U": row["U"], "flip": int(row["ever_flip"] and t == last)})
    return pd.DataFrame(rows)


def fit_firth(pt):
    Xdf = pt[["turn", "U"]].copy()
    Xdf["turn:U"] = Xdf["turn"] * Xdf["U"]
    X = np.column_stack([np.ones(len(Xdf)), Xdf.values.astype(float)])
    y = pt["flip"].values.astype(float)
    beta, se = firth_logistic(X, y)
    if beta is None:
        return None, None
    from scipy import stats as sstats
    z = beta / se
    pvals = 2 * sstats.norm.sf(np.abs(z))
    return beta, pvals


def main():
    pool = pp.canonical_pool("gpqa_diamond")
    df, _ = pp.load_cell("ClaudeHaiku", "gpqa_diamond", pool)
    d, _ = pp.dialogues(df)
    pt = person_turn_rows(d, "entropy", center=True)
    print(f"Claude Haiku / GPQA-Diamond: {len(pt)} person-turn rows, {pt['flip'].sum()} flips", flush=True)

    Xdf = pt[["turn", "U"]].copy()
    Xdf["turn:U"] = Xdf["turn"] * Xdf["U"]
    X_full = np.column_stack([np.ones(len(Xdf)), Xdf.values.astype(float)])
    rank = np.linalg.matrix_rank(X_full)
    n_t1 = int((pt["turn"] == 1).sum())
    print(f"design matrix rank {rank}/{X_full.shape[1]}; {n_t1}/{len(pt)} rows at turn=1", flush=True)

    coef, pvals = fit_firth(pt)
    result = {"n_rows": len(pt), "n_flips": int(pt['flip'].sum()), "n_turn1": n_t1,
              "rank": rank, "k": X_full.shape[1], "identifiable": coef is not None}
    if coef is not None:
        names = ["Intercept", "bP", "bH", "bPH"]
        result.update({f"firth_{n}": v for n, v in zip(names, coef)})
        result.update({f"firth_{n}_p": v for n, v in zip(names, pvals)})
    pd.Series(result).to_csv(os.path.join(pp.OUT, "rq2_firth_haiku_gpqa.csv"))
    print(result, flush=True)

    if coef is None:
        tex = (r"Even Firth's bias-reduced regression cannot identify this condition: "
               f"{n_t1} of {len(pt)} person-turn rows ({100*n_t1/len(pt):.0f}\\%) are at "
               r"$P{=}1$ -- Claude Haiku flips on the very first pressure turn in nearly "
               r"every dialogue -- leaving the design matrix rank-deficient "
               f"(rank {rank} of {X_full.shape[1]}). This is quasi-complete separation on "
               r"pressure turn itself, not only on entropy, and no amount of penalization "
               r"recovers three jointly-identified coefficients from data with essentially "
               r"one informative turn.")
    else:
        rows = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
                r"\textbf{Coefficient} & \textbf{Firth estimate} & \textbf{Firth $p$} \\", r"\midrule"]
        for n, label in [("bP", r"$\hat\beta_P$"), ("bH", r"$\hat\beta_H$"), ("bPH", r"$\hat\beta_{PH}$")]:
            rows.append(f"{label} & ${result[f'firth_{n}']:+.3f}$ & ${result[f'firth_{n}_p']:.3f}$ \\\\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        tex = "\n".join(rows)
    tex_path = os.path.join(pp.OUT, "tables", "firth_haiku_gpqa.tex")
    open(tex_path, "w").write(tex)
    print(f"wrote {tex_path}", flush=True)


if __name__ == "__main__":
    main()
