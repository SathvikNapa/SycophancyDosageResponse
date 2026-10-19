"""
rq3_cross_benchmark.py — Cross-benchmark label transition table (NAACL plan
Experiment 3.2): for every model, shows its resister/capitulator/flat.n.s.
label on each of the three benchmarks side by side, and separately flags
benchmark-conditioned fragmentation (fragmentation is its own dimension,
not one of rq3()'s three labels -- a cell is a de facto "fragmenter" when
fragmentation is the dominant outcome, frag_pct_lost > 50%, per
methods.tex's definition).

Pure reanalysis of analysis/paper_out/results.csv, no new computation.

Usage: python analysis/rq3_cross_benchmark.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp


def main():
    df = pd.read_csv(os.path.join(pp.OUT, "results.csv"))
    df["frag_dominant"] = df["frag_pct_lost"] > 50

    wide = df.pivot(index="model", columns="dataset", values="label")
    frag = df.pivot(index="model", columns="dataset", values="frag_dominant")

    rows = []
    for model in wide.index:
        labs = {ds: wide.loc[model, ds] if ds in wide.columns else None for ds in pp.DATASETS}
        frags = {ds: bool(frag.loc[model, ds]) if ds in frag.columns and pd.notna(frag.loc[model, ds]) else False
                  for ds in pp.DATASETS}
        display = {ds: ("fragmenter" if frags.get(ds) else labs.get(ds)) for ds in pp.DATASETS}
        present = [v for v in display.values() if v is not None and not pd.isna(v)]
        changed = len(set(present)) > 1 if len(present) > 1 else False
        rows.append({"model": model, **{f"label_{ds}": display.get(ds) for ds in pp.DATASETS},
                     "changed_across_benchmarks": changed})

    out = pd.DataFrame(rows)
    out_path = os.path.join(pp.OUT, "rq3_cross_benchmark_transitions.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string())
    print(f"\nwrote {out_path}")

    n_changed = out["changed_across_benchmarks"].sum()
    print(f"\n{n_changed} of {len(out)} models change label across at least one benchmark pair")

    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"\textbf{Model} & \textbf{MMLU-Pro} & \textbf{GPQA-Diamond} & \textbf{HLE} \\", r"\midrule"]
    for _, r in out.iterrows():
        cells = [str(r.get(f"label_{ds}", "--")).replace("flat/n.s.", "flat") for ds in pp.DATASETS]
        lines.append(f"{pp.LABELS.get(r['model'], r['model'])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = os.path.join(pp.OUT, "tables", "rq3_cross_benchmark.tex")
    open(tex_path, "w").write("\n".join(lines))
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
