"""
Merge the classical and quantum tracks into the paper's comparison tables.

Reads
-----
  ../pennylane_monolith/results_qml/monolith_qml.json          quantum (Monolith QAE)
  ../classical_baselines/results_b/classical_tuned_final.json   classical, tuned
  ../classical_baselines/results_b/classical_grid_b.json        classical, untuned

Writes (to ../results_tables/)
--------------------------
  comparison_summary.txt     human-readable, everything in one place
  table_matched.tex          parameter-matched table (TinyMLP 142p vs QAE 144p)
  table_generous.tex         generous-budget reference models
  comparison_rows.csv        the same numbers, flat, for plotting

Run verify_tracks_match.py first: these tables are only meaningful if both
tracks saw identical series.

    python3 aggregate_comparison.py
"""

import argparse
import csv
import json
import os
import statistics as st
import sys

# Standard-deviation convention. The QUANTICS 2026 extension's existing tables
# (tab:matched_baseline in main.tex, and cmes_comparison_summary.md) were all
# produced with the POPULATION standard deviation, ddof=0 - e.g. the published
# 68.98 +/- 1.92 is ddof=0; the same numbers with ddof=1 give +/- 2.14.
# Default to 0 so newly generated rows sit next to the existing ones without a
# silent convention change. Use --ddof 1 to switch everything (old rows
# included, since they are recomputed here from the raw JSON) to the sample SD.
DDOF = 0

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)  # sebastian/ -- this script lives in sebastian/analysis/
QML_ROWS = os.path.join(PARENT, "pennylane_monolith", "results_qml", "monolith_qml.json")
CLS_TUNED = os.path.join(PARENT, "classical_baselines", "results_b", "classical_tuned_final.json")
CLS_SELECTED = os.path.join(PARENT, "classical_baselines", "results_b", "classical_tuned_selected.json")
CLS_SEARCH = os.path.join(PARENT, "classical_baselines", "results_b", "classical_tuned_search.json")
CLS_UNTUNED = os.path.join(PARENT, "classical_baselines", "results_b", "classical_grid_b.json")
OUT_DIR = os.path.join(PARENT, "results_tables")

MATCHED_MODEL = "TinyMLP"          # the ~142-parameter, QAE-matched baseline
GENEROUS_MODELS = ["MLP-AE", "LSTM-AE"]

PRETTY = {
    "mackey_glass_tau17": r"Mackey--Glass ($\tau=17$)",
    "mackey_glass_tau30": r"Mackey--Glass ($\tau=30$)",
    "energy": "Energy (VIC demand)",
    "finance": "Finance (AAPL close)",
    "beer": "Beer production",
}
ORDER = ["mackey_glass_tau17", "mackey_glass_tau30", "energy", "finance", "beer"]


def _load(path):
    if not os.path.exists(path):
        print(f"note: {os.path.relpath(path, HERE)} not found - skipping", file=sys.stderr)
        return []
    return json.load(open(path))


def _quantum_dataset_label(row):
    """Quantum rows store dataset + tau separately; the classical track folds
    tau into the dataset name. Normalise to the classical convention. Rows
    written before the dataset field existed are Mackey-Glass by construction."""
    ds = row.get("dataset", "mackey_glass")
    if ds == "mackey_glass":
        return f"mackey_glass_tau{row['tau']}"
    return ds


def _agg(values):
    if len(values) > 1:
        sd = st.pstdev(values) if DDOF == 0 else st.stdev(values)
    else:
        sd = 0.0
    return {"n": len(values), "mean": sum(values) / len(values), "sd": sd}


def collect():
    """-> ({(dataset, sigma): {series_name: aggregate}}, [warnings])

    IMPORTANT - canonical-variant filtering. Both result files legitimately
    accumulate rows from more than one variant of the "same" model:

      * the quantum sweep's `--quick` smoke test writes n_epochs=20 rows next
        to the real n_epochs=300 ones;
      * `run_classical_tuned_b.py --quick` runs a trimmed grid, so its final
        rows carry a DIFFERENT config than the one the full search later
        selects.

    Pooling those together silently mixes a 20-epoch smoke run into a
    300-epoch mean (and a hidden_size=16 LSTM into a hidden_size=64 one). An
    earlier version of this script did exactly that and produced a
    -143.70 +/- 152.87 cell. So: for each (dataset, model) we keep only the
    CANONICAL variant and drop the rest, loudly.

    Canonical = the selected config recorded in classical_tuned_selected.json
    for classical rows; the largest n_epochs present for quantum rows."""
    cells, warnings = {}, []

    # ---- quantum: canonical = largest n_epochs per dataset ---------------
    qml_rows = _load(QML_ROWS)
    max_epochs = {}
    for r in qml_rows:
        ds = _quantum_dataset_label(r)
        max_epochs[ds] = max(max_epochs.get(ds, 0), r["n_epochs"])

    # ---- classical: canonical = the config the search actually selected --
    selected = {}
    for info in _load(CLS_SELECTED):
        selected[(info["dataset"], info["model"])] = \
            json.dumps(info["selected"]["config"], sort_keys=True)

    # Fallback: rebuild any missing selection straight from the search rows,
    # applying the same rule the search used (best mean training-partition
    # Delta). Needed because an early version of run_classical_tuned_b.py
    # OVERWROTE the selection file when re-run for a different --datasets,
    # dropping the earlier datasets' entries. The search log accumulates, so
    # the record is recoverable - and recomputing it here is exact, not a guess.
    search_by_key = {}
    for r in _load(CLS_SEARCH):
        if r.get("stage") != "search":
            continue
        search_by_key.setdefault((r["dataset"], r["model"]), {}) \
                     .setdefault(r["config_key"], []).append(r["select_delta_pct"])
    for key, by_cfg in search_by_key.items():
        if key in selected:
            continue
        best = max(by_cfg.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
        selected[key] = best[0]
        warnings.append(f"selection for {key[0]}/{key[1]} was missing from "
                        f"{os.path.basename(CLS_SELECTED)}; recomputed from the "
                        f"search log -> {best[0]}")

    def put(ds, sigma, name, value, extra=None):
        cells.setdefault((ds, sigma), {}).setdefault(name, {"values": [], "extra": extra or {}})
        cells[(ds, sigma)][name]["values"].append(value)

    dropped = {}

    def drop(label, reason):
        dropped[(label, reason)] = dropped.get((label, reason), 0) + 1

    for r in qml_rows:
        ds = _quantum_dataset_label(r)
        if r["n_epochs"] != max_epochs[ds]:
            drop(f"QAE/{ds}", f"n_epochs={r['n_epochs']} (canonical {max_epochs[ds]})")
            continue
        put(ds, r["sigma"], "QAE", r["delta_test_pct"],
            {"n_params": r["n_weights"], "n_epochs": r["n_epochs"]})

    for r in _load(CLS_TUNED):
        if r.get("stage") != "final":
            continue
        key = (r["dataset"], r["model"])
        cfg = json.dumps(r["config"], sort_keys=True)
        want = selected.get(key)
        if want is None:
            warnings.append(f"no selection record for {key[0]}/{key[1]} - "
                            f"cannot verify which config is canonical; "
                            f"copy classical_tuned_selected.json alongside the results")
        elif cfg != want:
            drop(f"{r['model']} (tuned)/{r['dataset']}", f"config {cfg} (canonical {want})")
            continue
        put(r["dataset"], r["sigma"], f"{r['model']} (tuned)",
            r["delta_test_pct"], {"n_params": r["n_params"], "config": r["config"]})

    for r in _load(CLS_UNTUNED):
        put(r["dataset"], r["sigma"], f"{r['model']} (untuned)",
            r["delta_test_pct"], {"n_params": r["n_params"]})

    for (label, reason), count in sorted(dropped.items()):
        warnings.append(f"dropped {count} non-canonical row(s) for {label}: {reason}")

    out = {}
    for key, series in cells.items():
        out[key] = {name: {**_agg(d["values"]), **d["extra"]} for name, d in series.items()}

    # A pooled cell whose rows disagree on parameter count is, by definition,
    # still mixing variants - catch anything the filters above missed.
    for (ds, sigma), series in out.items():
        for name, cell in series.items():
            params = cell.get("n_params")
            if isinstance(params, (list, set)):
                warnings.append(f"MIXED parameter counts in {ds}/{sigma}/{name}: {params}")
    return out, warnings


def _fmt(cell):
    if cell is None:
        return "--"
    return f"{cell['mean']:+.2f} \\pm {cell['sd']:.2f}"


def _fmt_txt(cell):
    if cell is None:
        return "        --     "
    return f"{cell['mean']:+7.2f} +/- {cell['sd']:5.2f} (n={cell['n']})"


def keys_in_order(cells):
    return sorted(cells, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1]))


def write_summary(cells, path):
    names = sorted({n for c in cells.values() for n in c})
    lines = ["Classical vs quantum denoising, Delta_test (%) - higher is better.",
             "All numbers: same series, same windowing (W=6, stride=2, chronological",
             "75/25), same fixed eval-noise realisation (seed 99123, clip=False),",
             "same metric Delta = 100*(1 - MSE(pred,true)/MSE(noisy,true)).",
             ""]
    for name in names:
        params = {c[name].get("n_params") for c in cells.values() if name in c}
        params.discard(None)
        lines.append(f"  {name}: {sorted(params)} trainable parameters")
    lines.append("")
    header = f"{'dataset':22s} {'sigma':>5s}  " + "  ".join(f"{n:^28s}" for n in names)
    lines += [header, "-" * len(header)]
    for ds, sigma in keys_in_order(cells):
        row = cells[(ds, sigma)]
        lines.append(f"{ds:22s} {sigma:5.1f}  " +
                     "  ".join(f"{_fmt_txt(row.get(n)):^28s}" for n in names))
    text = "\n".join(lines) + "\n"
    open(path, "w").write(text)
    return text


def write_matched_tex(cells, path):
    qae_name, cls_name = "QAE", f"{MATCHED_MODEL} (tuned)"
    body = []
    for ds, sigma in keys_in_order(cells):
        row = cells[(ds, sigma)]
        if qae_name not in row and cls_name not in row:
            continue
        body.append(f"\t\t\t{PRETTY.get(ds, ds)} & {sigma:.1f} & "
                    f"${_fmt(row.get(cls_name))}$ & ${_fmt(row.get(qae_name))}$ \\\\")
    tex = "\n".join([
        r"% Auto-generated by aggregate_comparison.py - do not edit by hand.",
        r"\begin{table}[H]",
        r"\caption{Parameter-matched benchmark across chaotic and real-world series.",
        r"Both models see the same series, windowing, fixed evaluation noise realisation",
        r"and MSE-based $\Delta_{test}$. Classical hyperparameters were selected on the",
        r"training partition only, once per series at $\sigma=0.2$, and applied unchanged",
        r"at every noise level. Values are mean $\pm$ standard deviation across seeds.}",
        r"\label{tab:matched_baseline_extended}",
        r"\centering", r"\small",
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"Series & $\sigma$ & TinyMLP (142 parameters) & Monolith QAE (144 parameters) \\",
        r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}", r"\vspace{2mm}", r"\end{table}", ""])
    open(path, "w").write(tex)
    return tex


def write_generous_tex(cells, path):
    cols = [f"{m} (tuned)" for m in GENEROUS_MODELS]
    body = []
    for ds, sigma in keys_in_order(cells):
        row = cells[(ds, sigma)]
        if not any(c in row for c in cols):
            continue
        body.append(f"\t\t\t{PRETTY.get(ds, ds)} & {sigma:.1f} & " +
                    " & ".join(f"${_fmt(row.get(c))}$" for c in cols) + r" \\")
    tex = "\n".join([
        r"% Auto-generated by aggregate_comparison.py - do not edit by hand.",
        r"\begin{table}[H]",
        r"\caption{Generous-budget classical reference models. These are NOT",
        r"parameter-matched to the QAE (thousands of parameters vs 144) and are",
        r"reported only to bound what a larger classical denoiser achieves under the",
        r"same protocol.}",
        r"\label{tab:generous_baseline}",
        r"\centering", r"\small",
        r"\begin{tabular}{l c " + " ".join("c" for _ in cols) + r"}",
        r"\toprule",
        r"Series & $\sigma$ & " + " & ".join(c.replace("(tuned)", "").strip() for c in cols) + r" \\",
        r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}", r"\vspace{2mm}", r"\end{table}", ""])
    open(path, "w").write(tex)
    return tex


def write_csv(cells, path):
    names = sorted({n for c in cells.values() for n in c})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "sigma", "model", "n_seeds", "delta_test_mean", "delta_test_sd",
                    "n_params"])
        for ds, sigma in keys_in_order(cells):
            for name in names:
                cell = cells[(ds, sigma)].get(name)
                if cell:
                    w.writerow([ds, sigma, name, cell["n"], round(cell["mean"], 4),
                                round(cell["sd"], 4), cell.get("n_params", "")])


def main():
    global DDOF
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddof", type=int, choices=[0, 1], default=DDOF,
                    help="0 = population SD (default; matches the tables already "
                         "in main.tex), 1 = sample SD")
    ap.add_argument("--include-beer", action="store_true",
                    help="keep the beer series (paper_279's dataset, dropped from "
                         "the extension by default)")
    args = ap.parse_args()
    DDOF = args.ddof

    cells, warnings = collect()
    if not args.include_beer:
        cells = {k: v for k, v in cells.items() if k[0] != "beer"}
    if not cells:
        sys.exit("No result files found - nothing to aggregate.")

    # Uneven seed counts within one model's row block usually mean a partial
    # or interrupted sweep - worth seeing before the numbers are trusted.
    per_model_ns = {}
    for (ds, sigma), series in cells.items():
        for name, cell in series.items():
            per_model_ns.setdefault(name, set()).add(cell["n"])
    for name, ns in sorted(per_model_ns.items()):
        if len(ns) > 1:
            warnings.append(f"uneven seed counts for {name}: n in {sorted(ns)} "
                            f"- a sweep may be incomplete")

    if warnings:
        print("!" * 72)
        print("CHECK THESE BEFORE USING THE NUMBERS:")
        for w in warnings:
            print(f"  ! {w}")
        print("!" * 72 + "\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(write_summary(cells, os.path.join(OUT_DIR, "comparison_summary.txt")))
    write_matched_tex(cells, os.path.join(OUT_DIR, "table_matched.tex"))
    write_generous_tex(cells, os.path.join(OUT_DIR, "table_generous.tex"))
    write_csv(cells, os.path.join(OUT_DIR, "comparison_rows.csv"))
    print(f"Wrote tables to {os.path.relpath(OUT_DIR, HERE)}/")


if __name__ == "__main__":
    main()
