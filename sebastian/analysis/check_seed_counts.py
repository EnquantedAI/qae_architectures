"""Fail loudly unless every (series, sigma) cell has all 7 canonical seeds
for both the quantum and the classical track.

    python3 analysis/check_seed_counts.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = set(range(2023, 2030))
SIGMAS = [0.0] + [round(0.10 + 0.05 * i, 2) for i in range(9)]

qml = json.load(open(os.path.join(ROOT, "pennylane_monolith", "results_qml", "monolith_qml.json")))
cls = json.load(open(os.path.join(ROOT, "classical_baselines", "results_b", "classical_tuned_final.json")))
sel = json.load(open(os.path.join(ROOT, "classical_baselines", "results_b", "classical_tuned_selected.json")))
canon = {(i["dataset"], i["model"]): json.dumps(i["selected"]["config"], sort_keys=True) for i in sel}

have_q, have_c = {}, {}
for r in qml:
    if r["n_epochs"] == 300:
        have_q.setdefault((r.get("dataset", "mackey_glass"), r.get("tau"), round(r["sigma"], 2)), set()).add(r["seed"])
for r in cls:
    if r.get("stage") == "final" and json.dumps(r["config"], sort_keys=True) == canon.get((r["dataset"], r["model"])):
        have_c.setdefault((r["dataset"], r["model"], round(r["sigma"], 2)), set()).add(r["seed"])

bad = []
for ds, taus in (("mackey_glass", [17, 30]), ("energy", [None]), ("finance", [None])):
    for tau in taus:
        for s in SIGMAS:
            m = SEEDS - have_q.get((ds, tau, s), set())
            if m:
                bad.append(f"QAE {ds} tau={tau} sigma={s}: missing seeds {sorted(m)}")
for ds in ("mackey_glass_tau17", "mackey_glass_tau30", "energy", "finance"):
    for model in ("TinyMLP", "MLP-AE", "LSTM-AE"):
        for s in SIGMAS:
            m = SEEDS - have_c.get((ds, model, s), set())
            if m:
                bad.append(f"{model} {ds} sigma={s}: missing seeds {sorted(m)}")

if bad:
    print(f"{len(bad)} incomplete cell(s):")
    print("\n".join(bad))
    sys.exit(1)
print("OK: every cell has all 7 seeds (QAE and classical).")
