"""Flatten results/*.json into CSV tables for the report (results/tables/). Safe to rerun."""

import json
from pathlib import Path

import pandas as pd

from hw1.data import LABELS

HW1 = Path(__file__).resolve().parents[1]
RES = HW1 / "results"
OUT = RES / "tables"
METRICS = ["top1", "top3", "S"]


def _load(name: str) -> dict | None:
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def _cm_rows(dataset: str, method: str, cm: list[list[int]]) -> list[dict]:
    labels = LABELS[dataset]
    return [
        {"dataset": dataset, "method": method, "true": labels[i], "predicted": labels[j], "count": int(v)}
        for i, row in enumerate(cm) for j, v in enumerate(row)
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, cms, written = [], [], []

    def save(df: pd.DataFrame, name: str) -> None:
        df.to_csv(OUT / name, index=False, float_format="%.4f")
        written.append(f"{name} ({len(df)} rows)")

    if hc := _load("handcrafted.json"):
        save(pd.DataFrame([{"dataset": k, "model": m, **v} for k, r in hc.items() for m, v in r["models"].items()]),
             "handcrafted_models.csv")
        save(pd.DataFrame([{"dataset": k, "group": g, "n_features": v["n"], **{m: v[m] for m in METRICS}}
                           for k, r in hc.items() for g, v in r["groups"].items()]), "handcrafted_groups.csv")
        save(pd.DataFrame([{"dataset": k, "class": LABELS[k][c], "feature": f, "train_mean": vals[c]}
                           for k, r in hc.items() for f, vals in r["trends_train"].items() for c in range(6)]),
             "handcrafted_trends_train.csv")
        save(pd.DataFrame([{"dataset": k, "rank": i + 1, **d} for k, r in hc.items() for i, d in enumerate(r["anova_top_train"])]),
             "handcrafted_anova_top_train.csv")
        save(pd.DataFrame([{"dataset": k, "model": r["best"]["model"], **r["best"]["errors"]} for k, r in hc.items()]),
             "handcrafted_error_structure.csv")
        for k, r in hc.items():
            b = r["best"]
            summary.append({"dataset": k, "method": f"hand-crafted 30 s, {b['model']}", **{m: b[m] for m in METRICS}})
            cms += _cm_rows(k, f"hand-crafted {b['model']}", b["cm"])

    if mi := _load("handcrafted_mi.json"):
        save(pd.DataFrame([{"dataset": k, "features": c, "S": s} for k, r in mi.items() for c, s in r["energy_plus"].items()]),
             "handcrafted_group_combinations.csv")
        save(pd.DataFrame([{"dataset": k, "top_k_by_mi": int(c), "S": s} for k, r in mi.items() for c, s in r["mi_top_k"].items()]),
             "handcrafted_mi_topk.csv")
        save(pd.DataFrame([{"dataset": k, "group": g, "mi_sum": v["sum"], "mi_max": v["max"],
                            "mean_abs_corr_with_energy": r["mean_abs_corr_with_energy"].get(g)}
                           for k, r in mi.items() for g, v in r["mi_by_group"].items()]), "handcrafted_mi_by_group.csv")

    if ch := _load("chunks.json"):
        save(pd.DataFrame([{"dataset": k, "chunk_s": int(c[1:]), **v} for k, r in ch.items() for c, v in r["E1_chunk_length"].items()]),
             "chunks_length.csv")
        save(pd.DataFrame([{"dataset": k, "input": n, **v} for k, r in ch.items() for n, v in r["E2_multiscale"].items()]),
             "chunks_multiscale.csv")
        save(pd.DataFrame([{"dataset": k, "group": g, "chunk_s": int(c[1:]), "S": s}
                           for k, r in ch.items() for g, row in r["E3_group_by_length"].items() for c, s in row.items()]),
             "chunks_group_by_length.csv")
        for k, r in ch.items():
            best = max(r["E1_chunk_length"], key=lambda c: r["E1_chunk_length"][c]["S"])
            summary.append({"dataset": k, "method": f"hand-crafted {best[1:]} s chunks, logreg", **r["E1_chunk_length"][best]})

    if me := _load("mert.json"):
        save(pd.DataFrame([{"dataset": k, "layer": int(l), **v} for k, r in me.items() for l, v in r["sweep_mean"].items()]),
             "mert_layer_sweep.csv")
        rows = []
        for k, r in me.items():
            rows.append({"dataset": k, "variant": f"best single layer ({r['best_layer']}) time-mean", **r["sweep_mean"][str(r["best_layer"])]})
            for name in ("best layer mean+std", "layer-average mean"):
                rows.append({"dataset": k, "variant": name, **r[name]})
            summary.append({"dataset": k, "method": f"MERT-v2 layer {r['best_layer']} (val-selected), logreg", **r["sweep_mean"][str(r["best_layer"])]})
            cms += _cm_rows(k, f"MERT-v2 layer {r['best_layer']} logreg", r["best_cm"])
        save(pd.DataFrame(rows), "mert_variants.csv")

    if fu := _load("fusion.json"):
        save(pd.DataFrame([{"dataset": k, "method": n, **{m: v[m] for m in METRICS}} for k, r in fu.items() for n, v in r["methods"].items()]),
             "fusion.csv")
        for k, r in fu.items():
            for n, v in r["methods"].items():
                summary.append({"dataset": k, "method": n, **{m: v[m] for m in METRICS}})
                if "cm" in v:
                    cms += _cm_rows(k, n, v["cm"])

    if st := _load("stems.json"):
        save(pd.DataFrame([{"dataset": k, "method": n, **{m: v[m] for m in METRICS}} for k, r in st.items() for n, v in r["methods"].items()]),
             "stems.csv")
        save(pd.DataFrame([{"dataset": k, "rank": i + 1, **d} for k, r in st.items() for i, d in enumerate(r["mixbalance_anova_train"])]),
             "mixbalance_anova_train.csv")
        save(pd.DataFrame([{"dataset": k, "class": LABELS[k][c], "feature": f, "train_mean": vals[c]}
                           for k, r in st.items() for f, vals in r["mixbalance_class_means_train"].items() for c in range(6)]),
             "mixbalance_class_means_train.csv")
        for k, r in st.items():
            for n, v in r["methods"].items():
                summary.append({"dataset": k, "method": f"stems: {n}", **{m: v[m] for m in METRICS}})

    if summary:
        save(pd.DataFrame(summary).drop_duplicates(["dataset", "method"]).sort_values(["dataset", "S"], ascending=[True, False]),
             "summary.csv")
    if cms:
        save(pd.DataFrame(cms), "confusion_matrices_long.csv")
    print("\n".join(written))


if __name__ == "__main__":
    main()
