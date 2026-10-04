"""Stage `select`: đọc bảng val, chẩn đoán từng checkpoint, chọn cấu hình ỔN ĐỊNH nhất, rồi chọn ngưỡng trên val.

Chẩn đoán:
  - underfit : train-EX thấp -> model chưa học được (tăng lr/epoch, kiểm tra dữ liệu)
  - overfit  : train-EX cao nhưng val-EX kém xa, hoặc val-loss tăng + val-EX giảm so với epoch trước
  - ok
Độ ổn định: gom theo (lr, epoch) qua các seed, điểm = mean(val-EX) - k * std(val-EX).
Ngưỡng: confidence nhỏ nhất sao cho precision trên val >= target, tức là coverage lớn nhất.
"""

import csv
import json
import statistics
from collections import defaultdict

from src.common import OUTPUTS, REPORTS, lineage_tags, params, read_jsonl, setup_mlflow, write_json
from src.evaluate import choose_threshold, selective_metrics, threshold_curve


def diagnose(rows, s):
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    for run_rows in by_run.values():
        run_rows.sort(key=lambda r: r["epoch"])
        prev = None
        for r in run_rows:
            flags = []
            if r["train_ex"] < s["underfit_train_ex"]:
                flags.append("underfit")
            if r["gap_train_val_ex"] > s["overfit_gap"]:
                flags.append("overfit:gap")
            if prev and r["val_loss"] and prev["val_loss"] and r["val_loss"] > prev["val_loss"] \
                    and r["val_ex"] < prev["val_ex"]:
                flags.append("overfit:val_worse")
            r["diagnosis"] = ",".join(flags) or "ok"
            prev = r
    return rows


def stability(rows, k):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["lr"], r["epoch"])].append(r)
    out = []
    for (lr, epoch), g in groups.items():
        exs = [r["val_ex"] for r in g]
        std = statistics.pstdev(exs) if len(exs) > 1 else 0.0
        out.append({"lr": lr, "epoch": epoch, "n_seeds": len(g), "val_ex_mean": statistics.mean(exs),
                    "val_ex_std": std, "val_ex_min": min(exs), "score": statistics.mean(exs) - k * std,
                    "all_ok": all(r["diagnosis"] == "ok" for r in g), "rows": g})
    return sorted(out, key=lambda x: -x["score"])


def md_table(rows, cols):
    def fmt(v):
        if isinstance(v, float):
            return f"{v:g}" if 0 < abs(v) < 1e-3 else f"{v:.4f}"
        return str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows])


def main():
    s = params()["select"]
    rows = diagnose(json.loads((REPORTS / "val_metrics.json").read_text(encoding="utf-8")), s)
    groups = stability(rows, s["stability_k"])

    pool = [g for g in groups if g["all_ok"]] or groups  # ưu tiên nhóm không có dấu hiệu under/overfit
    best_group = pool[0]
    chosen = max(best_group["rows"], key=lambda r: r["val_ex"])

    scored = read_jsonl(OUTPUTS / "preds" / chosen["run"] / chosen["checkpoint"] / "val.jsonl")
    thr = choose_threshold(scored, s["target_precision"])
    curve = threshold_curve(scored)

    # ---- báo cáo
    run_cols = ["run", "lr", "seed", "epoch", "train_loss", "val_loss", "train_ex", "val_ex", "gap_train_val_ex",
                "val_ex_easy", "val_ex_medium", "val_ex_hard", "val_ex_extra", "val_exec_error_rate",
                "val_mean_confidence", "diagnosis"]
    rows.sort(key=lambda r: (r["lr"], r["seed"], r["epoch"]))
    grp_cols = ["lr", "epoch", "n_seeds", "val_ex_mean", "val_ex_std", "val_ex_min", "score", "all_ok"]
    report = [
        "# Bảng val\n", "## Từng checkpoint\n", md_table(rows, run_cols),
        "\n\n## Độ ổn định theo cấu hình (gom các seed)\n", md_table(groups, grp_cols),
        f"\n\n## Lựa chọn\n- Cấu hình: lr={best_group['lr']:g}, epoch={best_group['epoch']} "
        f"(score={best_group['score']:.4f}, mean={best_group['val_ex_mean']:.4f} ± {best_group['val_ex_std']:.4f})",
        f"- Checkpoint: `{chosen['model_path']}` (val-EX={chosen['val_ex']:.4f}, chẩn đoán: {chosen['diagnosis']})",
        f"- Ngưỡng confidence: {thr['threshold']:.4f} -> precision={thr['precision']:.4f}, "
        f"coverage={thr['coverage']:.4f} trên val (target {s['target_precision']}, "
        f"{'đạt' if thr['target_reached'] else 'KHÔNG đạt'})",
    ]
    if not best_group["all_ok"]:
        report.append("- ⚠️ Không cấu hình nào sạch hoàn toàn; xem cột diagnosis trước khi dùng.")
    (REPORTS / "val_table.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    with open(REPORTS / "val_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=run_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(REPORTS / "threshold_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "coverage", "precision", "answered"])
        w.writeheader()
        w.writerows(curve)

    selection = {
        "run": chosen["run"], "checkpoint": chosen["checkpoint"], "model_path": chosen["model_path"],
        "mlflow_run_id": chosen["mlflow_run_id"], "lr": chosen["lr"], "seed": chosen["seed"],
        "epoch": chosen["epoch"], "diagnosis": chosen["diagnosis"],
        "val_ex": chosen["val_ex"], "val_metrics": {k: v for k, v in chosen.items() if k.startswith("val_")},
        "stability": {k: best_group[k] for k in grp_cols},
        "threshold": thr, "val_at_threshold": selective_metrics(scored, thr["threshold"]),
        "target_precision": s["target_precision"],
    }
    write_json(REPORTS / "selection.json", selection)

    mlflow = setup_mlflow("text2sql-train")
    with mlflow.start_run(run_name="selection"):
        mlflow.set_tags({**lineage_tags(), "stage": "select", "selected_run_id": chosen["mlflow_run_id"]})
        mlflow.log_params({"chosen_run": chosen["run"], "chosen_checkpoint": chosen["checkpoint"],
                           "threshold": thr["threshold"]})
        mlflow.log_metrics({"val_ex": chosen["val_ex"], "val_precision_at_thr": thr["precision"],
                            "val_coverage_at_thr": thr["coverage"], "stability_score": best_group["score"]})
        for f in ("val_table.md", "val_table.csv", "threshold_curve.csv", "selection.json"):
            mlflow.log_artifact(str(REPORTS / f))
    with mlflow.start_run(run_id=chosen["mlflow_run_id"]):
        mlflow.set_tag("selected", f"{chosen['checkpoint']}")

    print("\n".join(report[-4:]))


if __name__ == "__main__":
    main()
