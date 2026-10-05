"""Stage `select`: đọc bảng val, chẩn đoán từng checkpoint, chọn cấu hình ỔN ĐỊNH nhất, rồi chọn ngưỡng trên val.

Chẩn đoán:
  - underfit : train-acc thấp -> model chưa học được (tăng lr/epoch, kiểm tra dữ liệu)
  - overfit  : train-acc cao nhưng val-acc kém xa, hoặc val-NLL tăng + val-acc giảm so với epoch trước
  - miscalibrated : ECE trên val cao -> confidence không đáng tin, ngưỡng sẽ khó chuyển giao
  - ok
Độ ổn định: gom theo (lr, epoch) qua các seed, điểm = mean(val-acc) - k * std(val-acc).
Ngưỡng: quét mọi ngưỡng confidence trên val, lấy ngưỡng có F1 lớn nhất trong các ngưỡng qua cổng; ghi TP/FP/FN/TN, P/R/F1/FPR tại đó
rồi soát cổng chất lượng (params.yaml: gate). Không qua cổng thì register không gắn `candidate`.
"""

import csv
import json
import statistics
from collections import defaultdict

from src.common import OUTPUTS, REPORTS, lineage_tags, params, read_jsonl, setup_mlflow, write_json
from src.metrics import check_gate, choose_threshold, confusion_line, per_intent, threshold_curve, top_confusions


def diagnose(rows, s):
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    for run_rows in by_run.values():
        run_rows.sort(key=lambda r: r["step"])
        prev = None
        for r in run_rows:
            flags = []
            if r["train_acc"] < s["underfit_train_acc"]:
                flags.append("underfit")
            if r["gap_train_val_acc"] > s["overfit_gap"]:
                flags.append("overfit:gap")
            if prev and r["val_nll"] > prev["val_nll"] and r["val_acc"] < prev["val_acc"]:
                flags.append("overfit:val_worse")
            if r["val_ece"] > s["max_val_ece"]:
                flags.append("miscalibrated")
            r["diagnosis"] = ",".join(flags) or "ok"
            prev = r
    return rows


def stability(rows, k):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["lr"], r["epoch"])].append(r)
    out = []
    for (lr, epoch), g in groups.items():
        accs = [r["val_acc"] for r in g]
        std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        out.append({"lr": lr, "epoch": epoch, "n_seeds": len(g), "val_acc_mean": statistics.mean(accs),
                    "val_acc_std": std, "val_acc_min": min(accs), "score": statistics.mean(accs) - k * std,
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
    P = params()
    s, gate = P["select"], P["gate"]
    rows = diagnose(json.loads((REPORTS / "val_metrics.json").read_text(encoding="utf-8")), s)
    groups = stability(rows, s["stability_k"])

    pool = [g for g in groups if g["all_ok"]] or groups  # ưu tiên nhóm không có dấu hiệu under/overfit
    best_group = pool[0]
    chosen = max(best_group["rows"], key=lambda r: r["val_acc"])

    scored = read_jsonl(OUTPUTS / "preds" / chosen["run"] / chosen["checkpoint"] / "val.jsonl")
    thr = choose_threshold(scored, gate)
    gate_res = check_gate(thr, gate)
    curve = threshold_curve(scored)
    weak = per_intent(scored)[:8]
    conf = top_confusions(scored, 8)

    # ---- báo cáo
    run_cols = ["run", "lr", "seed", "epoch", "train_loss", "val_nll", "train_acc", "val_acc", "gap_train_val_acc",
                "val_macro_f1", "val_scenario_acc", "val_acc_agree3", "val_ece", "val_brier",
                "val_mean_confidence", "diagnosis"]
    rows.sort(key=lambda r: (r["lr"], r["seed"], r["step"]))
    grp_cols = ["lr", "epoch", "n_seeds", "val_acc_mean", "val_acc_std", "val_acc_min", "score", "all_ok"]
    report = [
        "# Bảng val\n", "## Từng checkpoint\n", md_table(rows, run_cols),
        "\n\n## Độ ổn định theo cấu hình (gom các seed)\n", md_table(groups, grp_cols),
        "\n\n## Intent yếu nhất (checkpoint được chọn)\n", md_table(weak, ["intent", "n", "recall", "precision", "f1"]),
        "\n\n## Nhầm lẫn nhiều nhất (vàng -> dự đoán)\n",
        "\n".join(f"- {g} -> {p}: {n}" for g, p, n in conf) or "- (không có)",
        f"\n\n## Lựa chọn\n- Cấu hình: lr={best_group['lr']:g}, epoch={best_group['epoch']} "
        f"(score={best_group['score']:.4f}, mean={best_group['val_acc_mean']:.4f} ± {best_group['val_acc_std']:.4f})",
        f"- Checkpoint: `{chosen['checkpoint_path']}` (val-acc={chosen['val_acc']:.4f}, "
        f"ECE={chosen['val_ece']:.4f}, chẩn đoán: {chosen['diagnosis']})",
        f"- Ngưỡng confidence: {thr['threshold']:.4f} = F1 max trên val "
        f"{'trong các ngưỡng qua cổng' if thr['within_gate'] else '(KHÔNG ngưỡng nào qua cổng, lấy F1 max toàn bộ)'}"
        "; dưới ngưỡng chuyển người.",
        f"- Tại ngưỡng: {confusion_line(thr)}",
        f"- Cổng chất lượng (val): **{'PASS' if gate_res['pass'] else 'FAIL'}**"
        + (f" ({'; '.join(gate_res['fails'])})" if gate_res["fails"] else ""),
    ]
    if not best_group["all_ok"]:
        report.append("- ⚠️ Không cấu hình nào sạch hoàn toàn; xem cột diagnosis trước khi dùng.")
    (REPORTS / "val_table.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    with open(REPORTS / "val_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=run_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(REPORTS / "threshold_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr",
                                           "coverage", "answered"])
        w.writeheader()
        w.writerows(curve)

    selection = {
        "run": chosen["run"], "checkpoint": chosen["checkpoint"], "checkpoint_path": chosen["checkpoint_path"],
        "mlflow_run_id": chosen["mlflow_run_id"], "lr": chosen["lr"], "seed": chosen["seed"],
        "epoch": chosen["epoch"], "diagnosis": chosen["diagnosis"],
        "val_acc": chosen["val_acc"], "val_metrics": {k: v for k, v in chosen.items() if k.startswith("val_")},
        "stability": {k: best_group[k] for k in grp_cols},
        "threshold": thr, "gate_val": gate_res,
    }
    write_json(REPORTS / "selection.json", selection)

    mlflow = setup_mlflow("intent-train")
    with mlflow.start_run(run_name="selection"):
        mlflow.set_tags({**lineage_tags(), "stage": "select", "selected_run_id": chosen["mlflow_run_id"]})
        mlflow.log_params({"chosen_run": chosen["run"], "chosen_checkpoint": chosen["checkpoint"],
                           "threshold": thr["threshold"]})
        mlflow.set_tag("gate_val", "pass" if gate_res["pass"] else "fail")
        mlflow.log_metrics({"val_acc": chosen["val_acc"], "val_ece": chosen["val_ece"],
                            "stability_score": best_group["score"],
                            **{f"val_{k}_at_thr": thr[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall",
                                                                    "f1", "fpr", "coverage")}})
        for f in ("val_table.md", "val_table.csv", "threshold_curve.csv", "selection.json"):
            mlflow.log_artifact(str(REPORTS / f))
    with mlflow.start_run(run_id=chosen["mlflow_run_id"]):
        mlflow.set_tag("selected", f"{chosen['checkpoint']}")

    print("\n".join(report[-5:]))


if __name__ == "__main__":
    main()
