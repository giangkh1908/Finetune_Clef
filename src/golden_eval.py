"""Stage `golden` (cuối mỗi vòng lặp): chấm model vừa được chọn trên val bằng golden, ĐÚNG 1 LẦN mỗi milestone.

Vòng lặp:  (đổi data/params/model) -> train -> chọn trên val -> register -> GOLDEN -> quyết định + gợi ý vòng sau
- Chấm đúng artifact đã đăng ký (pyfunc: model + code + ngưỡng chọn trên val), không chỉnh gì theo golden.
- Golden dùng để QUYẾT ĐỊNH (promote `champion` hay không) và CHẨN ĐOÁN, không để chọn hyperparam.
- Lịch sử nằm ở reports/golden/history/ + ledger.csv (append-only). Milestone đã chấm một model thì không chấm
  model khác dưới cùng milestone: muốn thử thay đổi mới phải mở milestone mới (params.yaml: milestone.id).
"""

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone

from src.common import (GOLDEN_FILE, REPORTS, file_md5, lineage_tags, params, read_jsonl, setup_mlflow,
                        write_json, write_jsonl)
from src.metrics import per_intent, score_rows, selective_metrics, summarize, top_confusions

GOLDEN_DIR = REPORTS / "golden"
HISTORY = GOLDEN_DIR / "history"
LEDGER = GOLDEN_DIR / "ledger.csv"


def next_actions(result, sel, gcfg, target):
    """Đọc chẩn đoán của val + golden -> việc nên làm ở vòng sau."""
    acts = []
    diag = sel["diagnosis"]
    if "underfit" in diag:
        acts.append("Underfit trên train: tăng epoch hoặc lr, tăng lora.r, hoặc thử Clef 27B (xem reports/bakeoff.md).")
    if "overfit" in diag:
        acts.append("Overfit (train-acc cao, val kém): giảm epoch hoặc lr, tăng augmentation (paraphrase/rename), "
                    "thêm dữ liệu gán nhãn thật.")
    if "miscalibrated" in diag:
        acts.append("ECE cao: tăng train.brier_weight / label_smoothing, hoặc calibrate nhiệt độ trên val.")
    if sel["stability"]["val_acc_std"] > 0.01:
        acts.append(f"Kết quả dao động giữa các seed (std={sel['stability']['val_acc_std']:.3f}): giảm lr, thêm seed "
                    "để chọn chắc hơn.")
    if result["acc_drop_vs_val"] > gcfg["max_acc_drop_vs_val"]:
        acts.append("Golden-acc thấp hơn val-acc nhiều: lựa chọn đã quá khớp val hoặc val không đại diện. KHÔNG chỉnh "
                    "theo golden; mở rộng val hoặc giảm số cấu hình thử trên val.")
    if target - result["golden_precision_at_thr"] > gcfg["max_precision_drop"]:
        acts.append("Metric ổn nhưng quyết định sai: ngưỡng chọn trên val không chuyển giao. Calibrate lại trên val "
                    "(tăng target_precision, hoặc cải thiện confidence), rồi chấm golden ở milestone mới.")
    if result.get("golden_acc_agree3") and result["golden_acc_agree3"] - result["golden_acc"] > 0.03:
        acts.append("Acc trên câu cả 3 người chấm đồng ý cao hơn hẳn: một phần lỗi do nhãn MASSIVE mơ hồ. Cân nhắc "
                    "prepare.min_agree_train hoặc gán nhãn lại.")
    weak = ", ".join(f"{w['intent']} (f1={w['f1']:.2f})" for w in result["weakest_intents"][:5])
    acts.append(f"Intent yếu nhất trên golden: {weak}. Sửa mô tả trong schema/massive_intent.yaml cho các cặp hay "
                "nhầm (xem reports/golden/<milestone>.errors.jsonl), hoặc gán nhãn thêm cho chúng.")
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", default=None, help="mặc định lấy params.yaml: milestone.id")
    args = ap.parse_args()

    P = params()
    ms = args.milestone or P["milestone"]["id"]
    note = P["milestone"].get("note", "")
    gcfg, target = P["golden"], P["select"]["target_precision"]
    reg = json.loads((REPORTS / "registered.json").read_text(encoding="utf-8"))
    sel = json.loads((REPORTS / "selection.json").read_text(encoding="utf-8"))
    name, version = reg["name"], str(reg["version"])
    hist = HISTORY / f"{ms}.json"
    out_path = GOLDEN_DIR / f"{ms}.json"

    if hist.exists():
        prev = json.loads(hist.read_text(encoding="utf-8"))
        if prev["model_version"] == version:  # cùng model, cùng milestone -> không chấm lại
            shutil.copy(hist, out_path)
            shutil.copy(HISTORY / f"{ms}.next_action.md", REPORTS / "next_action.md")
            print(f"{ms} đã chấm {name} v{version}, dùng lại kết quả (golden không chấm 2 lần).")
            return
        sys.exit(f"Milestone {ms} đã dùng golden cho v{prev['model_version']}. Model mới (v{version}) phải thuộc "
                 f"milestone mới: đổi milestone.id trong params.yaml (vd. dvc exp run -S milestone.id=M2).")

    mlflow = setup_mlflow("intent-golden")
    from mlflow import MlflowClient

    client = MlflowClient()
    mv = client.get_model_version(name, version)
    golden_md5 = file_md5(GOLDEN_FILE)
    if mv.tags.get("data.golden.md5") and mv.tags["data.golden.md5"] != golden_md5:
        sys.exit("golden.jsonl khác lúc đăng ký model (md5 lệch) -> dữ liệu đã đổi, kiểm tra dvc.lock.")

    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    impl = model.unwrap_python_model()
    golden = read_jsonl(GOLDEN_FILE)
    scored = score_rows(impl.predict_rows(golden))
    m = summarize(scored, "golden_")
    s = selective_metrics(scored, impl.threshold)

    val_acc = float(mv.tags.get("val_acc", sel["val_acc"]))
    result = {
        "milestone": ms, "note": note, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": f"{name}/v{version}", "model_version": version, **m,
        "threshold": impl.threshold, "golden_precision_at_thr": s["precision"], "golden_coverage_at_thr": s["coverage"],
        "val_acc": val_acc, "acc_drop_vs_val": val_acc - m["golden_acc"],
        "val_precision_at_thr": sel["threshold"]["precision"], "golden_md5": golden_md5,
        "weakest_intents": per_intent(scored)[:8],
        "top_confusions": top_confusions(scored, 10),
    }
    result["pass_acc"] = result["acc_drop_vs_val"] <= gcfg["max_acc_drop_vs_val"]
    result["pass_threshold"] = target - s["precision"] <= gcfg["max_precision_drop"]
    result["promoted_to_champion"] = result["pass_acc"] and result["pass_threshold"]

    # Champion hiện tại (nếu có) phải bị vượt trên golden thì mới thay.
    try:
        champ = client.get_model_version_by_alias(name, "champion")
        champ_acc = float(champ.tags.get("golden_acc", "nan"))
        result["prev_champion"] = f"v{champ.version} (golden_acc={champ_acc:.4f})"
        if champ.version != version and not m["golden_acc"] > champ_acc:
            result["promoted_to_champion"] = False
            result["not_promoted_reason"] = "không vượt champion hiện tại"
    except Exception:  # noqa: BLE001 - chưa có champion
        result["prev_champion"] = None

    acts = next_actions(result, sel, gcfg, target)
    md = [f"# {ms}: {result['model']}", f"_{note}_\n",
          f"- golden-acc **{m['golden_acc']:.4f}** | val-acc {val_acc:.4f} | chênh {result['acc_drop_vs_val']:+.4f}",
          f"- macro-F1 {m['golden_macro_f1']:.4f} | scenario-acc {m['golden_scenario_acc']:.4f} | "
          f"acc (nhãn 3/3 đồng ý) {m.get('golden_acc_agree3', float('nan')):.4f} | ECE {m['golden_ece']:.4f}",
          f"- Ngưỡng {impl.threshold:.4f}: precision golden {s['precision']:.4f} (val {sel['threshold']['precision']:.4f}, "
          f"target {target}), tự động xử lý {s['coverage']:.1%}, chuyển người {1 - s['coverage']:.1%}",
          "- Nhầm nhiều nhất: " + "; ".join(f"{g}->{p} ({n})" for g, p, n in result["top_confusions"][:5]),
          f"- Quyết định: **{'PROMOTE champion' if result['promoted_to_champion'] else 'KHÔNG promote'}**"
          + (f" ({result.get('not_promoted_reason')})" if result.get("not_promoted_reason") else ""),
          "\n## Vòng sau nên làm", *[f"{i}. {a}" for i, a in enumerate(acts, 1)],
          f"\nMở vòng mới: sửa params.yaml (milestone.id = M{int(ms[1:]) + 1 if ms[1:].isdigit() else '?'} + thay đổi) "
          "rồi `dvc repro`, hoặc `dvc exp run -S milestone.id=... -S ...`."]
    md_text = "\n".join(md) + "\n"

    HISTORY.mkdir(parents=True, exist_ok=True)
    write_json(hist, result)
    (HISTORY / f"{ms}.next_action.md").write_text(md_text, encoding="utf-8")
    shutil.copy(hist, out_path)
    (REPORTS / "next_action.md").write_text(md_text, encoding="utf-8")
    write_jsonl(GOLDEN_DIR / f"{ms}.errors.jsonl", [{k: v for k, v in r.items() if k != "probs"}
                                                 for r in scored if not r["correct"]])

    cols = ["milestone", "time", "model", "golden_acc", "golden_macro_f1", "golden_scenario_acc", "golden_ece",
            "val_acc", "acc_drop_vs_val", "threshold", "golden_precision_at_thr", "golden_coverage_at_thr",
            "promoted_to_champion", "note"]
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(result)

    with mlflow.start_run(run_name=f"golden-{ms}"):
        mlflow.set_tags({**lineage_tags(), "stage": "golden", "milestone": ms, "model_version": version})
        mlflow.log_metrics({k: float(v) for k, v in result.items() if isinstance(v, (int, float)) and k != "model_version"})
        mlflow.log_artifact(str(hist))
        mlflow.log_artifact(str(HISTORY / f"{ms}.next_action.md"))
    for k in ("golden_acc", "golden_precision_at_thr", "golden_coverage_at_thr"):
        client.set_model_version_tag(name, version, k, f"{result[k]:.4f}")
    client.set_model_version_tag(name, version, "golden_milestone", ms)
    if result["promoted_to_champion"]:
        client.set_registered_model_alias(name, "champion", version)
    print(md_text)


if __name__ == "__main__":
    main()
