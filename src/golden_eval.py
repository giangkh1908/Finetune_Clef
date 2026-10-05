"""Stage `golden` (cuối mỗi vòng lặp): chấm model vừa được chọn trên val bằng golden, ĐÚNG 1 LẦN mỗi milestone.

Vòng lặp:  (đổi data/params/model) -> train -> chọn trên val -> register -> GOLDEN -> quyết định + gợi ý vòng sau
- Chấm đúng artifact đã đăng ký (pyfunc: model + code + ngưỡng chọn trên val), không chỉnh gì theo golden.
- Golden dùng để QUYẾT ĐỊNH (promote `champion` hay không) và CHẨN ĐOÁN, không để chọn hyperparam.
- Tại ngưỡng đã chọn trên val: TP/FP/FN/TN, P/R/F1/FPR trên golden -> cổng chất lượng (params.yaml: gate), cộng 2
  kiểm tra chuyển giao (acc và F1 không tụt quá xa so với val). Qua cả 3 -> version "eligible".
- Champion = version eligible có golden F1 cao nhất trong MỌI version đã đăng ký, chỉ so các version chấm trên cùng
  tập golden (cùng md5). Bằng nhau thì lấy precision cao hơn, rồi version mới hơn.
- Version không qua cổng val (register không gắn `candidate`) thì không chấm golden: milestone vẫn còn golden.
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
from src.metrics import (check_gate, confusion_line, per_intent, score_rows, selective_metrics, summarize,
                         top_confusions)

GOLDEN_DIR = REPORTS / "golden"
HISTORY = GOLDEN_DIR / "history"
LEDGER = GOLDEN_DIR / "ledger.csv"
CM = ("tp", "fp", "fn", "tn")
RATES = ("precision", "recall", "f1", "fpr", "coverage")

GATE_HINTS = {
    "precision": "Quá nhiều câu sai được tự động: model cần đúng hơn, hoặc tự tin đúng chỗ hơn (calibration).",
    "recall": "Đẩy sang người quá nhiều câu model làm đúng: confidence thấp trên câu đúng (tăng epoch, giảm "
              "label_smoothing).",
    "f1": "F1 tại ngưỡng thấp: xem precision/recall bên trên để biết vế nào kéo xuống.",
    "fpr": "Ngưỡng không chặn được câu sai: confidence của câu sai quá cao (miscalibrated, xem ECE).",
}


def next_actions(result, sel, gcfg):
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
    if not result["pass_threshold"]:
        acts.append("F1@ngưỡng trên golden tụt xa so với val: ngưỡng chọn trên val không chuyển giao. Cải thiện "
                    "calibration (brier_weight / label_smoothing) hoặc mở rộng val, rồi chấm golden ở milestone mới.")
    for f in result["gate_golden"]["fails"]:
        acts.append(f"Cổng golden: {f}. {GATE_HINTS[f.split('=')[0]]}")
    if result.get("golden_acc_agree3") and result["golden_acc_agree3"] - result["golden_acc"] > 0.03:
        acts.append("Acc trên câu cả 3 người chấm đồng ý cao hơn hẳn: một phần lỗi do nhãn MASSIVE mơ hồ. Cân nhắc "
                    "prepare.min_agree_train hoặc gán nhãn lại.")
    weak = ", ".join(f"{w['intent']} (f1={w['f1']:.2f})" for w in result["weakest_intents"][:5])
    acts.append(f"Intent yếu nhất trên golden: {weak}. Sửa mô tả trong schema/massive_intent.yaml cho các cặp hay "
                "nhầm (xem reports/golden/<milestone>.errors.jsonl), hoặc gán nhãn thêm cho chúng.")
    return acts


def gate_md(title, m, g):
    return [f"- {title}: {confusion_line(m)}",
            f"  - Cổng chất lượng: **{'PASS' if g['pass'] else 'FAIL'}**"
            + (f" ({'; '.join(g['fails'])})" if g["fails"] else "")]


def pick_champion(client, name, golden_md5):
    """Mọi version eligible chấm trên cùng tập golden, xếp theo (golden_f1, golden_precision, version) giảm dần."""
    rows = []
    for mv in client.search_model_versions(f"name='{name}'"):
        t = client.get_model_version(name, mv.version).tags
        if t.get("golden_eligible") == "true" and t.get("golden_md5") == golden_md5:
            rows.append({"version": str(mv.version), "milestone": t.get("golden_milestone", "?"),
                         **{f"golden_{k}": float(t[f"golden_{k}"]) for k in ("f1", "precision", "recall", "fpr")}})
    rows.sort(key=lambda r: (r["golden_f1"], r["golden_precision"], int(r["version"])), reverse=True)
    return rows


def skip_val_gate(ms, note, name, version, sel, out_path):
    """Không qua cổng val: ghi nhận, không chấm golden (golden của milestone này vẫn còn nguyên)."""
    fails = sel["gate_val"]["fails"]
    write_json(out_path, {"milestone": ms, "model": f"{name}/v{version}", "model_version": version,
                          "golden_evaluated": False, "promoted_to_champion": False,
                          "not_promoted_reason": "không qua cổng val: " + "; ".join(fails)})
    md = [f"# {ms}: {name}/v{version}", f"_{note}_\n",
          *gate_md("Val tại ngưỡng", sel["threshold"], sel["gate_val"]),
          "- Quyết định: **KHÔNG chấm golden, KHÔNG promote**.",
          "\n## Vòng sau nên làm", *[f"{i}. {f}. {GATE_HINTS[f.split('=')[0]]}" for i, f in enumerate(fails, 1)],
          "\nSửa params/schema rồi `dvc repro`: golden của milestone này chưa bị dùng."]
    (REPORTS / "next_action.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", default=None, help="mặc định lấy params.yaml: milestone.id")
    args = ap.parse_args()

    P = params()
    ms = args.milestone or P["milestone"]["id"]
    note = P["milestone"].get("note", "")
    gcfg, gate = P["golden"], P["gate"]
    reg = json.loads((REPORTS / "registered.json").read_text(encoding="utf-8"))
    sel = json.loads((REPORTS / "selection.json").read_text(encoding="utf-8"))
    name, version = reg["name"], str(reg["version"])
    hist = HISTORY / f"{ms}.json"
    out_path = GOLDEN_DIR / f"{ms}.json"
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    if not reg.get("gate_val"):
        skip_val_gate(ms, note, name, version, sel, out_path)
        return

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
    v = sel["threshold"]  # metric tại cùng ngưỡng trên val

    result = {
        "milestone": ms, "note": note, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": f"{name}/v{version}", "model_version": version, "golden_evaluated": True, **m,
        "threshold": impl.threshold,
        **{f"golden_{k}": s[k] for k in CM + RATES}, **{f"val_{k}": v[k] for k in CM + RATES},
        "val_acc": sel["val_acc"], "acc_drop_vs_val": sel["val_acc"] - m["golden_acc"],
        "f1_drop_vs_val": v["f1"] - s["f1"], "golden_md5": golden_md5,
        "weakest_intents": per_intent(scored)[:8],
        "top_confusions": top_confusions(scored, 10),
    }
    result["gate_golden"] = check_gate(s, gate)
    result["pass_acc"] = result["acc_drop_vs_val"] <= gcfg["max_acc_drop_vs_val"]
    result["pass_threshold"] = result["f1_drop_vs_val"] <= gcfg["max_f1_drop_vs_val"]
    result["eligible"] = result["gate_golden"]["pass"] and result["pass_acc"] and result["pass_threshold"]

    # Ghi metric vào version trước, rồi chọn champion trên MỌI version (kể cả version vừa chấm).
    vtags = {**{f"golden_{k}": str(s[k]) for k in CM}, **{f"golden_{k}": f"{s[k]:.4f}" for k in RATES},
             "golden_acc": f"{m['golden_acc']:.4f}", "golden_milestone": ms, "golden_md5": golden_md5,
             "gate_golden": "pass" if result["gate_golden"]["pass"] else
             "fail: " + "; ".join(result["gate_golden"]["fails"]),
             "golden_eligible": "true" if result["eligible"] else "false"}
    for k, val in vtags.items():
        client.set_model_version_tag(name, version, k, val)

    try:
        prev_champ = str(client.get_model_version_by_alias(name, "champion").version)
    except Exception:  # noqa: BLE001 - chưa có champion
        prev_champ = None
    ranking = pick_champion(client, name, golden_md5)
    champ = ranking[0]["version"] if ranking else prev_champ
    result.update({"prev_champion": f"v{prev_champ}" if prev_champ else None,
                   "champion": f"v{champ}" if champ else None,
                   "promoted_to_champion": champ == version and champ != prev_champ,
                   "champion_ranking": ranking})
    if not result["eligible"]:
        reasons = (result["gate_golden"]["fails"] + ([] if result["pass_acc"] else ["acc tụt so với val"])
                   + ([] if result["pass_threshold"] else ["F1 tụt so với val"]))
        result["not_promoted_reason"] = "không qua cổng golden: " + "; ".join(reasons)
    elif champ != version:
        result["not_promoted_reason"] = f"F1 golden không vượt champion v{champ}"

    acts = next_actions(result, sel, gcfg)
    md = [f"# {ms}: {result['model']}", f"_{note}_\n",
          f"- golden-acc **{m['golden_acc']:.4f}** | val-acc {sel['val_acc']:.4f} | chênh {result['acc_drop_vs_val']:+.4f}",
          f"- macro-F1 {m['golden_macro_f1']:.4f} | scenario-acc {m['golden_scenario_acc']:.4f} | "
          f"acc (nhãn 3/3 đồng ý) {m.get('golden_acc_agree3', float('nan')):.4f} | ECE {m['golden_ece']:.4f}",
          f"- Ngưỡng {impl.threshold:.4f} (F1 max trên val): tự động {s['coverage']:.1%}, "
          f"chuyển người {1 - s['coverage']:.1%}",
          *gate_md("Val   ", v, sel["gate_val"]),
          *gate_md("Golden", s, result["gate_golden"]),
          f"- F1 golden − val: {-result['f1_drop_vs_val']:+.4f} (cho phép −{gcfg['max_f1_drop_vs_val']})",
          "- Nhầm nhiều nhất: " + "; ".join(f"{g}->{p} ({n})" for g, p, n in result["top_confusions"][:5]),
          f"- Quyết định: **{'PROMOTE champion' if result['promoted_to_champion'] else 'KHÔNG promote'}**"
          + (f" ({result['not_promoted_reason']})" if result.get("not_promoted_reason") else "")
          + f". Champion: {result['champion'] or '(chưa có)'} (trước: {result['prev_champion'] or '(chưa có)'})",
          "\n## Xếp hạng champion (version qua cổng golden, cùng tập golden)",
          "| version | milestone | F1 | precision | recall | FPR |", "|---|---|---|---|---|---|",
          *[f"| v{r['version']} | {r['milestone']} | {r['golden_f1']:.4f} | {r['golden_precision']:.4f} | "
            f"{r['golden_recall']:.4f} | {r['golden_fpr']:.4f} |" for r in ranking],
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

    cols = ["milestone", "time", "model", "golden_acc", "golden_macro_f1", "golden_ece", "threshold",
            *[f"golden_{k}" for k in CM + RATES], "val_f1", "eligible", "promoted_to_champion", "champion", "note"]
    if LEDGER.exists() and LEDGER.read_text(encoding="utf-8").splitlines()[0] != ",".join(cols):
        LEDGER.rename(LEDGER.with_suffix(".v1.csv"))  # đổi cột: giữ ledger cũ, mở ledger mới
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(result)

    with mlflow.start_run(run_name=f"golden-{ms}"):
        mlflow.set_tags({**lineage_tags(), "stage": "golden", "milestone": ms, "model_version": version,
                         "eligible": str(result["eligible"]), "champion": str(result["champion"])})
        mlflow.log_metrics({k: float(v) for k, v in result.items()
                            if isinstance(v, (int, float)) and k != "model_version"})
        mlflow.log_artifact(str(hist))
        mlflow.log_artifact(str(HISTORY / f"{ms}.next_action.md"))
    if champ and champ != prev_champ:
        client.set_registered_model_alias(name, "champion", champ)
    print(md_text)


if __name__ == "__main__":
    main()
