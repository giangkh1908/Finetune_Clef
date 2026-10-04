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
from src.evaluate import LEVELS, score_rows, selective_metrics, summarize

GOLDEN_DIR = REPORTS / "golden"
HISTORY = GOLDEN_DIR / "history"
LEDGER = GOLDEN_DIR / "ledger.csv"


def next_actions(result, sel, gcfg, target):
    """Đọc chẩn đoán của val + golden -> việc nên làm ở vòng sau."""
    acts = []
    diag = sel["diagnosis"]
    if "underfit" in diag:
        acts.append("Underfit trên train: tăng epoch hoặc lr, hoặc dùng model lớn hơn (xem reports/bakeoff.md).")
    if "overfit" in diag:
        acts.append("Overfit (train-EX cao, val kém): giảm epoch hoặc lr, tăng weight_decay, thêm dữ liệu train đa dạng "
                    "hơn (vd. gretelai/synthetic_text_to_sql).")
    if sel["stability"]["val_ex_std"] > 0.01:
        acts.append(f"Kết quả dao động giữa các seed (std={sel['stability']['val_ex_std']:.3f}): giảm lr, thêm seed để "
                    "chọn chắc hơn.")
    if result["ex_drop_vs_val"] > gcfg["max_ex_drop_vs_val"]:
        acts.append("Golden-EX thấp hơn val-EX nhiều: lựa chọn đã quá khớp val hoặc val không đại diện. KHÔNG chỉnh "
                    "theo golden; mở rộng val hoặc giảm số cấu hình thử trên val.")
    if target - result["golden_precision_at_thr"] > gcfg["max_precision_drop"]:
        acts.append("Metric ổn nhưng quyết định sai: ngưỡng chọn trên val không chuyển giao. Calibrate lại trên val "
                    "(tăng target_precision, hoặc cải thiện confidence), rồi chấm golden ở milestone mới.")
    weak = min(LEVELS, key=lambda k: result.get(f"golden_ex_{k}", 1.0))
    acts.append(f"Nhóm yếu nhất trên golden: {weak} (EX={result.get(f'golden_ex_{weak}', float('nan')):.3f}). "
                "Xem reports/golden/<milestone>.errors.jsonl để phân loại lỗi (sai bảng/cột, JOIN, giá trị, lồng).")
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

    mlflow = setup_mlflow("text2sql-golden")
    from mlflow import MlflowClient

    client = MlflowClient()
    mv = client.get_model_version(name, version)
    golden_md5 = file_md5(GOLDEN_FILE)
    if mv.tags.get("data.golden.md5") and mv.tags["data.golden.md5"] != golden_md5:
        sys.exit("golden.jsonl khác lúc đăng ký model (md5 lệch) -> dữ liệu đã đổi, kiểm tra dvc.lock.")

    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    impl = model.unwrap_python_model()
    golden = read_jsonl(GOLDEN_FILE)
    scored = score_rows(impl.engine.predict(golden))
    m = summarize(scored, "golden_")
    s = selective_metrics(scored, impl.threshold)

    val_ex = float(mv.tags.get("val_ex", sel["val_ex"]))
    result = {
        "milestone": ms, "note": note, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": f"{name}/v{version}", "model_version": version, **m,
        "threshold": impl.threshold, "golden_precision_at_thr": s["precision"], "golden_coverage_at_thr": s["coverage"],
        "val_ex": val_ex, "ex_drop_vs_val": val_ex - m["golden_ex"],
        "val_precision_at_thr": sel["threshold"]["precision"], "golden_md5": golden_md5,
    }
    result["pass_ex"] = result["ex_drop_vs_val"] <= gcfg["max_ex_drop_vs_val"]
    result["pass_threshold"] = target - s["precision"] <= gcfg["max_precision_drop"]
    result["promoted_to_champion"] = result["pass_ex"] and result["pass_threshold"]

    # Champion hiện tại (nếu có) phải bị vượt trên golden thì mới thay.
    try:
        champ = client.get_model_version_by_alias(name, "champion")
        champ_ex = float(champ.tags.get("golden_ex", "nan"))
        result["prev_champion"] = f"v{champ.version} (golden_ex={champ_ex:.4f})"
        if champ.version != version and not m["golden_ex"] > champ_ex:
            result["promoted_to_champion"] = False
            result["not_promoted_reason"] = "không vượt champion hiện tại"
    except Exception:  # noqa: BLE001 - chưa có champion
        result["prev_champion"] = None

    acts = next_actions(result, sel, gcfg, target)
    md = [f"# {ms}: {result['model']}", f"_{note}_\n",
          f"- golden-EX **{m['golden_ex']:.4f}** | val-EX {val_ex:.4f} | chênh {result['ex_drop_vs_val']:+.4f}",
          f"- Ngưỡng {impl.threshold:.4f}: precision golden {s['precision']:.4f} (val {sel['threshold']['precision']:.4f}, "
          f"target {target}), coverage {s['coverage']:.4f}",
          "- Theo độ khó: " + ", ".join(f"{k}={m.get(f'golden_ex_{k}', 0):.3f}" for k in LEVELS),
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
    write_jsonl(GOLDEN_DIR / f"{ms}.errors.jsonl", [r for r in scored if not r["correct"]])

    cols = ["milestone", "time", "model", "golden_ex", *[f"golden_ex_{k}" for k in LEVELS], "val_ex",
            "ex_drop_vs_val", "threshold", "golden_precision_at_thr", "golden_coverage_at_thr",
            "promoted_to_champion", "note"]
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(result)

    with mlflow.start_run(run_name=f"golden-{ms}"):
        mlflow.set_tags({**lineage_tags(), "stage": "golden", "milestone": ms, "model_version": version})
        mlflow.log_metrics({k: float(v) for k, v in result.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(hist))
        mlflow.log_artifact(str(HISTORY / f"{ms}.next_action.md"))
    for k in ("golden_ex", "golden_precision_at_thr", "golden_coverage_at_thr"):
        client.set_model_version_tag(name, version, k, f"{result[k]:.4f}")
    client.set_model_version_tag(name, version, "golden_milestone", ms)
    if result["promoted_to_champion"]:
        client.set_registered_model_alias(name, "champion", version)
    print(md_text)


if __name__ == "__main__":
    main()
