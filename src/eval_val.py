"""Stage `eval_val`: chấm MỌI checkpoint của MỌI run trên val (+ train_probe để đo train-acc).
Kết quả đi vào bảng val (reports/val_metrics.json) và log ngược vào MLflow run tương ứng theo epoch.
Dự đoán (kèm xác suất 60 intent) được lưu lại để stage `select` chọn ngưỡng.
"""

import json
from pathlib import Path

from src.common import OUTPUTS, REPORTS, ROOT, load_split, params, read_jsonl, setup_mlflow, write_json, write_jsonl
from src.engines import ClefEngine
from src.finetune import free_gpu
from src.metrics import score_rows, summarize


def epoch_train_loss(run_dir: Path, step: int, spe: int) -> float | None:
    log = {h["step"]: h for h in read_jsonl(run_dir / "train_log.jsonl")}  # resume có thể ghi lặp: lấy bản cuối
    xs = [h["loss"] for s, h in log.items() if step - spe < s <= step]
    return sum(xs) / len(xs) if xs else None


def main():
    P = params()
    t, g = P["train"], P["eval"]
    runs = json.loads((REPORTS / "train_runs.json").read_text(encoding="utf-8"))
    splits = {"val": load_split("val"), "train_probe": load_split("train_probe")}
    mlflow = setup_mlflow("intent-train")

    table = []
    for info in runs:
        run_dir = ROOT / info["dir"]
        spe = info["steps_per_epoch"]
        for ck in sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
            step = int(ck.name.split("-")[1])
            pred_dir = OUTPUTS / "preds" / info["run"] / ck.name
            engine = None
            metrics = {}
            for split, rows in splits.items():
                path = pred_dir / f"{split}.jsonl"
                if not path.exists():
                    engine = engine or ClefEngine(t["base_model"], t["base_revision"], ck, g["dtype"],
                                                  batch_size=g["batch_size"], max_length=g["max_length"])
                    write_jsonl(path, score_rows(engine.predict(rows)))
                metrics.update(summarize(read_jsonl(path), "val_" if split == "val" else "train_"))
            del engine
            free_gpu()

            row = {"run": info["run"], "lr": info["lr"], "seed": info["seed"], "checkpoint": ck.name,
                   "checkpoint_path": str(ck.relative_to(ROOT)), "mlflow_run_id": info["mlflow_run_id"],
                   "step": step, "epoch": round(step / spe, 2),
                   "train_loss": epoch_train_loss(run_dir, step, spe), **metrics}
            row["gap_train_val_acc"] = row["train_acc"] - row["val_acc"]
            table.append(row)
            print(f"{info['run']} {ck.name}: train-acc={row['train_acc']:.3f} val-acc={row['val_acc']:.3f} "
                  f"val-nll={row['val_nll']:.3f} val-ece={row['val_ece']:.3f}")

            with mlflow.start_run(run_id=info["mlflow_run_id"]):
                mlflow.log_metrics({f"valtable/{k}": v for k, v in row.items()
                                    if isinstance(v, float) and k not in ("lr", "epoch")}, step=step)

    write_json(REPORTS / "val_metrics.json", table)


if __name__ == "__main__":
    main()
