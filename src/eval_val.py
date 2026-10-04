"""Stage `eval_val`: chấm MỌI checkpoint của MỌI run trên val (+ train_probe để đo train-EX).
Kết quả đi vào bảng val (reports/val_metrics.json) và log ngược vào MLflow run tương ứng theo epoch.
Dự đoán đã chấm được lưu lại kèm confidence để stage `select` chọn ngưỡng.
"""

import json
from pathlib import Path

from src.common import OUTPUTS, REPORTS, ROOT, load_split, params, read_jsonl, setup_mlflow, write_json, write_jsonl
from src.engine import Engine
from src.evaluate import score_rows, summarize


def epoch_losses(ck: Path) -> dict:
    state = json.loads((ck / "trainer_state.json").read_text())
    epoch = round(state["epoch"])
    tr = [h["loss"] for h in state["log_history"] if "loss" in h and epoch - 1 < h["epoch"] <= epoch]
    ev = [h["eval_loss"] for h in state["log_history"] if "eval_loss" in h and round(h["epoch"]) == epoch]
    return {"epoch": epoch, "step": state["global_step"],
            "train_loss": sum(tr) / len(tr) if tr else None, "val_loss": ev[-1] if ev else None}


def main():
    P = params()
    g = P["generation"]
    runs = json.loads((REPORTS / "train_runs.json").read_text(encoding="utf-8"))
    splits = {"val": load_split("val"), "train_probe": load_split("train_probe")}
    mlflow = setup_mlflow("text2sql-train")

    table = []
    for info in runs:
        run_dir = ROOT / info["dir"]
        for ck in sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
            model_path = ck / "merged" if (ck / "merged").exists() else ck
            pred_dir = OUTPUTS / "preds" / info["run"] / ck.name
            engine = None
            metrics = {}
            for split, rows in splits.items():
                path = pred_dir / f"{split}.jsonl"
                if not path.exists():
                    engine = engine or Engine(str(model_path), g["backend"], g["batch_size"], g["max_new_tokens"])
                    write_jsonl(path, score_rows(engine.predict(rows)))
                prefix = "val_" if split == "val" else "train_"
                metrics.update(summarize(read_jsonl(path), prefix))
            del engine

            row = {"run": info["run"], "lr": info["lr"], "seed": info["seed"], "checkpoint": ck.name,
                   "model_path": str(model_path.relative_to(ROOT)), "mlflow_run_id": info["mlflow_run_id"],
                   **epoch_losses(ck), **metrics}
            row["gap_train_val_ex"] = row["train_ex"] - row["val_ex"]
            table.append(row)
            print(f"{info['run']} {ck.name}: train-EX={row['train_ex']:.3f} val-EX={row['val_ex']:.3f}")

            with mlflow.start_run(run_id=info["mlflow_run_id"]):
                mlflow.log_metrics({f"valtable/{k}": v for k, v in row.items()
                                    if isinstance(v, float) and k not in ("lr",)}, step=row["epoch"])

    write_json(REPORTS / "val_metrics.json", table)


if __name__ == "__main__":
    main()
