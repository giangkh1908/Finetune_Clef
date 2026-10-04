"""Stage `train`: grid lr x nhiều seed trên Clef(-flash) bằng clef-finetune (LoRA + joint schema head).
Mỗi tổ hợp là một MLflow run (loss theo step, lineage dữ liệu + schema), checkpoint (adapter + head) lưu mỗi epoch.
Run đã xong (có file DONE) được bỏ qua; run dở dang chạy tiếp từ checkpoint mới nhất (máy thuê bị ngắt).
"""

import itertools
import json

from src.common import OUTPUTS, REPORTS, ROOT, lineage_tags, load_split, params, read_jsonl, setup_mlflow, write_json
from src.finetune import build_cfg, free_gpu, run, steps_per_epoch


def run_name(lr, seed):
    return f"lr{lr:g}_s{seed}"


def main():
    P = params()
    t = P["train"]
    base = t["base_model"]  # quyết định từ bakeoff, ghi trong params.yaml
    train = load_split("train")

    mlflow = setup_mlflow("intent-train")
    runs = []
    for lr, seed in itertools.product(t["grid"]["lr"], t["seeds"]):
        name = run_name(lr, seed)
        out = OUTPUTS / "runs" / name
        if (out / "DONE").exists():
            print(f"bỏ qua {name} (đã xong)")
            runs.append(json.loads((out / "DONE").read_text()))
            continue
        cfg = build_cfg(P, train, out, lr=lr, seed=seed, epochs=t["epochs"])
        resume = any(out.glob("checkpoint-*"))
        with mlflow.start_run(run_name=name) as mrun:
            mlflow.set_tags({**lineage_tags(), "stage": "train", "base_model": base, "method": "lora+head"})
            mlflow.log_params({"lr": lr, "seed": seed, "epochs": t["epochs"], "base_model": base,
                               "base_revision": t["base_revision"], "max_steps": cfg["train"]["max_steps"],
                               "grad_accum": t["grad_accum"], "lora_r": t["lora"]["r"], "lora_alpha": t["lora"]["alpha"],
                               "head_lr": cfg["train"]["head_lr"], "label_smoothing": t["label_smoothing"],
                               "brier_weight": t["brier_weight"], "resumed": resume,
                               **{f"augment.{k}": v for k, v in t["augment"].items()}})
            res = run(cfg, resume=resume)
            for h in read_jsonl(out / "train_log.jsonl"):
                mlflow.log_metrics({k: h[k] for k in ("loss", "ce", "brier", "grad_norm", "lr") if k in h},
                                   step=h["step"])
            info = {"run": name, "lr": lr, "seed": seed, "mlflow_run_id": mrun.info.run_id, "base_model": base,
                    "steps_per_epoch": steps_per_epoch(len(train), t["grad_accum"]),
                    "lora_targets": res["lora_targets"], "dir": str(out.relative_to(ROOT))}
            (out / "DONE").write_text(json.dumps(info))
            runs.append(info)
        (out / "train.clef.jsonl").unlink(missing_ok=True)  # dựng lại được từ data/ + schema/
        free_gpu()

    write_json(REPORTS / "train_runs.json", runs)


if __name__ == "__main__":
    main()
