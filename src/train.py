"""Stage `train`: grid hyperparam x nhiều seed trên base model mà bakeoff chọn.
Mỗi tổ hợp là một MLflow run (loss train/val theo step, lineage dữ liệu), checkpoint lưu mỗi epoch.
Run đã train xong (có file DONE) được bỏ qua, nên có thể chạy tiếp khi máy thuê bị ngắt.
"""

import itertools
import json

from src.common import OUTPUTS, REPORTS, lineage_tags, load_split, params, setup_mlflow, write_json
from src.sft import build_trainer, load_base


def run_name(lr, seed):
    return f"lr{lr:g}_s{seed}"


def main():
    P = params()
    t = P["train"]
    base, method = t["base_model"], t["method"]  # quyết định từ bakeoff, ghi trong params.yaml
    train, val = load_split("train"), load_split("val")

    mlflow = setup_mlflow("text2sql-train")
    runs = []
    for lr, seed in itertools.product(t["grid"]["lr"], t["seeds"]):
        name = run_name(lr, seed)
        out = OUTPUTS / "runs" / name
        if (out / "DONE").exists():
            print(f"bỏ qua {name} (đã xong)")
            runs.append(json.loads((out / "DONE").read_text()))
            continue
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({**lineage_tags(), "stage": "train", "base_model": base, "method": method})
            hp = {"epochs": t["epochs"], "lr": lr, "batch_size": t["batch_size"], "grad_accum": t["grad_accum"],
                  "max_length": t["max_length"], "warmup_ratio": t["warmup_ratio"],
                  "weight_decay": t["weight_decay"], "seed": seed, "save": True}
            mlflow.log_params({**hp, "base_model": base, "method": method})
            model, tok = load_base(base, method, t["lora"])
            trainer = build_trainer(model, tok, train, val, str(out), hp)
            mlflow.log_params({"skipped_long_train": trainer.n_skipped["train"],
                               "skipped_long_val": trainer.n_skipped["val"]})
            trainer.train()
            for ck in sorted(out.glob("checkpoint-*")):
                tok.save_pretrained(ck)
                if hasattr(trainer.model, "merge_and_unload"):  # LoRA: lưu bản merge để Engine load thẳng
                    from peft import AutoPeftModelForCausalLM
                    AutoPeftModelForCausalLM.from_pretrained(ck).merge_and_unload().save_pretrained(ck / "merged")
                    tok.save_pretrained(ck / "merged")
            info = {"run": name, "lr": lr, "seed": seed, "mlflow_run_id": run.info.run_id,
                    "base_model": base, "method": method, "dir": str(out.relative_to(OUTPUTS.parent))}
            (out / "DONE").write_text(json.dumps(info))
            runs.append(info)
            del trainer, model
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()

    write_json(REPORTS / "train_runs.json", runs)


if __name__ == "__main__":
    main()
