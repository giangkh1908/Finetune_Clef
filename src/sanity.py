"""Stage `sanity`: cho model học thuộc 50 mẫu train rồi chấm lại chính 50 mẫu đó.

Model đủ sức phải đạt gần 100%. Nếu không đạt, gần như chắc chắn pipeline có bug (prompt/template, label
masking, token kết thúc, cách decode hoặc chấm) -> sửa trước khi tốn GPU cho train thật.
"""

import random
import sys
import tempfile

from src.common import REPORTS, lineage_tags, load_split, params, setup_mlflow, write_json, write_jsonl
from src.engine import Engine
from src.evaluate import score_rows, summarize
from src.sft import build_trainer, check_label_masking, load_base


def main():
    P = params()
    p, g = P["sanity"], P["generation"]
    base = P["train"]["base_model"]
    vi_train = [r for r in load_split("train") if r["lang"] == "vi"]
    rows = random.Random(0).sample(vi_train, p["n_examples"])

    mlflow = setup_mlflow("text2sql-sanity")
    with mlflow.start_run(run_name=f"overfit{p['n_examples']}-{base.split('/')[-1]}"):
        mlflow.set_tags({**lineage_tags(), "stage": "sanity", "base_model": base})
        mlflow.log_params({f"sanity.{k}": v for k, v in p.items()})

        model, tok = load_base(base)
        hp = {"epochs": p["epochs"], "lr": p["lr"], "batch_size": p["batch_size"], "grad_accum": 1,
              "max_length": P["train"]["max_length"], "warmup_ratio": 0.0, "weight_decay": 0.0,
              "scheduler": "constant", "seed": 0, "save": False, "gradient_checkpointing": False,
              "logging_steps": 5}
        trainer = build_trainer(model, tok, rows, None, tempfile.mkdtemp(), hp)

        problems = check_label_masking(trainer, tok)
        for pr in problems:
            print("[LABEL]", pr)

        trainer.train()
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        final_loss = losses[-1] if losses else float("nan")

        preds = Engine.from_model(trainer.model, tok, g["batch_size"], g["max_new_tokens"]).predict(rows)
        scored = score_rows(preds)
        m = summarize(scored)
        wrong = [r for r in scored if not r["correct"]]
        write_jsonl(REPORTS / "sanity_wrong.jsonl", wrong)

        result = {
            "n": len(rows), "ex": m["ex"], "final_train_loss": final_loss,
            "label_mask_problems": problems,
            "pass_ex": m["ex"] >= p["pass_ex"],
            "pass_loss": final_loss <= p["max_final_loss"],
            "pass_label_mask": not problems,
        }
        result["pass"] = result["pass_ex"] and result["pass_loss"] and result["pass_label_mask"]
        mlflow.log_metrics({"sanity_ex": m["ex"], "sanity_final_loss": final_loss,
                            "sanity_pass": float(result["pass"])})
        mlflow.log_artifact(str(REPORTS / "sanity_wrong.jsonl"))
        write_json(REPORTS / "sanity.json", result)

    print(result)
    if not result["pass"]:
        print("SANITY FAIL -> pipeline có bug. Xem reports/sanity_wrong.jsonl (pred vs sql, raw_output).")
        if not result["pass_loss"]:
            print("  loss không xuống: kiểm tra lr, label masking, dữ liệu đưa vào trainer.")
        elif not result["pass_ex"]:
            print("  loss thấp nhưng EX thấp: lỗi ở inference (template/enable_thinking, eos, clean_sql) hoặc bộ chấm.")
        sys.exit(1)


if __name__ == "__main__":
    main()
