"""Stage `sanity` (GPU, vài phút): kiểm tra pipeline trước khi tốn GPU cho train thật.

1. Độ dài: schema 60 intent + câu dài nhất phải vừa eval.max_length (không bị cắt mất câu của người dùng).
2. Batch == từng câu: chấm theo batch (có padding) phải cho cùng đáp án và xác suất gần như cùng giá trị.
3. Overfit: LoRA học thuộc 50 câu train (không augmentation, không label smoothing) rồi chấm lại chính 50 câu đó.
   Model đủ sức phải đạt gần 100%. Không đạt -> pipeline có bug (record/nhãn, thứ tự option, LoRA không gắn vào
   đâu, head bị đóng băng, chấm sai) -> sửa trước khi train thật.
"""

import random
import sys

from src.common import OUTPUTS, REPORTS, lineage_tags, load_split, params, setup_mlflow, write_json, write_jsonl
from src.engines import ClefEngine
from src.finetune import build_cfg, free_gpu, run
from src.metrics import score_rows, summarize


def main():
    P = params()
    p, t, g = P["sanity"], P["train"], P["eval"]
    base, rev = t["base_model"], t["base_revision"]
    train = load_split("train")
    rows = random.Random(0).sample(train, p["n_examples"])
    longest = max(train, key=lambda r: len(r["text"]))

    mlflow = setup_mlflow("intent-sanity")
    with mlflow.start_run(run_name=f"sanity-{base.split('/')[-1]}"):
        mlflow.set_tags({**lineage_tags(), "stage": "sanity", "base_model": base})
        mlflow.log_params({f"sanity.{k}": v for k, v in p.items()})
        result = {"n": len(rows)}

        # (1) + (2) + zero-shot trên chính 50 câu (để thấy overfit có tác dụng)
        eng = ClefEngine(base, rev, None, g["dtype"], batch_size=g["batch_size"], max_length=g["max_length"])
        # encode_record lặng lẽ cắt state nếu thiếu chỗ -> tự kiểm: phần cố định (prompt + schema) + câu dài nhất.
        state_tokens = len(eng.tok(longest["text"], add_special_tokens=False).input_ids)
        fixed = len(eng.encode([{**longest, "text": ""}])[0].input_ids)
        result["schema_tokens"] = fixed
        result["max_input_tokens"] = fixed + state_tokens
        result["pass_length"] = fixed + state_tokens <= g["max_length"]

        probe = rows[: p["batch_check_n"]]
        one = eng.predict(probe, batch_size=1)
        many = eng.predict(probe, batch_size=p["batch_check_n"])
        diff = max(abs(a["probs"][k] - b["probs"][k]) for a, b in zip(one, many) for k in a["probs"])
        same = sum(a["pred"] == b["pred"] for a, b in zip(one, many))
        result["batch_max_prob_diff"] = diff
        result["pass_batch"] = diff <= p["max_batch_prob_diff"] and same == len(probe)
        result["zeroshot_acc_50"] = summarize(score_rows(eng.predict(rows)))["acc"]
        del eng
        free_gpu()

        # (3) overfit 50 câu
        out = OUTPUTS / "sanity"
        cfg = build_cfg(P, rows, out, lr=p["lr"], seed=0, epochs=p["epochs"], augment=False,
                        grad_accum=p["grad_accum"], save_every_epoch=False,
                        overrides={"train": {"label_smoothing": 0.0, "warmup_steps": 0}})
        res = run(cfg)
        losses = [h["loss"] for h in res["history"]]
        final_loss = sum(losses[-5:]) / len(losses[-5:])
        free_gpu()

        eng = ClefEngine(base, rev, res["checkpoint"], g["dtype"], batch_size=g["batch_size"],
                         max_length=g["max_length"])
        scored = score_rows(eng.predict(rows))
        m = summarize(scored)
        write_jsonl(REPORTS / "sanity_wrong.jsonl", [{k: r[k] for k in ("id", "text", "intent", "pred", "confidence")}
                                                    for r in scored if not r["correct"]])
        del eng
        free_gpu()

        result.update({
            "overfit_acc": m["acc"], "first_train_loss": losses[0], "final_train_loss": final_loss,
            "lora_targets": res["lora_targets"], "trainable_params": res["trainable_params"],
            "pass_acc": m["acc"] >= p["pass_acc"],
            "pass_loss": final_loss <= p["max_final_loss"] and final_loss < losses[0],
        })
        result["pass"] = all(result[k] for k in ("pass_length", "pass_batch", "pass_acc", "pass_loss"))
        mlflow.log_metrics({k: float(v) for k, v in result.items() if isinstance(v, (int, float, bool))})
        mlflow.log_artifact(str(REPORTS / "sanity_wrong.jsonl"))
        write_json(REPORTS / "sanity.json", result)

    print(result)
    if not result["pass"]:
        print("SANITY FAIL -> pipeline có bug, không train thật.")
        if not result["pass_length"]:
            print(f"  schema {result['schema_tokens']} token + câu dài nhất vượt eval.max_length: tăng max_length "
                  "hoặc rút gọn mô tả intent trong schema/massive_intent.yaml.")
        if not result["pass_batch"]:
            print("  batch khác từng câu: padding/attention_mask có vấn đề -> đặt eval.batch_size: 1.")
        if not result["pass_loss"]:
            print("  loss không xuống: kiểm tra lr, LoRA targets (lora_targets), head có được train (train_head).")
        elif not result["pass_acc"]:
            print("  loss thấp mà acc thấp: lỗi ở chấm (thứ tự option, nhãn) -> xem reports/sanity_wrong.jsonl.")
        sys.exit(1)


if __name__ == "__main__":
    main()
