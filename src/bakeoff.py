"""Bakeoff (NGOÀI vòng lặp, chạy bằng `dvc repro bakeoff/dvc.yaml`): so các base model ứng viên với CÙNG ngân sách, chỉ dùng train + val.

Mỗi ứng viên: (1) zero-shot trên val, (2) fine-tune ngắn trên một tập con train cố định rồi chấm val,
(3) đo latency batch=1 và số tham số. Chọn theo val-EX sau fine-tune trong ràng buộc triển khai;
model nhỏ hơn được ưu tiên nếu kém model tốt nhất không quá `prefer_smaller_within`.
"""

import gc
import random
import shutil

from src.common import OUTPUTS, REPORTS, lineage_tags, load_split, params, setup_mlflow, write_json
from src.engine import Engine
from src.evaluate import score_rows, summarize
from src.sft import build_trainer, load_base, save_for_inference


def free_gpu():
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def run_candidate(c, P, train_sub, val, mlflow):
    b, g, t = P["bakeoff"], P["generation"], P["train"]
    with mlflow.start_run(run_name=f"bakeoff-{c['name']}"):
        mlflow.set_tags({**lineage_tags(), "stage": "bakeoff", "base_model": c["hf_id"]})
        method = c.get("method", t["method"])
        mlflow.log_params({"hf_id": c["hf_id"], "method": method, "train_subset": len(train_sub),
                           "epochs": b["epochs"], "lr": b["lr"]})

        # (1) zero-shot
        eng = Engine(c["hf_id"], g["backend"], g["batch_size"], g["max_new_tokens"])
        n_params = sum(p.numel() for p in eng.model.parameters()) / 1e9 if g["backend"] == "hf" else c.get("params_b")
        zs = summarize(score_rows(eng.predict(val)), "zeroshot_val_")
        latency = eng.latency_ms(val[: b["latency_probe"]])
        del eng
        free_gpu()

        # (2) fine-tune ngắn
        model, tok = load_base(c["hf_id"], method, t["lora"])
        hp = {"epochs": b["epochs"], "lr": b["lr"], "batch_size": t["batch_size"], "grad_accum": t["grad_accum"],
              "max_length": t["max_length"], "warmup_ratio": t["warmup_ratio"], "weight_decay": t["weight_decay"],
              "seed": 42, "save": False}
        out_dir = OUTPUTS / "bakeoff" / c["name"]
        trainer = build_trainer(model, tok, train_sub, None, str(out_dir / "trainer"), hp)
        trainer.train()
        save_for_inference(trainer, tok, str(out_dir / "model"))
        del trainer, model
        free_gpu()

        eng = Engine(str(out_dir / "model"), g["backend"], g["batch_size"], g["max_new_tokens"])
        ft = summarize(score_rows(eng.predict(val)), "ft_val_")
        del eng
        free_gpu()
        shutil.rmtree(out_dir, ignore_errors=True)  # chỉ giữ số liệu, không giữ trọng số bakeoff

        row = {"name": c["name"], "hf_id": c["hf_id"], "method": method, "params_b": round(n_params or 0, 3),
               "latency_ms_p50": round(latency, 1), **zs, **ft}
        mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float))})
        return row


def choose(rows, b):
    ok = [r for r in rows if r["params_b"] <= b["max_params_b"]]
    if not ok:
        raise SystemExit("Không ứng viên nào thoả max_params_b")
    best = max(r["ft_val_ex"] for r in ok)
    near = [r for r in ok if r["ft_val_ex"] >= best - b["prefer_smaller_within"]]
    return min(near, key=lambda r: (r["params_b"], -r["ft_val_ex"]))


def to_markdown(rows, chosen):
    cols = ["name", "params_b", "latency_ms_p50", "zeroshot_val_ex", "ft_val_ex",
            "ft_val_ex_easy", "ft_val_ex_medium", "ft_val_ex_hard", "ft_val_ex_extra", "ft_val_exec_error_rate"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: -r["ft_val_ex"]):
        cells = [f"{r.get(c, ''):.3f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols]
        if r["name"] == chosen["name"]:
            cells[0] = f"**{cells[0]}** ✅"
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    P = params()
    b = P["bakeoff"]
    train = load_split("train")
    train_sub = random.Random(0).sample(train, min(b["train_subset"], len(train)))
    val = load_split("val")

    mlflow = setup_mlflow("text2sql-bakeoff")
    rows = []
    for c in b["candidates"]:
        print(f"==== bakeoff: {c['name']} ({c['hf_id']})")
        rows.append(run_candidate(c, P, train_sub, val, mlflow))
        write_json(REPORTS / "bakeoff.json", rows)  # ghi dần để không mất kết quả nếu ứng viên sau lỗi

    chosen = choose(rows, b)
    md = to_markdown(rows, chosen)
    (REPORTS / "bakeoff.md").write_text(md + "\n", encoding="utf-8")
    write_json(REPORTS / "bakeoff_choice.json", {"name": chosen["name"], "hf_id": chosen["hf_id"],
                                                 "method": chosen["method"], "ft_val_ex": chosen["ft_val_ex"]})
    print(md)
    print(f"Đề xuất: {chosen['name']} -> nếu đồng ý, ghi `train.base_model: {chosen['hf_id']}` vào params.yaml")


if __name__ == "__main__":
    main()
