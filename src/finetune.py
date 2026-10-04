"""Cầu nối params.yaml -> clef-finetune (LoRA trên backbone + joint schema head, loss CE làm mượt + Brier).

clef-finetune đọc file JSONL dạng System One request, nên trước khi train ta ghi request đầy đủ (src/records.py)
vào thư mục run. Đơn vị của clef-finetune là optimizer step (batch 1 x grad_accum câu); ở đây quy đổi từ epoch.
"""

import math
from pathlib import Path

from src.common import write_jsonl
from src.records import PARAPHRASES_FILE, to_record


def steps_per_epoch(n_rows: int, grad_accum: int) -> int:
    return math.ceil(n_rows / grad_accum)


def build_cfg(P: dict, rows: list[dict], out_dir: Path, *, lr: float, seed: int, epochs: float,
              base_model: str | None = None, revision: str | None = None, augment: bool = True,
              grad_accum: int | None = None, save_every_epoch: bool = True, overrides: dict | None = None) -> dict:
    t = P["train"]
    grad_accum = grad_accum or t["grad_accum"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_file = out_dir / "train.clef.jsonl"
    write_jsonl(train_file, [to_record(r) for r in rows])
    spe = steps_per_epoch(len(rows), grad_accum)
    max_steps = max(1, round(spe * epochs))
    aug = dict(P["train"]["augment"]) if augment else {}
    if aug.get("paraphrase_prob", 0) > 0:
        aug["templates_file"] = str(PARAPHRASES_FILE)
    cfg = {
        "model": {"path": base_model or t["base_model"], "revision": revision or t["base_revision"],
                  "dtype": t["dtype"]},
        "data": {"train": str(train_file), "eval": None, "max_length": P["eval"]["max_length"]},
        "augment": aug,
        "lora": dict(t["lora"]),
        "train": {
            "seed": seed, "max_steps": max_steps, "grad_accum": grad_accum, "lr": lr,
            "head_lr": lr * t["head_lr_ratio"], "warmup_steps": max(1, round(max_steps * t["warmup_ratio"])),
            "weight_decay": t["weight_decay"], "label_smoothing": t["label_smoothing"],
            "brier_weight": t["brier_weight"], "ordinal_weight": 0.0, "train_head": t["train_head"], "head_dtype": "float32",
            "gradient_checkpointing": t["gradient_checkpointing"], "augment_train": augment,
            "save_every": spe if save_every_epoch else max_steps, "log_every": 10,
        },
        "output_dir": str(out_dir),
    }
    for k, v in (overrides or {}).items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) else v
    return cfg


def run(cfg: dict, resume: bool = False) -> dict:
    from clef_finetune.train import load_config, train

    return train(load_config(None, cfg), resume=resume)  # trộn với DEFAULTS của clef-finetune (vd. ordinal_weight)


def free_gpu():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
