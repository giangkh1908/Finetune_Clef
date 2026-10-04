"""Hàm SFT dùng chung cho sanity / bakeoff / train. Loss chỉ tính trên phần SQL (completion-only)."""

import os

from src.prompting import render_completion, render_prompt


def to_dataset(rows, tokenizer, max_length):
    from datasets import Dataset

    data, skipped = [], 0
    for r in rows:
        prompt = render_prompt(tokenizer, r["schema"], r["question"])
        # TRL tự thêm eos nếu completion chưa kết thúc bằng eos -> bỏ "\n" sau <|im_end|>.
        completion = render_completion(tokenizer, r["schema"], r["question"], r["sql"]).rstrip("\n")
        n = len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"])
        if n > max_length:  # cắt bớt sẽ mất câu SQL -> bỏ hẳn
            skipped += 1
            continue
        data.append({"prompt": prompt, "completion": completion})
    return Dataset.from_list(data), skipped


def load_base(model_id, method="full", lora=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if method == "lora":
        from peft import LoraConfig, get_peft_model

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        model = get_peft_model(model, LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules="all-linear", task_type="CAUSAL_LM"))
    else:
        # Trọng số fp32 + autocast bf16: ổn định hơn train thuần bf16.
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    return model, tok


def build_trainer(model, tok, train_rows, val_rows, out_dir, hp: dict, report_to="mlflow"):
    """hp: epochs, lr, batch_size, grad_accum, max_length, warmup_ratio, weight_decay, seed,
    save (bool), gradient_checkpointing (bool)."""
    from trl import SFTConfig, SFTTrainer

    train_ds, skip_tr = to_dataset(train_rows, tok, hp["max_length"])
    eval_ds, skip_val = (to_dataset(val_rows, tok, hp["max_length"]) if val_rows else (None, 0))
    print(f"train {len(train_ds)} mẫu (bỏ {skip_tr} quá dài) | val {len(eval_ds) if eval_ds else 0} (bỏ {skip_val})")

    save = hp.get("save", True)
    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=hp["epochs"],
        learning_rate=hp["lr"],
        lr_scheduler_type=hp.get("scheduler", "cosine"),
        warmup_ratio=hp.get("warmup_ratio", 0.03),
        weight_decay=hp.get("weight_decay", 0.0),
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp.get("grad_accum", 1),
        gradient_checkpointing=hp.get("gradient_checkpointing", True),
        bf16=True,
        max_length=hp["max_length"],
        completion_only_loss=True,
        packing=False,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch" if save else "no",
        save_only_model=True,
        logging_steps=hp.get("logging_steps", 10),
        seed=hp["seed"],
        data_seed=hp["seed"],
        report_to=report_to,
        dataloader_num_workers=2,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tok)
    trainer.n_skipped = {"train": skip_tr, "val": skip_val}
    return trainer


def check_label_masking(trainer, tok, n=20) -> list[str]:
    """Kiểm tra bug phổ biến nhất: token được tính loss phải đúng là phần SQL + token kết thúc."""
    ds = trainer.train_dataset
    problems = []
    if "completion_mask" not in ds.column_names:
        return [f"không thấy completion_mask trong dataset đã tokenize (cột: {ds.column_names})"]
    end_tok = "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token
    for i in range(min(n, len(ds))):
        ids, mask = ds[i]["input_ids"], ds[i]["completion_mask"]
        target = tok.decode([t for t, m in zip(ids, mask) if m], skip_special_tokens=False)
        if not target.strip().endswith(end_tok):
            problems.append(f"mẫu {i}: phần tính loss không kết thúc bằng {end_tok}: {target[-80:]!r}")
        if "### Câu hỏi" in target or "CREATE TABLE" in target:
            problems.append(f"mẫu {i}: phần tính loss lẫn cả prompt: {target[:80]!r}")
        if sum(mask) == 0:
            problems.append(f"mẫu {i}: không có token nào được tính loss")
    return problems


def save_for_inference(trainer, tok, out_dir):
    """Lưu model (merge LoRA nếu có) + tokenizer để Engine load trực tiếp."""
    model = trainer.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
