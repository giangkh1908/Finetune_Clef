"""Bakeoff (NGOÀI vòng lặp, chạy bằng `dvc repro bakeoff/dvc.yaml`): Clef-flash xử lý tiếng Việt tốt đến đâu,
so với các model quyết định khác (Laya), TRƯỚC khi đầu tư gán nhãn dữ liệu thật. Chỉ dùng train + val.

Mỗi ứng viên:
  (1) zero-shot trên val: acc, macro-F1, NLL, ECE, coverage tự động @ target precision
  (2) đo latency batch=1 và số tham số
  (3) chỉ ứng viên `kind: clef`: fine-tune LoRA ngắn với vài cỡ dữ liệu (`ft_sizes`) rồi chấm val
      -> đường "thêm N câu gán nhãn thì được bao nhiêu điểm" = cơ sở để quyết định có gán nhãn thật hay không.
Laya chỉ chấm zero-shot (pipeline này không fine-tune Laya). Model trong vòng lặp phải là Clef (clef-finetune).
"""

import random
import shutil

from src.common import OUTPUTS, REPORTS, lineage_tags, load_split, params, setup_mlflow, write_json
from src.engines import ClefEngine, build_engine
from src.finetune import build_cfg, free_gpu, run
from src.metrics import choose_threshold, score_rows, summarize


def evaluate(engine, val, target):
    scored = score_rows(engine.predict(val))
    m = summarize(scored, "val_")
    thr = choose_threshold(scored, target)
    m["val_auto_coverage"] = thr["coverage"] if thr["target_reached"] else 0.0
    if "options_collapsed" in scored[0]:
        m["val_options_collapsed_frac"] = sum(r["options_collapsed"] for r in scored) / len(scored)
    return m


def run_candidate(c, P, train, val, mlflow):
    b, g = P["bakeoff"], P["eval"]
    target = P["select"]["target_precision"]
    with mlflow.start_run(run_name=f"bakeoff-{c['name']}"):
        mlflow.set_tags({**lineage_tags(), "stage": "bakeoff", "kind": c["kind"], "hf_id": c["hf_id"]})
        mlflow.log_params({k: v for k, v in c.items() if k != "name"})

        eng = build_engine(c, P)
        row = {"name": c["name"], "kind": c["kind"], "hf_id": c["hf_id"], "params_b": round(eng.n_params_b(), 3),
               "latency_ms_p50": round(eng.latency_ms(val[: b["latency_probe"]]), 1)}
        row.update({f"zs_{k}": v for k, v in evaluate(eng, val, target).items()})
        del eng
        free_gpu()
        print(f"  zero-shot val-acc={row['zs_val_acc']:.4f}")

        if c["kind"] == "clef" and c.get("finetune", True):
            for n in b["ft_sizes"]:
                sub = random.Random(0).sample(train, min(n, len(train)))
                out = OUTPUTS / "bakeoff" / f"{c['name']}-n{n}"
                cfg = build_cfg(P, sub, out, lr=b["lr"], seed=0, epochs=b["epochs"], base_model=c["hf_id"],
                                revision=c.get("revision"), save_every_epoch=False)
                res = run(cfg)
                free_gpu()
                eng = ClefEngine(c["hf_id"], c.get("revision"), res["checkpoint"], g["dtype"],
                                 batch_size=g["batch_size"], max_length=g["max_length"])
                m = evaluate(eng, val, target)
                del eng
                free_gpu()
                shutil.rmtree(out, ignore_errors=True)  # chỉ giữ số liệu, không giữ trọng số bakeoff
                row.update({f"ft{n}_{k}": v for k, v in m.items()})
                print(f"  fine-tune {n} câu: val-acc={m['val_acc']:.4f}")
        mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float))})
        return row


def best_acc(r, b):
    """Điểm tốt nhất mà ứng viên đạt được trong ngân sách bakeoff (zero-shot hoặc fine-tune lớn nhất)."""
    keys = ["zs_val_acc"] + [f"ft{n}_val_acc" for n in b["ft_sizes"]]
    return max(r[k] for k in keys if k in r)


def choose(rows, b):
    ok = [r for r in rows if r["kind"] == "clef" and r["params_b"] <= b["max_params_b"]]
    if not ok:
        raise SystemExit("Không ứng viên Clef nào thoả max_params_b")
    best = max(best_acc(r, b) for r in ok)
    near = [r for r in ok if best_acc(r, b) >= best - b["prefer_smaller_within"]]
    return min(near, key=lambda r: (r["params_b"], -best_acc(r, b)))


def to_markdown(rows, chosen, b):
    ft = [f"ft{n}_val_acc" for n in b["ft_sizes"]]
    cols = ["name", "kind", "params_b", "latency_ms_p50", "zs_val_acc", "zs_val_macro_f1", "zs_val_ece",
            "zs_val_auto_coverage", *ft, f"ft{b['ft_sizes'][-1]}_val_ece", f"ft{b['ft_sizes'][-1]}_val_auto_coverage"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: -best_acc(r, b)):
        cells = [f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, "-")) for c in cols]
        if r["name"] == chosen["name"]:
            cells[0] = f"**{cells[0]}** ✅"
        lines.append("| " + " | ".join(cells) + " |")
    best_laya = max((r for r in rows if r["kind"] == "laya"), key=lambda r: r["zs_val_acc"], default=None)
    notes = [
        "",
        f"- `zs_*`: zero-shot. `ftN_*`: LoRA {b['epochs']} epoch trên N câu train. `auto_coverage`: tỷ lệ câu được "
        f"tự động xử lý mà precision trên val vẫn ≥ target (0 = không đạt target ở ngưỡng nào).",
        "- Đọc đường ft theo N: tăng mạnh từ N nhỏ lên N lớn -> gán nhãn thêm dữ liệu thật đáng tiền; "
        "đã phẳng -> nút thắt nằm ở schema/mô tả intent hoặc chất lượng nhãn, không phải số lượng.",
    ]
    if best_laya and best_laya["zs_val_acc"] > best_acc(chosen, b):
        notes.append(f"- ⚠️ {best_laya['name']} (zero-shot) vượt Clef đã chọn. Vòng lặp hiện chỉ fine-tune được Clef; "
                     "muốn dùng Laya phải viết stage train riêng cho Laya.")
    if best_laya and best_laya.get("zs_val_options_collapsed_frac", 0) > 0:
        notes.append(f"- {best_laya['name']}: {best_laya['zs_val_options_collapsed_frac']:.0%} câu bị gộp lựa chọn "
                     "(ngân sách token của Laya không phân biệt được hết 60 intent).")
    return "\n".join(lines + notes)


def main():
    P = params()
    b = P["bakeoff"]
    train, val = load_split("train"), load_split("val")
    mlflow = setup_mlflow("intent-bakeoff")
    rows = []
    for c in b["candidates"]:
        print(f"==== bakeoff: {c['name']} ({c['hf_id']})")
        rows.append(run_candidate(c, P, train, val, mlflow))
        write_json(REPORTS / "bakeoff.json", rows)  # ghi dần để không mất kết quả nếu ứng viên sau lỗi

    chosen = choose(rows, b)
    md = to_markdown(rows, chosen, b)
    (REPORTS / "bakeoff.md").write_text(md + "\n", encoding="utf-8")
    write_json(REPORTS / "bakeoff_choice.json", {"name": chosen["name"], "hf_id": chosen["hf_id"],
                                                 "revision": next(c.get("revision") for c in b["candidates"]
                                                                  if c["name"] == chosen["name"]),
                                                 "best_val_acc": best_acc(chosen, b)})
    print(md)
    print(f"Đề xuất: {chosen['name']} -> nếu đồng ý, ghi `train.base_model: {chosen['hf_id']}` (+ base_revision) "
          "vào params.yaml")


if __name__ == "__main__":
    main()
