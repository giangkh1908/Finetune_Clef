"""Stage `check_data`: các kiểm tra chạy trên CPU, trước khi tốn tiền GPU. Có check nào fail thì pipeline dừng.

Phần kiểm tra cần tokenizer/model (label masking, overfit 50 mẫu) nằm ở stage `sanity`.
"""

import sys

from src.common import GOLDEN_FILE, REPORTS, load_split, read_jsonl, write_json
from src.evaluate import choose_threshold, score_rows, summarize


def norm_q(q):
    return " ".join(q.lower().split())


def main():
    train, probe, val = load_split("train"), load_split("train_probe"), load_split("val")
    golden = read_jsonl(GOLDEN_FILE)  # chỉ dùng để kiểm tra rò rỉ, không chấm model
    checks = {}

    def check(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    dbs = {s: {r["db_id"] for r in rows} for s, rows in (("train", train), ("val", val), ("golden", golden))}
    check("db_disjoint_train_val", not dbs["train"] & dbs["val"], str(sorted(dbs["train"] & dbs["val"])[:5]))
    check("db_disjoint_train_golden", not dbs["train"] & dbs["golden"], str(sorted(dbs["train"] & dbs["golden"])[:5]))
    check("db_disjoint_val_golden", not dbs["val"] & dbs["golden"], str(sorted(dbs["val"] & dbs["golden"])[:5]))

    train_q = {norm_q(r["question"]) for r in train}
    leak_v = sum(norm_q(r["question"]) in train_q for r in val)
    leak_g = sum(norm_q(r["question"]) in train_q for r in golden)
    # Spider có vài câu hỏi chung chung trùng chữ ("có bao nhiêu ...") nhưng khác DB nên vẫn khác bài toán.
    check("question_overlap_val_low", leak_v / len(val) < 0.01, f"{leak_v}/{len(val)}")
    check("question_overlap_golden_low", leak_g / len(golden) < 0.01, f"{leak_g}/{len(golden)}")

    train_ids = {r["id"] for r in train}
    check("train_probe_subset_of_train", all(r["id"] in train_ids for r in probe), f"{len(probe)} mẫu")
    check("no_empty_fields", all(r["question"] and r["sql"] and r["schema"] for r in train + val + golden))

    # Bộ chấm: SQL vàng làm dự đoán phải đạt 100%.
    gold_as_pred = score_rows([{**r, "pred": r["sql"], "confidence": 1.0} for r in val])
    ex = summarize(gold_as_pred)["ex"]
    check("scorer_gold_as_pred_is_100", ex == 1.0, f"EX={ex:.4f}")

    # Bộ chấm: dự đoán sai chắc chắn (kết quả rỗng) phải bị chấm sai khi kết quả vàng khác rỗng.
    nonempty = [r for r in gold_as_pred if r["correct"]][:200]
    broken = score_rows([{**r, "pred": f"SELECT * FROM ({r['sql']}) WHERE 0"} for r in nonempty])
    false_pos = [r for r in broken if r["correct"]]
    from src.common import DB_DIR
    from src.sql_utils import execute
    false_pos = [r for r in false_pos
                 if execute(str(DB_DIR / r["db_id"] / f"{r['db_id']}.sqlite"), r["sql"])[1]]  # vàng khác rỗng
    check("scorer_rejects_wrong_results", not false_pos, f"{len(false_pos)} false positive")

    # Logic chọn ngưỡng trên dữ liệu tổng hợp: biết trước đáp án.
    synth = [{"exec_ok": True, "confidence": c, "correct": c >= 0.5} for c in (0.1, 0.2, 0.3, 0.6, 0.7, 0.9)]
    t = choose_threshold(synth, 1.0)
    check("threshold_logic", t["precision"] == 1.0 and t["coverage"] == 0.5, str(t))

    write_json(REPORTS / "checks.json", checks)
    failed = [k for k, v in checks.items() if not v["pass"]]
    if failed:
        print("CHECK FAIL:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
