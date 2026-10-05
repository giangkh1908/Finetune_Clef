"""Stage `check_data`: các kiểm tra chạy trên CPU, trước khi tốn tiền GPU. Có check nào fail thì pipeline dừng.

Phần kiểm tra cần tokenizer/model (độ dài schema, batch == từng câu, overfit 50 mẫu) nằm ở stage `sanity`.
"""

import sys
from collections import Counter

from clef_finetune.data import label_index, option_ids, validate_record

from src.common import GOLDEN_FILE, REPORTS, load_split, params, read_jsonl, write_json
from src.metrics import check_gate, choose_threshold, score_rows, summarize
from src.prepare import norm_text
from src.records import intents, question_id, questions, scenario_of, to_record


def onehot(labels, k, p=1.0):
    rest = (1 - p) / (len(labels) - 1)
    return {x: (p if x == k else rest) for x in labels}


def main():
    P = params()
    train, probe, val = load_split("train"), load_split("train_probe"), load_split("val")
    golden = read_jsonl(GOLDEN_FILE)  # chỉ dùng để kiểm tra rò rỉ / nhãn, không chấm model
    allrows = train + val + golden
    checks = {}

    def check(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    # ---- tách tập
    ids = {s: {r["id"] for r in rows} for s, rows in (("train", train), ("val", val), ("golden", golden))}
    check("ids_disjoint", not (ids["train"] & ids["val"] or ids["train"] & ids["golden"] or ids["val"] & ids["golden"]))
    check("train_probe_subset_of_train", all(r["id"] in ids["train"] for r in probe), f"{len(probe)} mẫu")

    # prepare gom câu trùng chữ vào cùng một tập -> val/golden không được chứa câu đã có trong train.
    max_ov = P["check"]["max_text_overlap"]
    train_t = {norm_text(r["text"]) for r in train}
    for name, rows in (("val", val), ("golden", golden)):
        ov = sum(norm_text(r["text"]) in train_t for r in rows)
        check(f"text_overlap_{name}_low", ov / len(rows) <= max_ov, f"{ov}/{len(rows)} (ngưỡng {max_ov:.0%})")
        check(f"seen_in_train_flag_{name}", sum(r["seen_in_train"] for r in rows) == ov)

    # Tỷ lệ chia đúng như params.yaml (golden 15% tổng; val 18% của phần còn lại), sai lệch do gom nhóm phải nhỏ.
    sp = P["prepare"]["split"]
    n_all = len(allrows)
    g_frac, v_frac = len(golden) / n_all, len(val) / (len(val) + len(train))
    check("split_ratio", abs(g_frac - sp["golden"]) < 0.01 and abs(v_frac - sp["val_of_rest"]) < 0.01,
          f"golden={g_frac:.4f} val/(train+val)={v_frac:.4f}")
    for name, rows in (("val", val), ("golden", golden)):
        miss = {r["intent"] for r in train} - {r["intent"] for r in rows}
        check(f"all_intents_in_{name}", not miss, str(sorted(miss)))

    # Phân phối intent / scenario của từng tập phải giống phân phối chung (phân tầng đúng, không tập nào bị lệch).
    max_tvd = P["check"]["max_label_dist_tvd"]
    for key in ("intent", "scenario"):
        c_all = Counter(r[key] for r in allrows)
        for name, rows in (("train", train), ("val", val), ("golden", golden)):
            c = Counter(r[key] for r in rows)
            tvd = 0.5 * sum(abs(c[k] / len(rows) - c_all[k] / n_all) for k in c_all)
            check(f"{key}_dist_{name}", tvd <= max_tvd, f"TVD={tvd:.4f} (ngưỡng {max_tvd})")

    # ---- nhãn và schema
    schema_intents = set(intents())
    used = {r["intent"] for r in allrows}
    check("labels_in_schema", used <= schema_intents, str(sorted(used - schema_intents)))
    check("schema_has_no_unused_intent", schema_intents <= {r["intent"] for r in train},
          str(sorted(schema_intents - {r["intent"] for r in train})))
    check("scenario_is_intent_prefix", all(scenario_of(r["intent"]) == r["scenario"] for r in allrows))
    check("no_empty_text", all(r["text"].strip() for r in allrows))
    c = Counter(r["intent"] for r in train)
    check("min_train_per_intent", min(c.values()) >= P["check"]["min_train_per_intent"],
          f"ít nhất: {c.most_common()[-1]}")

    # Đúng validator mà clef-finetune dùng lúc train: mọi request phải hợp lệ, nhãn là option hợp lệ.
    errs = [(r["id"], e) for r in allrows for e in validate_record(to_record(r))]
    check("clef_records_valid", not errs, str(errs[:3]))
    q = questions()[question_id()]
    order = option_ids(q)  # encoder của Cloudflare sắp xếp option id
    check("option_order_sorted", order == sorted(order) and order == intents())
    sample = train[:200]
    check("label_index_roundtrip", all(order[label_index(q, r["intent"])] == r["intent"] for r in sample))

    # ---- bộ chấm
    labels = intents()
    as_pred = lambda rows, p: [{**r, "pred": r["intent"], "confidence": p, "probs": onehot(labels, r["intent"], p)}
                               for r in rows]
    m = summarize(score_rows(as_pred(val, 0.99)))
    check("scorer_gold_as_pred_is_100", m["acc"] == 1.0 and m["macro_f1"] == 1.0, f"acc={m['acc']:.4f}")
    wrong = [{**r, "pred": labels[(labels.index(r["intent"]) + 1) % len(labels)], "confidence": 0.9,
              "probs": onehot(labels, labels[(labels.index(r["intent"]) + 1) % len(labels)], 0.9)} for r in val]
    m = summarize(score_rows(wrong))
    check("scorer_rejects_wrong", m["acc"] == 0.0, f"acc={m['acc']:.4f}")
    uni = [{**r, "pred": labels[0], "confidence": 1 / len(labels), "probs": {x: 1 / len(labels) for x in labels}}
           for r in val]
    import math
    m = summarize(score_rows(uni))
    check("nll_uniform_is_log_k", abs(m["nll"] - math.log(len(labels))) < 1e-6, f"nll={m['nll']:.4f}")

    # Logic chọn ngưỡng trên dữ liệu tổng hợp: biết trước đáp án.
    # 4 đúng (0.6..0.9) + 1 đúng thấp (0.2) + 3 sai (0.1, 0.3, 0.4): F1 max ở ngưỡng 0.6 (TP=4 FP=0 FN=1 TN=3),
    # hạ ngưỡng xuống 0.2 thì F1 = 0.83, ngưỡng 0 thì 0.77.
    synth = [{"confidence": c, "correct": ok} for c, ok in
             ((0.1, False), (0.2, True), (0.3, False), (0.4, False), (0.6, True), (0.7, True), (0.8, True), (0.9, True))]
    t = choose_threshold(synth)
    check("threshold_logic", (t["threshold"], t["tp"], t["fp"], t["fn"], t["tn"]) == (0.6, 4, 0, 1, 3)
          and abs(t["f1"] - 8 / 9) < 1e-9 and t["fpr"] == 0.0, str(t))
    g = check_gate(t, {"min_precision": 0.9, "min_recall": 0.9, "min_f1": 0.5, "max_fpr": 0.1})
    check("gate_logic", not g["pass"] and len(g["fails"]) == 1 and g["fails"][0].startswith("recall"), str(g))
    # Có cổng recall >= 0.9: chỉ ngưỡng <= 0.2 qua (R=1) -> F1 max trong đó là ngưỡng 0.2 (F1=0.83), không phải 0.6.
    t = choose_threshold(synth, {"min_precision": 0.0, "min_recall": 0.9, "min_f1": 0.0, "max_fpr": 1.0})
    check("threshold_within_gate", t["threshold"] == 0.2 and t["within_gate"], str(t))

    write_json(REPORTS / "checks.json", checks)
    failed = [k for k, v in checks.items() if not v["pass"]]
    if failed:
        print("CHECK FAIL:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
