"""Stage `check_data`: các kiểm tra chạy trên CPU, trước khi tốn tiền GPU. Có check nào fail thì pipeline dừng.

Phần kiểm tra cần tokenizer/model (độ dài schema, batch == từng câu, overfit 50 mẫu) nằm ở stage `sanity`.
"""

import sys
from collections import Counter

from clef_finetune.data import label_index, option_ids, validate_record

from src.common import GOLDEN_FILE, REPORTS, load_split, params, read_jsonl, write_json
from src.metrics import choose_threshold, score_rows, summarize
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
    synth = [{"confidence": c, "correct": c >= 0.5} for c in (0.1, 0.2, 0.3, 0.6, 0.7, 0.9)]
    t = choose_threshold(synth, 1.0)
    check("threshold_logic", t["precision"] == 1.0 and t["coverage"] == 0.5, str(t))

    write_json(REPORTS / "checks.json", checks)
    failed = [k for k, v in checks.items() if not v["pass"]]
    if failed:
        print("CHECK FAIL:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
