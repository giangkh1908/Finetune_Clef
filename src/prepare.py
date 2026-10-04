"""Stage `prepare`: MASSIVE vi-VN (gộp cả train/dev/test gốc) -> chia lại train / val / golden + train_probe.

Tỷ lệ (params.yaml: prepare.split):
  100% dữ liệu -> golden 15%  |  85% còn lại -> train 82%, val 18%   (≈ 69.7% / 15.3% / 15% tổng)
Cách chia:
  - Phân tầng theo intent: intent nào cũng có mặt ở cả 3 tập với tỷ lệ như trên.
  - Gom theo câu đã chuẩn hoá: câu trùng chữ (vd. "tăng âm lượng" xuất hiện nhiều lần) luôn nằm cùng một tập,
    nên val/golden không chứa câu đã thấy trong train (split gốc của MASSIVE có ~5% trùng như vậy).
  - Không còn so được 1-1 với số công bố trên split gốc của MASSIVE; `orig_partition` được giữ lại để đối chiếu.
Request đầy đủ cho Clef (state + 60 lựa chọn intent) được dựng lúc chạy bởi src/records.py.

Meta mỗi câu:
  agree         : số người chấm (0-3) của MASSIVE xác nhận intent đúng -> đo trần chất lượng nhãn
  seen_in_train : câu (đã chuẩn hoá) có y hệt trong train (sau khi gom nhóm phải luôn là False; check_data kiểm tra)
"""

import json
import random
from collections import Counter, defaultdict

from src.common import GOLDEN_FILE, RAW, REPORTS, SPLIT_FILES, params, write_json, write_jsonl


def norm_text(s: str) -> str:
    return " ".join(s.lower().split())


def split_groups(rows: list[dict], golden_frac: float, val_frac_of_rest: float, seed: int) -> dict:
    """Chia theo nhóm câu trùng chữ, phân tầng theo intent (intent đa số của nhóm)."""
    groups = defaultdict(list)
    for r in rows:
        groups[norm_text(r["text"])].append(r)
    by_intent = defaultdict(list)
    for key in sorted(groups):
        g = groups[key]
        by_intent[Counter(r["intent"] for r in g).most_common(1)[0][0]].append(g)

    rng = random.Random(seed)
    out = {"train": [], "val": [], "golden": []}
    for intent in sorted(by_intent):
        gs = by_intent[intent]
        rng.shuffle(gs)
        n = sum(len(g) for g in gs)
        n_golden = round(n * golden_frac)
        n_val = round((n - n_golden) * val_frac_of_rest)
        taken = {"golden": 0, "val": 0}
        for g in gs:
            if taken["golden"] < n_golden:
                dst = "golden"
            elif taken["val"] < n_val:
                dst = "val"
            else:
                dst = "train"
            if dst != "train":
                taken[dst] += len(g)
            out[dst].extend(g)
    for v in out.values():
        v.sort(key=lambda r: r["id"])
    return out


def main():
    P = params()
    p = P["prepare"]
    src = RAW / "massive" / f"{P['fetch']['locale']}.jsonl"
    rows = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "id": f"{r['locale']}-{r['id']}",
                "text": r["utt"].strip(),
                "intent": r["intent"],
                "scenario": r["scenario"],
                "agree": sum(j["intent_score"] == 1 for j in r["judgments"]),
                "worker_id": r["worker_id"],
                "orig_partition": r["partition"],
            })

    s = p["split"]
    splits = split_groups(rows, s["golden"], s["val_of_rest"], s["seed"])

    dropped = Counter()
    if p["min_agree_train"] > 0:  # tuỳ chọn: bỏ câu train mà người chấm không xác nhận intent
        keep = [r for r in splits["train"] if r["agree"] >= p["min_agree_train"]]
        dropped["train_low_agree"] = len(splits["train"]) - len(keep)
        splits["train"] = keep

    train_texts = {norm_text(r["text"]) for r in splits["train"]}
    for split in ("val", "golden"):
        for r in splits[split]:
            r["seen_in_train"] = norm_text(r["text"]) in train_texts

    probe = random.Random(p["seed"]).sample(splits["train"], p["train_probe_size"])

    write_jsonl(SPLIT_FILES["train"], splits["train"])
    write_jsonl(SPLIT_FILES["train_probe"], probe)
    write_jsonl(SPLIT_FILES["val"], splits["val"])
    write_jsonl(GOLDEN_FILE, splits["golden"])

    total = sum(len(v) for v in splits.values())
    stats = {"total": total, "dropped": dict(dropped)}
    for split, rs in splits.items():
        c = Counter(r["intent"] for r in rs)
        stats[split] = {
            "rows": len(rs),
            "frac_of_total": round(len(rs) / total, 4),
            "intents": len(c),
            "scenarios": len({r["scenario"] for r in rs}),
            "majority_intent_frac": round(max(c.values()) / len(rs), 4),
            "min_intent_count": min(c.values()),
            "agree3_frac": round(sum(r["agree"] == 3 for r in rs) / len(rs), 4),
        }
        if split != "train":
            stats[split]["seen_in_train_frac"] = round(sum(r["seen_in_train"] for r in rs) / len(rs), 4)
    stats["val_frac_of_train_plus_val"] = round(len(splits["val"]) / (len(splits["val"]) + len(splits["train"])), 4)
    stats["train_probe"] = {"rows": len(probe)}
    write_json(REPORTS / "data_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
