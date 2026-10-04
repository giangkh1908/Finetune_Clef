"""Stage `prepare`: tạo train / val / golden.

ViText2SQL dịch cả tên bảng/cột sang tiếng Việt nên SQL của nó không chạy được trên DB nào. Nhưng thứ tự câu
trong từng DB trùng 1-1 với Spider, nên ghép theo vị trí để lấy lại schema + SQL gốc tiếng Anh chạy được.

Ánh xạ split (giữ nguyên cách chia của ViText2SQL, 3 tập không dùng chung database):
    ViText2SQL train -> train   (fine-tune)
    ViText2SQL dev   -> val     (chọn hyperparam, checkpoint, ngưỡng)
    ViText2SQL test  -> golden  (chấm 1 lần mỗi milestone)
"""

import json
import random
from collections import Counter, defaultdict

from src.common import DB_DIR, GOLDEN_FILE, RAW, REPORTS, SPLIT_FILES, params, write_json, write_jsonl
from src.sql_utils import build_schema, execute, hardness, normalize_sql

VI_TO_SPLIT = {"train": "train", "dev": "val", "test": "golden"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def strip_values(o):
    if isinstance(o, dict):
        return {k: strip_values(v) for k, v in o.items()}
    if isinstance(o, list):
        return [strip_values(x) for x in o]
    if isinstance(o, str):
        return "V"
    if isinstance(o, float):
        return "N"
    return o


def same_structure(a, b) -> bool:
    return json.dumps(strip_values(a), sort_keys=True) == json.dumps(strip_values(b), sort_keys=True)


def db_path(db_id):
    return DB_DIR / db_id / f"{db_id}.sqlite"


def main():
    p = params()["prepare"]
    sp_dir = RAW / "spider"
    vi_dir = RAW / "ViText2SQL" / "data" / p["level"]

    spider = sum((load(sp_dir / f) for f in ("train_spider.json", "train_others.json", "dev.json")), [])
    tables = {t["db_id"]: t for t in load(sp_dir / "tables.json")}

    sp_by_db = defaultdict(list)
    for e in spider:
        sp_by_db[e["db_id"]].append(e)
    vi_by_db = defaultdict(list)
    for vi_split, split in VI_TO_SPLIT.items():
        for e in load(vi_dir / f"{vi_split}.json"):
            vi_by_db[e["db_id"]].append((split, e))

    assert set(sp_by_db) == set(vi_by_db), "danh sách database không khớp"
    pairs, struct_ok = [], 0
    for db_id, vi_rows in vi_by_db.items():
        sp_rows = sp_by_db[db_id]
        assert len(sp_rows) == len(vi_rows), f"{db_id}: số câu không khớp"
        assert len({s for s, _ in vi_rows}) == 1, f"{db_id}: database nằm ở nhiều split"
        for sp, (split, v) in zip(sp_rows, vi_rows):
            struct_ok += same_structure(sp["sql"], v["sql"])
            pairs.append((split, sp, v))

    schemas = {db: build_schema(tables[db], str(db_path(db)), p["sample_rows"]) for db in sp_by_db}

    out = {"train": [], "val": [], "golden": []}
    dropped = Counter()
    for i, (split, sp, v) in enumerate(pairs):
        db_id = sp["db_id"]
        sql = normalize_sql(sp["query"])
        ok, res = execute(str(db_path(db_id)), sql)
        if not ok:  # vài câu SQL vàng của Spider tự lỗi
            dropped[split] += 1
            print(f"  [bỏ] {split} {db_id}: {sql[:80]} -> {res}")
            continue
        base = {"db_id": db_id, "schema": schemas[db_id], "sql": sql, "hardness": hardness(sp["sql"])}
        out[split].append({"id": f"{split}-{i}", "lang": "vi", "question": " ".join(v["question"].split()), **base})
        if split == "train" and p["add_english_to_train"]:
            out[split].append({"id": f"{split}-{i}-en", "lang": "en", "question": sp["question"].strip(), **base})

    rng = random.Random(p["seed"])
    rng.shuffle(out["train"])
    vi_train = [r for r in out["train"] if r["lang"] == "vi"]
    probe = sorted(rng.sample(vi_train, p["train_probe_size"]), key=lambda r: r["id"])

    write_jsonl(SPLIT_FILES["train"], out["train"])
    write_jsonl(SPLIT_FILES["train_probe"], probe)
    write_jsonl(SPLIT_FILES["val"], out["val"])
    write_jsonl(GOLDEN_FILE, out["golden"])

    stats = {"aligned_pairs": len(pairs), "struct_exact_ratio": round(struct_ok / len(pairs), 4)}
    for split, rows in {**out, "train_probe": probe}.items():
        hard = Counter(r["hardness"] for r in rows if r["lang"] == "vi")
        stats[split] = {
            "rows": len(rows),
            "vi_rows": sum(r["lang"] == "vi" for r in rows),
            "databases": len({r["db_id"] for r in rows}),
            "dropped_bad_gold": dropped.get(split, 0),
            **{f"hard_{k}": hard[k] for k in ("easy", "medium", "hard", "extra")},
        }
    write_json(REPORTS / "data_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
