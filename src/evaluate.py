"""Chấm điểm: Execution Accuracy (EX) trên SQLite thật + đường precision/coverage theo ngưỡng tin cậy.

Quyết định khi chạy thật: chỉ trả lời nếu SQL chạy được VÀ confidence >= ngưỡng, ngược lại từ chối.
=> EX đo chất lượng model; precision/coverage tại ngưỡng đo chất lượng *quyết định*.
"""

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from src.common import DB_DIR
from src.sql_utils import execute, needs_order, normalize_sql, results_match

LEVELS = ("easy", "medium", "hard", "extra")


def _score_one(args):
    rec, timeout = args
    db = str(DB_DIR / rec["db_id"] / f"{rec['db_id']}.sqlite")
    ok_g, gold = execute(db, rec["sql"], timeout)
    if not ok_g:
        return {"skip": True}
    ok_p, pred = execute(db, rec["pred"], timeout) if rec.get("pred") else (False, "empty prediction")
    return {
        "skip": False,
        "exec_ok": ok_p,
        "correct": bool(ok_p and results_match(gold, pred, needs_order(rec["sql"]))),
        "em": normalize_sql(rec.get("pred", "")).lower() == normalize_sql(rec["sql"]).lower(),
        "exec_error": None if ok_p else str(pred)[:300],
    }


def score_rows(rows: list[dict], timeout: float = 30.0, workers: int | None = None) -> list[dict]:
    """Trả về bản sao rows có thêm exec_ok / correct / em / exec_error (bỏ các câu SQL vàng lỗi)."""
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        res = list(ex.map(_score_one, [(r, timeout) for r in rows], chunksize=16))
    return [{**r, **s} for r, s in zip(rows, res) if not s["skip"]]


def summarize(scored: list[dict], prefix: str = "") -> dict:
    by = defaultdict(lambda: [0, 0])
    for r in scored:
        for k in ("all", r["hardness"]):
            by[k][0] += r["correct"]
            by[k][1] += 1
    n = max(len(scored), 1)
    m = {f"{prefix}ex": by["all"][0] / max(by["all"][1], 1), f"{prefix}n": len(scored)}
    for k in LEVELS:
        if by[k][1]:
            m[f"{prefix}ex_{k}"] = by[k][0] / by[k][1]
    m[f"{prefix}exact_match"] = sum(r["em"] for r in scored) / n
    m[f"{prefix}exec_error_rate"] = sum(not r["exec_ok"] for r in scored) / n
    if scored and "confidence" in scored[0]:
        m[f"{prefix}mean_confidence"] = sum(r["confidence"] for r in scored) / n
    return m


# ---------------------------------------------------------------- ngưỡng tin cậy


def decide(r: dict, threshold: float) -> bool:
    return bool(r["exec_ok"]) and r["confidence"] >= threshold


def selective_metrics(scored: list[dict], threshold: float) -> dict:
    answered = [r for r in scored if decide(r, threshold)]
    n_ans = len(answered)
    return {
        "threshold": threshold,
        "coverage": n_ans / max(len(scored), 1),
        "precision": sum(r["correct"] for r in answered) / n_ans if n_ans else 1.0,
        "answered": n_ans,
    }


def threshold_curve(scored: list[dict]) -> list[dict]:
    cands = sorted({0.0, *(round(r["confidence"], 4) for r in scored)})
    return [selective_metrics(scored, t) for t in cands]


def choose_threshold(scored: list[dict], target_precision: float) -> dict:
    """Coverage lớn nhất mà precision >= target. Không đạt được thì lấy ngưỡng có precision cao nhất."""
    curve = threshold_curve(scored)
    ok = [c for c in curve if c["precision"] >= target_precision and c["answered"] > 0]
    if ok:
        best = max(ok, key=lambda c: (c["coverage"], -c["threshold"]))
        return {**best, "target_reached": True}
    best = max((c for c in curve if c["answered"] > 0), key=lambda c: (c["precision"], c["coverage"]))
    return {**best, "target_reached": False}
