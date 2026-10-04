"""Chấm điểm phân loại intent + đường precision/coverage theo ngưỡng tin cậy.

Một dòng dự đoán: {id, intent (vàng), pred, confidence (= xác suất của pred), probs {intent: p}, + meta của câu}.
Quyết định khi chạy thật: confidence >= ngưỡng -> tự xử lý, ngược lại chuyển người (hoặc hỏi lại người dùng).
=> accuracy / NLL / ECE đo chất lượng model; precision/coverage tại ngưỡng đo chất lượng *quyết định*.

Accuracy, macro-F1, NLL, Brier, ECE lấy từ clef_finetune.metrics để khớp đúng với `clef-finetune eval`.
"""

from collections import Counter

from clef_finetune.metrics import brier_score, ece, macro_f1, nll

from src.records import scenario_of


def score_rows(rows: list[dict]) -> list[dict]:
    return [{**r, "correct": r["pred"] == r["intent"],
             "scenario_correct": scenario_of(r["pred"]) == scenario_of(r["intent"])} for r in rows]


def summarize(scored: list[dict], prefix: str = "") -> dict:
    if not scored:
        return {f"{prefix}n": 0}
    labels = sorted(scored[0]["probs"])
    idx = {k: i for i, k in enumerate(labels)}
    probs = [[r["probs"][k] for k in labels] for r in scored]
    gold = [idx[r["intent"]] for r in scored]
    pred = [idx[r["pred"]] for r in scored]
    n = len(scored)
    m = {
        f"{prefix}n": n,
        f"{prefix}acc": sum(r["correct"] for r in scored) / n,
        f"{prefix}macro_f1": macro_f1(pred, gold, len(labels)),
        f"{prefix}nll": nll(probs, gold),
        f"{prefix}brier": brier_score(probs, gold),
        f"{prefix}ece": ece(probs, gold),
        f"{prefix}scenario_acc": sum(r["scenario_correct"] for r in scored) / n,
        f"{prefix}mean_confidence": sum(r["confidence"] for r in scored) / n,
    }
    # Câu có y hệt trong train. prepare đã gom nhóm nên bình thường không có; còn lại để không mù nếu đổi cách chia.
    unseen = [r for r in scored if not r.get("seen_in_train")]
    if unseen and len(unseen) < n:
        m[f"{prefix}acc_unseen"] = sum(r["correct"] for r in unseen) / len(unseen)
    # Nhãn mà cả 3 người chấm của MASSIVE đồng ý là đúng intent: trần chất lượng nhãn.
    agree = [r for r in scored if r.get("agree") == 3]
    if agree and len(agree) < n:
        m[f"{prefix}acc_agree3"] = sum(r["correct"] for r in agree) / len(agree)
    return m


def per_intent(scored: list[dict]) -> list[dict]:
    tot, ok, predicted = Counter(), Counter(), Counter()
    for r in scored:
        tot[r["intent"]] += 1
        ok[r["intent"]] += r["correct"]
        predicted[r["pred"]] += 1
    out = []
    for k in sorted(tot):
        prec = ok[k] / predicted[k] if predicted[k] else 0.0
        rec = ok[k] / tot[k]
        out.append({"intent": k, "n": tot[k], "recall": rec, "precision": prec,
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0})
    return sorted(out, key=lambda x: x["f1"])


def top_confusions(scored: list[dict], k: int = 10) -> list[tuple[str, str, int]]:
    c = Counter((r["intent"], r["pred"]) for r in scored if not r["correct"])
    return [(g, p, n) for (g, p), n in c.most_common(k)]


# ---------------------------------------------------------------- ngưỡng tin cậy


def decide(r: dict, threshold: float) -> bool:
    return r["confidence"] >= threshold


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
