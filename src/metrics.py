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
# Bài toán nhị phân "tự động xử lý hay chuyển người" tại ngưỡng t (positive = model trả lời đúng):
#   TP: confidence >= t và đúng   -> tự động, đúng        FP: confidence >= t và sai -> tự động, SAI (lỗi lọt ra ngoài)
#   FN: confidence <  t và đúng   -> chuyển người thừa    TN: confidence <  t và sai -> chuyển người, chặn được lỗi
# precision = TP/(TP+FP): tỷ lệ đúng trong phần tự động | recall = TP/(TP+FN): phần câu đúng được tự động
# FPR = FP/(FP+TN): phần câu sai bị lọt qua ngưỡng      | coverage = (TP+FP)/n: phần câu được tự động


def decide(r: dict, threshold: float) -> bool:
    return r["confidence"] >= threshold


def selective_metrics(scored: list[dict], threshold: float) -> dict:
    tp = fp = fn = tn = 0
    for r in scored:
        if decide(r, threshold):
            tp += r["correct"]
            fp += not r["correct"]
        else:
            fn += r["correct"]
            tn += not r["correct"]
    n = max(len(scored), 1)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "coverage": (tp + fp) / n, "answered": tp + fp,
    }


def threshold_curve(scored: list[dict]) -> list[dict]:
    cands = sorted({0.0, *(round(r["confidence"], 4) for r in scored)})
    return [selective_metrics(scored, t) for t in cands]


def choose_threshold(scored: list[dict], gate: dict | None = None) -> dict:
    """Quét mọi ngưỡng, lấy ngưỡng có F1 lớn nhất trong các ngưỡng qua cổng chất lượng; không ngưỡng nào qua thì
    lấy F1 lớn nhất trên toàn đường cong (cổng sẽ FAIL). Bằng nhau thì lấy ngưỡng cao hơn = thận trọng hơn.
    F1 thuần không có ràng buộc hay kéo ngưỡng xuống rất thấp (FPR cao) vì câu đúng chiếm đa số."""
    curve = threshold_curve(scored)
    ok = [c for c in curve if gate and check_gate(c, gate)["pass"]]
    best = max(ok or curve, key=lambda c: (round(c["f1"], 6), c["threshold"]))
    return {**best, "within_gate": bool(ok)}


def coverage_at_precision(scored: list[dict], target_precision: float) -> float:
    """Coverage tự động lớn nhất mà precision >= target (0 nếu không ngưỡng nào đạt). Dùng để so model ở bakeoff."""
    ok = [c["coverage"] for c in threshold_curve(scored) if c["precision"] >= target_precision and c["answered"]]
    return max(ok, default=0.0)


def check_gate(m: dict, gate: dict) -> dict:
    """Cổng chất lượng trên metric tại ngưỡng. Trả về {pass, fails: [lý do]}."""
    rules = [("precision", ">=", gate["min_precision"]), ("recall", ">=", gate["min_recall"]),
             ("f1", ">=", gate["min_f1"]), ("fpr", "<=", gate["max_fpr"])]
    fails = [f"{k}={m[k]:.4f} {'<' if op == '>=' else '>'} {lim}" for k, op, lim in rules
             if (m[k] < lim if op == ">=" else m[k] > lim)]
    return {"pass": not fails, "fails": fails}


def confusion_line(m: dict) -> str:
    return (f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} | P={m['precision']:.4f} R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} FPR={m['fpr']:.4f} | coverage={m['coverage']:.4f}")
