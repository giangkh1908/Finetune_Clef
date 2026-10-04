"""Chạy model quyết định trên các câu đã chuẩn bị -> dòng dự đoán {pred, confidence, probs} cho src/metrics.py.

- ClefEngine: Clef / Clef-flash (base hoặc base + checkpoint LoRA+head của clef-finetune). Dùng encoder và model
  của Cloudflare qua clef_finetune, có batch (sanity kiểm tra batch cho kết quả giống từng câu).
- LayaEngine: Laya (convaiinnovations/laya-*, vanty120/laya-vi), chỉ dùng trong bakeoff để so sánh zero-shot.
Cả hai nhận đúng một request (src/records.py), nên điểm số so sánh được với nhau.
"""

import statistics
import time

from src.records import question_id, questions, to_record


def _row(r: dict, probs: dict) -> dict:
    pred = max(probs, key=probs.__getitem__)
    return {**r, "pred": pred, "confidence": probs[pred], "probs": {k: round(v, 6) for k, v in probs.items()}}


class ClefEngine:
    def __init__(self, base_model: str, revision: str | None = None, checkpoint=None, dtype: str = "bfloat16",
                 device: str | None = None, batch_size: int = 8, max_length: int = 4096):
        import torch
        from clef_finetune.model import build_eval_model, resolve_model_dir

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = resolve_model_dir(base_model, revision)
        self.model, self.tok, _, self.jsm = build_eval_model(model_dir, device, getattr(torch, dtype), checkpoint)
        self.batch_size, self.max_length = batch_size, max_length

    def n_params_b(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e9

    def encode(self, rows):
        return [self.jsm.encode_record(self.tok, to_record(r, with_label=False), max_length=self.max_length)
                for r in rows]

    def logits(self, encoded):
        import torch

        device = next(self.model.parameters()).device
        with torch.inference_mode():
            out = self.model(self.jsm.collate_records(encoded, self.tok.pad_token_id, device))
        return [rec_logits[0].float() for rec_logits in out]

    def predict(self, rows: list[dict], batch_size: int | None = None) -> list[dict]:
        bs = batch_size or self.batch_size
        enc = self.encode(rows)
        order = sorted(range(len(rows)), key=lambda i: len(enc[i].input_ids))  # gom câu dài gần nhau, ít padding
        out = [None] * len(rows)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            for i, lg in zip(idx, self.logits([enc[i] for i in idx])):
                q = enc[i].questions[0]
                out[i] = _row(rows[i], dict(zip(q.option_ids, lg.softmax(-1).tolist())))
        return out

    def latency_ms(self, rows: list[dict]) -> float:
        import torch

        self.predict(rows[:2], batch_size=1)  # warmup
        ts = []
        for r in rows:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            self.predict([r], batch_size=1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        return statistics.median(ts)


class LayaEngine:
    def __init__(self, model_id: str, subfolder: str | None = None, revision: str | None = None,
                 batch_size: int = 32, head_max_len: int | None = None):
        import laya

        self.agent = laya.load(model_id, subfolder=subfolder, revision=revision)
        self.batch_size = batch_size
        # Ngân sách token mặc định của Laya không phân biệt được hết 60 intent (mọi câu bị gộp lựa chọn) -> nới ra.
        self.head_max_len = head_max_len

    def n_params_b(self) -> float:
        return sum(p.numel() for p in self.agent.model.parameters()) / 1e9

    def predict(self, rows: list[dict], batch_size: int | None = None) -> list[dict]:
        qid = question_id()
        res = self.agent.predict_batch([r["text"] for r in rows], questions(), batch_size=batch_size or self.batch_size,
                                       max_len=self.head_max_len, head_max_len=self.head_max_len)
        out = []
        for r, x in zip(rows, res):
            row = _row(r, dict(x["answers"][qid]["probabilities"]))
            usage = x.get("usage") or {}
            row["truncated"] = bool(usage.get("truncated"))
            row["options_collapsed"] = bool(usage.get("options"))  # Laya không phân biệt được hết 60 lựa chọn
            out.append(row)
        return out

    def latency_ms(self, rows: list[dict]) -> float:
        self.predict(rows[:2], batch_size=1)
        ts = []
        for r in rows:
            t = time.perf_counter()
            self.predict([r], batch_size=1)
            ts.append((time.perf_counter() - t) * 1000)
        return statistics.median(ts)


def build_engine(c: dict, P: dict):
    """c: một ứng viên trong params.yaml (bakeoff.candidates) hoặc {kind: clef, hf_id, revision} cho vòng lặp."""
    g = P["eval"]
    if c["kind"] == "clef":
        return ClefEngine(c["hf_id"], c.get("revision"), c.get("checkpoint"), g["dtype"],
                          batch_size=g["batch_size"], max_length=g["max_length"])
    if c["kind"] == "laya":
        return LayaEngine(c["hf_id"], c.get("subfolder"), c.get("revision"), batch_size=g["laya_batch_size"],
                          head_max_len=g["laya_head_max_len"])
    raise ValueError(f"kind không hỗ trợ: {c['kind']}")
