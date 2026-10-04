"""Sinh SQL kèm confidence (trung bình xác suất token, dạng exp(mean logprob)).
Dùng chung cho sanity / bakeoff / eval_val / golden / serving để mọi nơi sinh giống hệt nhau."""

import math
import time

from src.prompting import clean_sql, render_prompt


class Engine:
    def __init__(self, model_path: str, backend: str = "hf", batch_size: int = 16, max_new_tokens: int = 300):
        from transformers import AutoTokenizer

        self.model_path = model_path
        self.backend = backend
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if backend == "vllm":
            from vllm import LLM

            self.llm = LLM(model=model_path, dtype="bfloat16", max_model_len=16384, seed=0)
        else:
            import torch
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map="auto").eval()

    @classmethod
    def from_model(cls, model, tok, batch_size=16, max_new_tokens=300):
        """Dùng model đang có trong bộ nhớ (vd. ngay sau khi train ở bước sanity)."""
        self = cls.__new__(cls)
        self.model_path, self.backend = "<in-memory>", "hf"
        self.batch_size, self.max_new_tokens = batch_size, max_new_tokens
        self.tok, self.model = tok, model
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        return self

    def prompts(self, rows):
        return [render_prompt(self.tok, r["schema"], r["question"]) for r in rows]

    def predict(self, rows: list[dict]) -> list[dict]:
        """rows cần 'schema' và 'question'. Trả về rows + pred, confidence, raw_output."""
        prompts = self.prompts(rows)
        outs = self._vllm(prompts) if self.backend == "vllm" else self._hf(prompts)
        return [{**r, "pred": clean_sql(text), "confidence": conf, "raw_output": text}
                for r, (text, conf) in zip(rows, outs)]

    def latency_ms(self, rows: list[dict]) -> float:
        """Latency trung vị khi xử lý từng câu một (batch=1)."""
        times = []
        for r in rows:
            t0 = time.perf_counter()
            self.predict([r]) if self.backend == "vllm" else self._hf(self.prompts([r]), batch_size=1)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        return times[len(times) // 2]

    # ------------------------------------------------------------ backends

    def _hf(self, prompts, batch_size=None):
        import torch

        bs = batch_size or self.batch_size
        model, tok = self.model, self.tok
        was_training = model.training
        model.eval()
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
        results = [None] * len(prompts)
        for b in range(0, len(order), bs):
            idx = order[b: b + bs]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id, return_dict_in_generate=True,
                                     output_scores=True)
            seqs = gen.sequences[:, enc["input_ids"].shape[1]:]
            logps = model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)
            eos_ids = set(tok.convert_tokens_to_ids([t for t in ("<|im_end|>",) if t in tok.get_vocab()]))
            eos_ids.add(tok.eos_token_id)
            for row, i in enumerate(idx):
                ids, lp = seqs[row].tolist(), logps[row].tolist()
                n = len(ids)
                for j, t in enumerate(ids):  # tính tới và gồm token kết thúc đầu tiên
                    if t in eos_ids:
                        n = j + 1
                        break
                vals = [v for v in lp[:n] if math.isfinite(v)]
                conf = math.exp(sum(vals) / len(vals)) if vals else 0.0
                results[i] = (tok.decode(ids[:n], skip_special_tokens=True), conf)
        if was_training:
            model.train()
        return results

    def _vllm(self, prompts):
        from vllm import SamplingParams

        res = self.llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens, logprobs=0))
        out = []
        for r in res:
            o = r.outputs[0]
            n = max(len(o.token_ids), 1)
            out.append((o.text, math.exp(o.cumulative_logprob / n) if o.cumulative_logprob is not None else 0.0))
        return out
