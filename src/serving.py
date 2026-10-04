"""MLflow pyfunc: thứ được đăng ký vào Model Registry và đem đi triển khai.
Gói kèm adapter + head, schema (60 intent + mô tả) và ngưỡng đã chọn trên val, nên inference luôn khớp lúc train.
Mỗi lần predict tạo một MLflow trace (encode -> Clef -> quyết định tự động / chuyển người)."""

import json
from pathlib import Path

import mlflow
import pandas as pd


class IntentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from src import records
        from src.engines import ClefEngine

        with open(context.artifacts["config"], encoding="utf-8") as f:
            self.config = json.load(f)
        # Dùng đúng schema đóng gói cùng model, không phải schema/ đang có trong repo lúc load.
        records.SCHEMA_FILE = Path(context.artifacts["schema"]) / records.SCHEMA_FILE.name
        records.load_schema.cache_clear()
        self.threshold = self.config["threshold"]
        c = self.config
        self.engine = ClefEngine(c["base_model"], c["base_revision"], context.artifacts["checkpoint"], c["dtype"],
                                 batch_size=c["batch_size"], max_length=c["max_length"])

    @mlflow.trace(name="intent.predict", span_type="CHAIN")
    def predict(self, context, model_input: pd.DataFrame, params=None):
        """Cột vào: text (câu tiếng Việt). Ra: intent, confidence, top3, decision (auto | human)."""
        rows = [{"id": str(i), "text": t} for i, t in enumerate(model_input["text"])]
        with mlflow.start_span(name="clef", span_type="LLM") as span:
            span.set_inputs({"text": [r["text"] for r in rows]})
            preds = self.engine.predict(rows)
            span.set_outputs({"intent": [p["pred"] for p in preds], "confidence": [p["confidence"] for p in preds]})
        out = []
        for p in preds:
            top3 = sorted(p["probs"].items(), key=lambda kv: -kv[1])[:3]
            out.append({"intent": p["pred"], "confidence": p["confidence"],
                        "top3": json.dumps(top3, ensure_ascii=False),
                        "decision": "auto" if p["confidence"] >= self.threshold else "human"})
        mlflow.update_current_trace(tags={"threshold": str(self.threshold)})
        return pd.DataFrame(out)

    # Dùng bởi golden_eval: chấm đúng engine + schema đã đóng gói.
    def predict_rows(self, rows):
        return self.engine.predict(rows)
