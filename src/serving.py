"""MLflow pyfunc: thứ được đăng ký vào Model Registry và đem đi triển khai.
Gói kèm model, ngưỡng đã chọn trên val, và code prompt nên inference luôn khớp lúc train.
Mỗi lần predict tạo một MLflow trace (schema -> generate -> execute -> decision)."""

import json

import mlflow
import pandas as pd


class Text2SQLModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from src.engine import Engine

        with open(context.artifacts["config"], encoding="utf-8") as f:
            self.config = json.load(f)
        self.threshold = self.config["threshold"]
        self.engine = Engine(context.artifacts["model_dir"], batch_size=self.config.get("batch_size", 16),
                             max_new_tokens=self.config.get("max_new_tokens", 300))

    @mlflow.trace(name="text2sql.predict", span_type="CHAIN")
    def predict(self, context, model_input: pd.DataFrame, params=None):
        """Cột vào: question + (schema hoặc db_path). Có db_path thì chạy SQL và trả kết quả."""
        from src.sql_utils import execute

        rows = model_input.to_dict("records")
        with mlflow.start_span(name="build_schema") as span:
            for r in rows:
                if not r.get("schema"):
                    r["schema"] = _schema_from_db(r["db_path"])
            span.set_attributes({"n": len(rows)})

        with mlflow.start_span(name="generate", span_type="LLM") as span:
            span.set_inputs({"questions": [r["question"] for r in rows]})
            preds = self.engine.predict(rows)
            span.set_outputs({"sql": [p["pred"] for p in preds], "confidence": [p["confidence"] for p in preds]})

        out = []
        for p in preds:
            res = {"sql": p["pred"], "confidence": p["confidence"], "exec_ok": None, "result": None}
            if p.get("db_path"):
                with mlflow.start_span(name="execute", span_type="TOOL") as span:
                    ok, val = execute(p["db_path"], p["pred"])
                    res["exec_ok"] = ok
                    res["result"] = json.dumps(val[:50] if ok else val, ensure_ascii=False, default=str)
                    span.set_outputs({"exec_ok": ok})
            exec_pass = res["exec_ok"] is not False
            res["decision"] = "answer" if exec_pass and p["confidence"] >= self.threshold else "abstain"
            out.append(res)
        mlflow.update_current_trace(tags={"threshold": str(self.threshold)})
        return pd.DataFrame(out)


def _schema_from_db(db_path):
    from src.infer_utils import table_info_from_sqlite
    from src.sql_utils import build_schema

    return build_schema(table_info_from_sqlite(db_path), db_path, sample_rows=3)
