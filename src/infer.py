"""Hỏi trên một database SQLite bằng model trong Registry. Mỗi lần gọi có trace trong MLflow (experiment text2sql-inference).

    python -m src.infer --db my.sqlite --question "Có bao nhiêu khách hàng ở Hà Nội?"
    python -m src.infer --alias candidate --db ... --question ...
"""

import argparse

import pandas as pd

from src.common import params, setup_mlflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--alias", default="champion")
    args = ap.parse_args()

    mlflow = setup_mlflow("text2sql-inference")
    model = mlflow.pyfunc.load_model(f"models:/{params()['project']['registered_model']}@{args.alias}")
    out = model.predict(pd.DataFrame([{"question": args.question, "db_path": args.db}])).iloc[0]
    print(f"SQL       : {out['sql']}")
    print(f"Confidence: {out['confidence']:.3f} -> {out['decision']}")
    print(f"Kết quả   : {out['result']}" if out["decision"] == "answer" else "Model không đủ chắc chắn, từ chối trả lời.")


if __name__ == "__main__":
    main()
