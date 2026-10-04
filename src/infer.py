"""Phân loại một câu tiếng Việt bằng model trong Registry. Mỗi lần gọi có trace trong MLflow (experiment intent-inference).

    python -m src.infer --text "đặt báo thức lúc sáu giờ sáng mai"
    python -m src.infer --alias candidate --text ...
"""

import argparse

import pandas as pd

from src.common import params, setup_mlflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--alias", default="champion")
    args = ap.parse_args()

    mlflow = setup_mlflow("intent-inference")
    model = mlflow.pyfunc.load_model(f"models:/{params()['project']['registered_model']}@{args.alias}")
    out = model.predict(pd.DataFrame([{"text": args.text}])).iloc[0]
    print(f"Intent    : {out['intent']}")
    print(f"Confidence: {out['confidence']:.3f} -> {out['decision']}")
    print(f"Top 3     : {out['top3']}")
    if out["decision"] == "human":
        print("Dưới ngưỡng chọn trên val: chuyển người xử lý / hỏi lại người dùng.")


if __name__ == "__main__":
    main()
