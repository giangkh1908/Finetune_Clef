"""Stage `register`: đóng gói checkpoint đã chọn (LoRA adapter + joint head, vài trăm MB) + schema + ngưỡng thành
MLflow pyfunc, đăng ký vào Model Registry: MỖI lần chạy là một version mới (kể cả khi không qua cổng, để ghi nhận).
Version gắn tag TP/FP/FN/TN, P/R/F1/FPR tại ngưỡng trên val + kết quả cổng `gate_val`.
Chỉ version qua cổng val mới được alias `candidate`; chỉ golden_eval mới chọn `champion`.

Backbone 9B KHÔNG nằm trong artifact: pyfunc tải Clef-flash từ Hugging Face ở đúng revision đã pin (Apache-2.0).
Cần một thư mục release độc lập (merge LoRA vào backbone, load bằng code của Cloudflare)? Dùng `clef-finetune merge`.
"""

import json
import shutil

from src.common import OUTPUTS, REPORTS, ROOT, SCHEMA_DIR, lineage_tags, params, setup_mlflow, write_json


def main():
    P = params()
    name = P["project"]["registered_model"]
    t, g = P["train"], P["eval"]
    sel = json.loads((REPORTS / "selection.json").read_text(encoding="utf-8"))

    export = OUTPUTS / "export" / f"{sel['run']}-{sel['checkpoint']}"
    ck_src = ROOT / sel["checkpoint_path"]
    shutil.rmtree(export, ignore_errors=True)
    shutil.copytree(ck_src / "adapter", export / "checkpoint" / "adapter")
    for f in ("joint_head.safetensors", "train_config.json"):  # bỏ training_state.pt (optimizer, không cần để chạy)
        shutil.copy(ck_src / f, export / "checkpoint" / f)
    shutil.copytree(SCHEMA_DIR, export / "schema")
    config = {"base_model": t["base_model"], "base_revision": t["base_revision"], "dtype": g["dtype"],
              "max_length": g["max_length"], "batch_size": g["batch_size"],
              "threshold": sel["threshold"]["threshold"]}
    write_json(export / "serving_config.json", config)

    mlflow = setup_mlflow("intent-train")
    from mlflow import MlflowClient

    from src.serving import IntentModel

    with mlflow.start_run(run_name=f"register-{sel['run']}-{sel['checkpoint']}"):
        tags = {**lineage_tags(), "stage": "register", "source_run_id": sel["mlflow_run_id"]}
        mlflow.set_tags(tags)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=IntentModel(),
            artifacts={"checkpoint": str(export / "checkpoint"), "schema": str(export / "schema"),
                       "config": str(export / "serving_config.json")},
            code_paths=[str(ROOT / "src")],
            pip_requirements=[l.strip() for l in (ROOT / "requirements-gpu.txt").read_text().splitlines()
                              if l.strip() and not l.startswith(("#", "-r"))],
            registered_model_name=name,
        )
    version = info.registered_model_version
    client = MlflowClient()
    gate_ok = sel["gate_val"]["pass"]
    if gate_ok:
        client.set_registered_model_alias(name, "candidate", version)
    thr = sel["threshold"]
    vtags = {
        "val_acc": f"{sel['val_acc']:.4f}",
        "val_ece": f"{sel['val_metrics']['val_ece']:.4f}",
        **{f"val_{k}": str(thr[k]) for k in ("tp", "fp", "fn", "tn")},
        **{f"val_{k}": f"{thr[k]:.4f}" for k in ("precision", "recall", "f1", "fpr", "coverage")},
        "threshold": f"{thr['threshold']:.4f}",
        "gate_val": "pass" if gate_ok else "fail: " + "; ".join(sel["gate_val"]["fails"]),
        "stability_score": f"{sel['stability']['score']:.4f}",
        "diagnosis": sel["diagnosis"],
        "base_model": t["base_model"], "base_revision": t["base_revision"],
        "train_run": sel["run"], "checkpoint": sel["checkpoint"],
        **{k: v for k, v in tags.items() if k.startswith(("data.", "schema.", "git_sha", "dvc_lock"))},
    }
    for k, v in vtags.items():
        client.set_model_version_tag(name, version, k, v)
    write_json(REPORTS / "registered.json", {"name": name, "version": version,
                                             "alias": "candidate" if gate_ok else None, "gate_val": gate_ok,
                                             "model_uri": f"models:/{name}/{version}"})
    print(f"Đăng ký {name} v{version}: " + ("alias candidate" if gate_ok else
                                           f"KHÔNG qua cổng val ({'; '.join(sel['gate_val']['fails'])}), không gắn alias"))


if __name__ == "__main__":
    main()
