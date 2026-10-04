"""Stage `register`: đóng gói checkpoint đã chọn (LoRA adapter + joint head, vài trăm MB) + schema + ngưỡng thành
MLflow pyfunc, đăng ký vào Model Registry với alias `candidate`. Chỉ golden_eval mới được nâng lên `champion`.

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
              "threshold": sel["threshold"]["threshold"], "target_precision": sel["target_precision"]}
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
    client.set_registered_model_alias(name, "candidate", version)
    vtags = {
        "val_acc": f"{sel['val_acc']:.4f}",
        "val_ece": f"{sel['val_metrics']['val_ece']:.4f}",
        "val_precision_at_thr": f"{sel['threshold']['precision']:.4f}",
        "val_coverage_at_thr": f"{sel['threshold']['coverage']:.4f}",
        "threshold": f"{sel['threshold']['threshold']:.4f}",
        "stability_score": f"{sel['stability']['score']:.4f}",
        "diagnosis": sel["diagnosis"],
        "base_model": t["base_model"], "base_revision": t["base_revision"],
        "train_run": sel["run"], "checkpoint": sel["checkpoint"],
        **{k: v for k, v in tags.items() if k.startswith(("data.", "schema.", "git_sha", "dvc_lock"))},
    }
    for k, v in vtags.items():
        client.set_model_version_tag(name, version, k, v)
    write_json(REPORTS / "registered.json", {"name": name, "version": version, "alias": "candidate",
                                             "model_uri": f"models:/{name}/{version}"})
    print(f"Đăng ký {name} v{version} (alias candidate)")


if __name__ == "__main__":
    main()
