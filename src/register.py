"""Stage `register`: đóng gói checkpoint đã chọn + ngưỡng thành pyfunc, đăng ký vào MLflow Model Registry
với alias `candidate`. Chỉ golden_eval mới được nâng lên `champion`."""

import json

from src.common import OUTPUTS, REPORTS, ROOT, lineage_tags, params, setup_mlflow, write_json


def export_bf16(src, dst):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).save_pretrained(dst)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)


def main():
    P = params()
    name = P["project"]["registered_model"]
    sel = json.loads((REPORTS / "selection.json").read_text(encoding="utf-8"))
    base_model = P["train"]["base_model"]

    export_dir = OUTPUTS / "export" / f"{sel['run']}-{sel['checkpoint']}"
    if not export_dir.exists():
        export_bf16(str(ROOT / sel["model_path"]), str(export_dir))
    config = {"threshold": sel["threshold"]["threshold"], "target_precision": sel["target_precision"],
              "base_model": base_model, **P["generation"]}
    write_json(export_dir.parent / "serving_config.json", config)

    mlflow = setup_mlflow("text2sql-train")
    from mlflow import MlflowClient

    from src.serving import Text2SQLModel

    with mlflow.start_run(run_name=f"register-{sel['run']}-{sel['checkpoint']}"):
        tags = {**lineage_tags(), "stage": "register", "source_run_id": sel["mlflow_run_id"]}
        mlflow.set_tags(tags)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=Text2SQLModel(),
            artifacts={"model_dir": str(export_dir), "config": str(export_dir.parent / "serving_config.json")},
            code_paths=[str(ROOT / "src")],
            pip_requirements=["torch", "transformers>=5.16", "pandas", "mlflow>=3.5", "pyyaml"],
            registered_model_name=name,
        )
    version = info.registered_model_version
    client = MlflowClient()
    client.set_registered_model_alias(name, "candidate", version)
    vtags = {
        "val_ex": f"{sel['val_ex']:.4f}",
        "val_precision_at_thr": f"{sel['threshold']['precision']:.4f}",
        "val_coverage_at_thr": f"{sel['threshold']['coverage']:.4f}",
        "threshold": f"{sel['threshold']['threshold']:.4f}",
        "stability_score": f"{sel['stability']['score']:.4f}",
        "diagnosis": sel["diagnosis"],
        "base_model": base_model,
        "train_run": sel["run"], "checkpoint": sel["checkpoint"],
        **{k: v for k, v in tags.items() if k.startswith(("data.", "git_sha", "dvc_lock"))},
    }
    for k, v in vtags.items():
        client.set_model_version_tag(name, version, k, v)
    write_json(REPORTS / "registered.json", {"name": name, "version": version, "alias": "candidate",
                                             "model_uri": f"models:/{name}/{version}"})
    print(f"Đăng ký {name} v{version} (alias candidate)")


if __name__ == "__main__":
    main()
