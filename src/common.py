"""Đường dẫn, đọc params, I/O jsonl, lineage dữ liệu và cấu hình MLflow dùng chung."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
SCHEMA_DIR = ROOT / "schema"

# golden nằm riêng thư mục; chỉ golden_eval.py được phép đọc.
SPLIT_FILES = {
    "train": DATA / "train.jsonl",
    "train_probe": DATA / "train_probe.jsonl",
    "val": DATA / "val.jsonl",
}
GOLDEN_FILE = DATA / "golden" / "golden.jsonl"


def params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_split(split: str) -> list[dict]:
    if split not in SPLIT_FILES:
        raise ValueError(f"split '{split}' không hợp lệ (golden chỉ đọc qua golden_eval.py)")
    return read_jsonl(SPLIT_FILES[split])


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def lineage_tags() -> dict:
    """Gắn vào mọi MLflow run: biết chính xác run được train/chấm trên byte dữ liệu nào, schema nào, code version nào."""
    tags = {"git_sha": git_sha()}
    for name, path in {**SPLIT_FILES, "golden": GOLDEN_FILE}.items():
        if Path(path).exists():
            tags[f"data.{name}.md5"] = file_md5(path)
    for path in sorted(SCHEMA_DIR.glob("*.yaml")):
        tags[f"schema.{path.stem}.md5"] = file_md5(path)
    lock = ROOT / "dvc.lock"
    if lock.exists():
        tags["dvc_lock.md5"] = file_md5(lock)
    return tags


def setup_mlflow(experiment: str):
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or params()["project"]["mlflow_tracking_uri"]
    if uri.startswith("sqlite:///") and not os.path.isabs(uri[len("sqlite:///"):]):
        uri = f"sqlite:///{(ROOT / uri[len('sqlite:///'):]).as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return mlflow
