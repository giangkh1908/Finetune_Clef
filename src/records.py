"""Biến một câu MASSIVE thành đúng System One request mà Clef nhận (state + questions + labels).

Dữ liệu trong data/ chỉ giữ câu + nhãn (gọn, md5 không đổi khi sửa chữ trong schema). Request đầy đủ được dựng ở đây,
dùng chung cho sanity / train / eval / serving, nên lúc train và lúc chạy thật model nhìn thấy cùng một thứ.
"""

from functools import lru_cache

import yaml

from src.common import SCHEMA_DIR

SCHEMA_FILE = SCHEMA_DIR / "massive_intent.yaml"
PARAPHRASES_FILE = SCHEMA_DIR / "paraphrases.yaml"


@lru_cache
def load_schema() -> dict:
    with open(SCHEMA_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def question_id() -> str:
    return load_schema()["question_id"]


def intents() -> list[str]:
    return sorted(load_schema()["intents"])


def questions() -> dict:
    s = load_schema()
    return {s["question_id"]: {"type": "choice", "instructions": s["instructions"], "criteria": dict(s["intents"])}}


def to_record(row: dict, with_label: bool = True) -> dict:
    rec = {"id": row["id"], "state": row["text"], "questions": questions()}
    if with_label:
        rec["labels"] = {question_id(): row["intent"]}
    return rec


def scenario_of(intent: str) -> str:
    """Trong MASSIVE, scenario = phần trước dấu '_' đầu tiên của intent (check_data kiểm tra điều này)."""
    return intent.split("_", 1)[0]
