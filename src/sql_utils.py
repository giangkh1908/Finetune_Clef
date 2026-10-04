"""Tiện ích SQLite: dựng schema cho prompt, chạy SQL có timeout, so sánh kết quả, độ khó kiểu Spider.
Chỉ dùng thư viện chuẩn để chạy được cả trên máy không có GPU."""

import os
import sqlite3
import time
from collections import Counter

# ---------------------------------------------------------------- schema


def _quote(name: str) -> str:
    return f'"{name}"' if not name.replace("_", "").isalnum() else name


def _fmt_value(v, max_len: int = 40) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    s = str(v).replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def build_schema(table_info: dict, db_path: str, sample_rows: int = 3) -> str:
    """CREATE TABLE với tên gốc tiếng Anh, khoá chính/khoá ngoại theo tables.json của Spider,
    kèm vài dòng dữ liệu mẫu để model biết định dạng giá trị."""
    tables = table_info["table_names_original"]
    cols = table_info["column_names_original"]
    pks = set(i for p in table_info["primary_keys"] for i in (p if isinstance(p, list) else [p]))
    fks = table_info["foreign_keys"]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    parts = []
    for t_idx, t_name in enumerate(tables):
        declared = {}
        try:
            for _, name, typ, *_ in conn.execute(f"PRAGMA table_info({_quote(t_name)})"):
                declared[name.lower()] = (typ or "").upper()
        except sqlite3.Error:
            pass
        lines = []
        col_ids = [i for i, (t, _) in enumerate(cols) if t == t_idx]
        for i in col_ids:
            c_name = cols[i][1]
            typ = declared.get(c_name.lower()) or table_info["column_types"][i].upper()
            lines.append(f"  {_quote(c_name)} {typ}".rstrip())
        tbl_pks = [cols[i][1] for i in col_ids if i in pks]
        if tbl_pks:
            lines.append(f"  PRIMARY KEY ({', '.join(_quote(c) for c in tbl_pks)})")
        for src, dst in fks:
            if cols[src][0] == t_idx:
                ref_t = tables[cols[dst][0]]
                lines.append(
                    f"  FOREIGN KEY ({_quote(cols[src][1])}) REFERENCES {_quote(ref_t)}({_quote(cols[dst][1])})"
                )
        ddl = f"CREATE TABLE {_quote(t_name)} (\n" + ",\n".join(lines) + "\n);"

        if sample_rows > 0:
            try:
                cur = conn.execute(f"SELECT * FROM {_quote(t_name)} LIMIT {sample_rows}")
                header = [d[0] for d in cur.description]
                rows = cur.fetchall()
                if rows:
                    sample = "\n".join(" | ".join(_fmt_value(v) for v in r) for r in rows)
                    ddl += f"\n/* {len(rows)} example rows:\n{' | '.join(header)}\n{sample}\n*/"
            except sqlite3.Error:
                pass
        parts.append(ddl)
    conn.close()
    return "\n\n".join(parts)


# ---------------------------------------------------------------- execution


def normalize_sql(sql: str) -> str:
    """Gộp khoảng trắng thừa (ngoài chuỗi trong nháy) và bỏ ; cuối."""
    out, quote, prev_space = [], None, False
    for ch in sql.strip():
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        if ch.isspace():
            if not prev_space:
                out.append(" ")
            prev_space = True
            continue
        prev_space = False
        out.append(ch)
    return "".join(out).strip().rstrip(";").strip()


def execute(db_path: str, sql: str, timeout: float = 30.0):
    """Trả về (ok, rows hoặc thông báo lỗi). DB mở chế độ chỉ đọc."""
    if not os.path.exists(db_path):
        return False, f"missing db {db_path}"
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        return True, conn.execute(sql).fetchall()
    except Exception as e:  # noqa: BLE001 - mọi lỗi của SQL sinh ra đều tính là sai
        return False, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


def _norm_row(row):
    return tuple(round(v, 6) if isinstance(v, float) else v for v in row)


def results_match(gold_rows, pred_rows, ordered: bool) -> bool:
    gold = [_norm_row(r) for r in gold_rows]
    pred = [_norm_row(r) for r in pred_rows]
    if ordered:
        return gold == pred
    return Counter(gold) == Counter(pred)


def needs_order(sql: str) -> bool:
    return "order by" in sql.lower()


# ---------------------------------------------------------------- hardness (port từ evaluation.py của Spider)

_LIKE_OP = 9  # WHERE_OPS.index('like')


def _cond_units(conds):
    return [c for c in conds[::2] if isinstance(c, list)]


def _nested(sql):
    out = []
    for cu in _cond_units(sql["from"]["conds"]) + _cond_units(sql["where"]) + _cond_units(sql["having"]):
        if isinstance(cu[3], dict):
            out.append(cu[3])
        if isinstance(cu[4], dict):
            out.append(cu[4])
    for k in ("intersect", "except", "union"):
        if sql[k] is not None:
            out.append(sql[k])
    return out


def _count_agg(units):
    return sum(1 for u in units if isinstance(u, list) and u and u[0] != 0)


def _component1(sql):
    n = sum(bool(sql[k]) for k in ("where", "groupBy", "orderBy"))
    n += sql["limit"] is not None
    n += max(len(sql["from"]["table_units"]) - 1, 0)
    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    n += sum(1 for t in ao if t == "or")
    cus = _cond_units(sql["from"]["conds"]) + _cond_units(sql["where"]) + _cond_units(sql["having"])
    n += sum(1 for cu in cus if cu[1] == _LIKE_OP)
    return n


def _others(sql):
    agg = _count_agg(sql["select"][1]) + _count_agg(sql["where"][::2]) + _count_agg(sql["groupBy"])
    if sql["orderBy"]:
        units = [u[1] for u in sql["orderBy"][1] if u[1]] + [u[2] for u in sql["orderBy"][1] if u[2]]
        agg += _count_agg(units)
    agg += _count_agg(sql["having"])
    n = int(agg > 1)
    n += len(sql["select"][1]) > 1
    n += len(sql["where"]) > 1
    n += len(sql["groupBy"]) > 1
    return n


def hardness(sql: dict) -> str:
    c1, c2, o = _component1(sql), len(_nested(sql)), _others(sql)
    if c1 <= 1 and o == 0 and c2 == 0:
        return "easy"
    if (o <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and o < 2 and c2 == 0):
        return "medium"
    if (o > 2 and c1 <= 2 and c2 == 0) or (2 < c1 <= 3 and o <= 2 and c2 == 0) or (c1 <= 1 and o == 0 and c2 <= 1):
        return "hard"
    return "extra"
