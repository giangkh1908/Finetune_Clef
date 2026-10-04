"""Dựng cấu trúc kiểu tables.json của Spider từ một file SQLite bất kỳ (cho serving)."""

import sqlite3


def table_info_from_sqlite(db_path: str) -> dict:
    """Dựng cấu trúc giống tables.json của Spider trực tiếp từ PRAGMA."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    cols, types, pks, fk_raw = [[-1, "*"]], ["text"], [], []
    for t_idx, t in enumerate(tables):
        for _, name, typ, _, _, pk in conn.execute(f'PRAGMA table_info("{t}")'):
            if pk:
                pks.append(len(cols))
            cols.append([t_idx, name])
            types.append(typ or "text")
        for row in conn.execute(f'PRAGMA foreign_key_list("{t}")'):
            fk_raw.append((t, row[3], row[2], row[4]))
    conn.close()
    lookup = {(tables[t].lower(), c.lower()): i for i, (t, c) in enumerate(cols) if t >= 0}
    fks = [[lookup[(st.lower(), sc.lower())], lookup[(dt.lower(), (dc or sc).lower())]]
           for st, sc, dt, dc in fk_raw
           if (st.lower(), sc.lower()) in lookup and (dt.lower(), (dc or sc).lower()) in lookup]
    return {"table_names_original": tables, "column_names_original": cols,
            "column_types": types, "primary_keys": pks, "foreign_keys": fks}

