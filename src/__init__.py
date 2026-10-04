import sys

# Console Windows mặc định cp1252, in tiếng Việt sẽ lỗi.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
