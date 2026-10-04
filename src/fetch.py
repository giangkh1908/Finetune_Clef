"""Stage `fetch`: tải MASSIVE 1.1 (Amazon, CC-BY-4.0) ở version cố định, chỉ giữ locale cần dùng.

Repo HF `AmazonScience/massive` dùng loading script (datasets >= 4 không chạy được), nên tải thẳng tarball gốc
trên S3 mà script đó trỏ tới, kiểm tra sha256 rồi mới giải nén. DVC băm byte đầu ra vào dvc.lock.
"""

import hashlib
import shutil
import tarfile
import urllib.request

from src.common import RAW, params, write_json


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = params()["fetch"]
    dst = RAW / "massive"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    tgz = RAW / "_massive.tar.gz"
    urllib.request.urlretrieve(p["massive_url"], tgz)
    got = sha256(tgz)
    if got != p["massive_sha256"]:
        tgz.unlink()
        raise SystemExit(f"sha256 lệch: {got} != {p['massive_sha256']} -> nguồn đã đổi, không dùng.")

    wanted = {f"1.1/data/{p['locale']}.jsonl", "1.1/LICENSE", "1.1/NOTICE.md", "1.1/CITATION.md"}
    with tarfile.open(tgz) as t:
        for m in t.getmembers():
            if m.name in wanted:
                m.name = m.name.split("/")[-1]
                t.extract(m, dst, filter="data")
    tgz.unlink()
    missing = {w.split("/")[-1] for w in wanted} - {x.name for x in dst.iterdir()}
    if missing:
        raise SystemExit(f"tarball thiếu {missing}")
    write_json(RAW / "SOURCES.json", {**p, "license": "CC-BY-4.0"})
    print("fetch xong:", sorted(x.name for x in dst.iterdir()))


if __name__ == "__main__":
    main()
