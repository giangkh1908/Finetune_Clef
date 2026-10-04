"""Stage `fetch`: tải dữ liệu nguồn ở version cố định. DVC băm toàn bộ byte đầu ra vào dvc.lock."""

import shutil
import subprocess
import zipfile

from src.common import RAW, params, write_json

KEEP_SPIDER = ["database", "tables.json", "train_spider.json", "train_others.json", "dev.json", "README.txt"]


def fetch_vitext2sql(repo: str, commit: str):
    dst = RAW / "ViText2SQL"
    shutil.rmtree(dst, ignore_errors=True)
    subprocess.run(["git", "clone", "--quiet", repo, str(dst)], check=True)
    subprocess.run(["git", "-C", str(dst), "checkout", "--quiet", commit], check=True)

    def _force_remove(func, path, _):  # object của git là read-only trên Windows
        import os
        os.chmod(path, 0o666)
        func(path)

    shutil.rmtree(dst / ".git", onerror=_force_remove)


def fetch_spider(gdrive_id: str):
    import gdown

    zip_path = RAW / "spider_data.zip"
    gdown.download(id=gdrive_id, output=str(zip_path), quiet=True)
    tmp = RAW / "_spider_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    src = tmp / "spider_data"
    dst = RAW / "spider"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir()
    for name in KEEP_SPIDER:  # bỏ test_database (Spider test) để giảm dung lượng, pipeline không dùng
        shutil.move(str(src / name), str(dst / name))
    shutil.rmtree(tmp)
    zip_path.unlink()


def main():
    p = params()["fetch"]
    RAW.mkdir(exist_ok=True)
    fetch_vitext2sql(p["vitext2sql_repo"], p["vitext2sql_commit"])
    fetch_spider(p["spider_gdrive_id"])
    write_json(RAW / "SOURCES.json", p)
    print("fetch xong:", sorted(x.name for x in RAW.iterdir()))


if __name__ == "__main__":
    main()
