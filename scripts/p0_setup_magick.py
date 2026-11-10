#!/usr/bin/env python3
"""把LIBERO-plus所需ImageMagick库解包到项目目录，不更改系统安装。

@author pony
@date 2026-11-06
@version v1.1.0
@last_modified 2026-11-06
@changelog
    - v1.1.0 (2026-11-06): 给Wand补项目内开发名软链接，兼容多架构库路径。
    - v1.0.0 (2026-11-06): 从系统已配置Ubuntu源下载缺失依赖并保留包版本与哈希。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess

from p0_common import ROOT, run_stage


def install(run_dir: pathlib.Path) -> dict:
    packages = ["libmagickwand-6.q16-7t64", "libmagickcore-6.q16-7t64", "imagemagick-6-common"]
    listing = subprocess.check_output(
        ["apt-cache", "depends", "--recurse", "--no-recommends", "--no-suggests", "--no-conflicts",
         "--no-breaks", "--no-replaces", "--no-enhances", *packages], text=True, timeout=60)
    names = sorted({line for line in listing.splitlines() if re.fullmatch(r"[a-z0-9][a-z0-9+.-]*(?::arm64)?", line)})
    missing = []
    for name in names:
        status = subprocess.run(["dpkg-query", "-W", "-f=${Status}", name], capture_output=True, text=True)
        if status.returncode != 0 or "install ok installed" not in status.stdout:
            missing.append(name)
    target = ROOT / "tools/imagemagick"
    archives = target / "packages"
    archives.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"local_root": str(target), "missing_packages": missing}), flush=True)
    for name in missing:
        subprocess.run(["apt-get", "download", name], cwd=archives, check=True, timeout=180)
    records = []
    for path in sorted(archives.glob("*.deb")):
        subprocess.run(["dpkg-deb", "-x", str(path), str(target)], check=True, timeout=60)
        fields = subprocess.check_output(["dpkg-deb", "-f", str(path), "Package", "Version", "Architecture"], text=True)
        records.append({"file": path.name, "metadata": fields.strip(),
                        "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()})
    library = target / "usr/lib/aarch64-linux-gnu/libMagickWand-6.Q16.so.7"
    if not library.is_file():
        raise RuntimeError("本地解包未生成预期的MagickWand库")
    for name in ["MagickWand", "MagickCore"]:
        link = target / f"usr/lib/lib{name}-6.Q16.so"
        if not link.exists():
            link.symlink_to(f"aarch64-linux-gnu/lib{name}-6.Q16.so.7")
    result = {"system_install_modified": False, "source": "本机已配置的Ubuntu apt源", "packages": records,
              "root": str(target), "library": str(library)}
    (run_dir / "packages.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(run_stage("magick_setup", install))
