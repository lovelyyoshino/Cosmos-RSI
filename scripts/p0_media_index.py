#!/usr/bin/env python3
"""整理Thor真实图片/展示视频的可播放索引，并支持备份机逐文件校验。

@author pony
@date 2026-11-07
@version v1.3.1
@last_modified 2026-11-08
@changelog
    - v1.3.1 (2026-11-08): 生成资格由各独立验收报告说明，浏览页不预设所有新样本均拒绝。
    - v1.3.0 (2026-11-07): 精简清单只显示关键图片与视频，历史原始证据继续留存。
    - v1.2.0 (2026-11-07): 在浏览页直接提供已归档初训配对CSV和测量来源，逐文件核对恢复附件。
    - v1.1.0 (2026-11-07): 区分初训进展、种子差异和控制输入，修正窄屏卡片溢出，保留RSI证据边界。
    - v1.0.1 (2026-11-07): 兼容分割校准视频没有notes、仅记录frame_mapping的正式清单。
    - v1.0.0 (2026-11-07): 保留相对路径和SHA256，只收录指定正式视频，备份机只做CPU文件核对。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import html
import json
import os
import pathlib
import platform
import time
import urllib.parse

from p0_common import ROOT, run_stage


def digest(path: pathlib.Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify_inventory(inventory_path: pathlib.Path) -> dict:
    started = time.monotonic()
    inventory = json.loads(inventory_path.read_text())
    failures = []
    for item in inventory["images"] + inventory["videos"]:
        path = ROOT / item["path"]
        if not path.resolve().is_relative_to(ROOT) or not path.is_file() or path.stat().st_size != item["bytes"] \
                or digest(path) != item["sha256"]:
            failures.append(item["path"])
    result = {"verified_at": dt.datetime.now().astimezone().isoformat(), "source_host": inventory["source_host"],
              "local_host": platform.node(), "local_root": str(ROOT), "inventory": str(inventory_path),
              "inventory_sha256": digest(inventory_path), "image_count": inventory["image_count"],
              "video_count": inventory["video_count"], "image_total_bytes": inventory["image_total_bytes"],
              "video_total_bytes": inventory["video_total_bytes"], "sha256_and_size_match": not failures,
              "failures": failures, "wall_s": time.monotonic() - started,
              "compute": "CPU逐文件核对大小和SHA256；无GPU、模型或实验样本计数含义"}
    (inventory_path.parent / "local_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"本地素材校验失败：{failures[:10]}")
    return result


def collect(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    manifest = json.loads(args.video_manifest.read_text())
    clips = manifest["clips"]
    if manifest["clip_count"] != len(clips) or not clips:
        raise RuntimeError("正式视频清单不完整")

    def entry(path):
        return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}

    curated = manifest.get("curated") is True
    previous = json.loads(args.previous_gallery.read_text()) if args.previous_gallery and not curated else {"featured_images": []}
    featured = [str(p.relative_to(ROOT)) for extra in args.extra_materials for p in sorted(extra.glob("*.png"))]
    featured += [str(p.relative_to(ROOT)) for p in getattr(args, "featured_image", [])]
    featured = list(dict.fromkeys(featured + previous["featured_images"]))
    if curated:
        selected = {ROOT / name for name in featured} | {args.video_manifest.parent / clip["poster"] for clip in clips}
        pictures = [entry(path) for path in sorted(selected)]
    else:
        pictures = [entry(p) for p in sorted((ROOT / "runs/p0").rglob("*"))
                    if p.is_file() and not p.is_symlink() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    videos = []
    for clip in clips:
        path = args.video_manifest.parent / clip["file"]
        value = entry(path)
        if value["bytes"] != clip["bytes"] or value["sha256"] != clip["sha256"]:
            raise RuntimeError("展示视频内容偏离正式清单")
        videos.append(value)
    inventory = {"created_at": dt.datetime.now().astimezone().isoformat(), "source_host": platform.node(),
                 "source_root": str(ROOT), "compute": "Thor CPU枚举与SHA256，不运行模型",
                 "scope": "精选图片与展示视频；历史证据保留在原目录" if curated else "全量历史图片与指定清单的展示视频；不包含原始视频、数据集或模型权重",
                 "curated": curated,
                 "image_count": len(pictures), "image_total_bytes": sum(p["bytes"] for p in pictures),
                 "video_count": len(videos), "video_total_bytes": sum(p["bytes"] for p in videos),
                 "video_directory": str(args.video_manifest.parent.relative_to(ROOT)),
                 "images": pictures, "videos": videos}
    (directory / "media_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n")
    links = []

    def link(path):
        path = pathlib.Path(path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.resolve(strict=True).is_relative_to(ROOT):
            raise RuntimeError("素材链接越过项目目录")
        links.append(str(path.relative_to(ROOT)))
        return html.escape(urllib.parse.quote(os.path.relpath(path, directory), safe="/"), quote=True)

    fragments = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
                 "<title>Thor RSI 真实实验素材</title><style>*{box-sizing:border-box}body{font:16px/1.65 system-ui,sans-serif;max-width:1448px;margin:auto;padding:24px;background:#f6f4ee;color:#25383c;overflow-wrap:anywhere}a{color:#176d79}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:18px}article,figure{min-width:0;margin:0;background:#fff;padding:16px;border:1px solid #d5dedb;border-radius:8px}video,img{display:block;width:100%;height:auto}small{color:#52686b}details{margin:10px 0}li{overflow-wrap:anywhere}</style>",
                 "<h1>Thor RSI 真实实验素材</h1>",
                 "<p>视频包括真实仿真、Cosmos输出及诊断。500/5000步对照展示固定初训进展；同一π0的成功/失败展示种子差异。生成与派生样本的训练资格以各自绑定的独立验收为准，尚无Cosmos或RSI收益；PPT最后制作。</p>",
                 f"<p>当前{'精选' if curated else '归档'}：{len(pictures)}个图片文件，{len(videos)}个正式展示视频。文件数不是独立实验样本数；历史证据保留。</p>",
                 f"<p><a href='{link(ROOT / 'docs/materials.md')}'>素材来源说明</a> · <a href='{link(args.video_manifest)}'>视频条件与逐帧来源</a></p>"]
    training = manifest.get("training_progress_evidence")
    if training:
        for item in training["files"]:
            path = args.video_manifest.parent / item["file"]
            if digest(path) != item["sha256"]:
                raise RuntimeError("初训配对附件偏离已归档摘要")
        original = args.video_manifest.parent / training["archived_manifest"]
        if digest(original) != training["archived_manifest_sha256"]:
            raise RuntimeError("初训原始测量来源已经改变")
        fragments.append(f"<p>初训对照附件：<a href='{link(args.video_manifest.parent / 'training_progress_pairs.csv')}'>全部配对数据CSV</a> · "
                         f"<a href='{link(original)}'>原始测量条件与来源JSON</a>。仅固定子集的初训进展，截图已随视频包保留。</p>")
    fragments.append("<h2>视频</h2><div class='grid'>")
    for clip in clips:
        title = html.escape(clip["title"])
        notes = " ".join(clip.get("notes", [clip.get("frame_mapping", "完整条件见逐文件来源清单。")]))
        path = args.video_manifest.parent / clip["file"]
        poster = args.video_manifest.parent / clip["poster"]
        fragments.append(f"<article><h3>{title}</h3><video controls preload='none' poster='{link(poster)}' src='{link(path)}'></video>"
                         f"<p><a href='{link(path)}'>打开MP4</a> · {clip['duration_s']:.2f}秒</p><small>{html.escape(notes)}</small></article>")
    fragments.append("</div><h2>重点图表与截图</h2><div class='grid'>")
    for name in featured:
        path = ROOT / name
        href = link(path)
        fragments.append(f"<figure><a href='{href}'><img loading='lazy' src='{href}' alt='{html.escape(path.name)}'></a>"
                         f"<figcaption>{html.escape(path.name)}<br><small>{html.escape(path.parent.name)}</small></figcaption></figure>")
    fragments.append(f"</div><h2>{'精选文件目录' if curated else '全部图片目录'}</h2><p>使用时以来源清单为准，完整历史证据在原目录保留。</p>")
    groups = defaultdict(list)
    for item in pictures:
        groups[str(pathlib.Path(item["path"]).parent)].append(item)
    for group, values in sorted(groups.items()):
        fragments.append(f"<details><summary>{html.escape(group)} · {len(values)}张</summary><ul>")
        fragments.extend(f"<li><a href='{link(value['path'])}'>{html.escape(pathlib.Path(value['path']).name)}</a> <small>{value['bytes']:,} B</small></li>"
                         for value in values)
        fragments.append("</ul></details>")
    fragments.append("</html>")
    (directory / "gallery.html").write_text("\n".join(fragments) + "\n")
    result = {"inventory": "media_inventory.json", "verification": "local_verification.json",
              "video_manifest": str(args.video_manifest.relative_to(ROOT)), "featured_images": featured,
              "featured_video_count": len(clips), "all_images": len(pictures),
              "local_links_checked": len(links), "missing_links": [],
              "notice": "纯本地浏览；初训进展不代表Cosmos或RSI收益，PPT未制作"}
    (directory / "gallery_sources.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-local", type=pathlib.Path)
    parser.add_argument("--video-manifest", type=pathlib.Path)
    parser.add_argument("--previous-gallery", type=pathlib.Path)
    parser.add_argument("--extra-materials", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--featured-image", type=pathlib.Path, action="append", default=[])
    arguments = parser.parse_args()
    if arguments.verify_local:
        print(json.dumps(verify_inventory(arguments.verify_local.resolve(strict=True)), ensure_ascii=False))
        raise SystemExit(0)
    if not arguments.video_manifest:
        parser.error("导出索引需要正式展示视频清单")
    for key in ["video_manifest", "previous_gallery"]:
        value = getattr(arguments, key)
        if value:
            path = value.resolve(strict=True)
            if not path.is_relative_to(ROOT / "runs/p0"):
                parser.error("索引输入必须位于项目P0目录")
            setattr(arguments, key, path)
    arguments.extra_materials = [path.resolve(strict=True) for path in arguments.extra_materials]
    arguments.featured_image = [path.resolve(strict=True) for path in arguments.featured_image]
    if any(not path.is_relative_to(ROOT / "runs/p0") for path in arguments.extra_materials + arguments.featured_image):
        parser.error("新增素材目录必须位于项目P0目录")
    raise SystemExit(run_stage("media_sync", lambda directory: collect(directory, arguments),
                              {"video_manifest": str(arguments.video_manifest),
                               "extra_materials": [str(p) for p in arguments.extra_materials]}))
