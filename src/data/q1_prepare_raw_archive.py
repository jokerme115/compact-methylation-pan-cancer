from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


RAR_TOOLS = (
    ("unrar", ["unrar", "x", "-o+", "{archive}", "{out_dir}/"]),
    ("7z", ["7z", "x", "-y", "-o{out_dir}", "{archive}"]),
    ("7zz", ["7zz", "x", "-y", "-o{out_dir}", "{archive}"]),
    ("bsdtar", ["bsdtar", "-xf", "{archive}", "-C", "{out_dir}"]),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or extract the Q1 TCGA raw methylation RAR archive.")
    parser.add_argument("--archive", required=True, help="Path to tcga_all.rar")
    parser.add_argument("--out-dir", required=True, help="Extraction directory")
    parser.add_argument("--extract", action="store_true", help="Extract the archive. Default only inspects tool availability.")
    parser.add_argument("--manifest", default="", help="Optional JSON manifest path")
    return parser.parse_args()


def available_tools() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for name, template in RAR_TOOLS:
        path = shutil.which(name)
        if path:
            out.append({"name": name, "path": path, "template": " ".join(template)})
    return out


def render_command(template: list[str], archive: Path, out_dir: Path) -> list[str]:
    return [part.format(archive=str(archive), out_dir=str(out_dir)) for part in template]


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    out_dir = Path(args.out_dir)
    if not archive.exists():
        raise FileNotFoundError(archive)

    tools = available_tools()
    manifest = {
        "archive": str(archive.resolve()),
        "archive_size_bytes": archive.stat().st_size,
        "out_dir": str(out_dir.resolve()),
        "available_rar_tools": tools,
        "extracted": False,
    }

    if not tools:
        manifest["next_action"] = "Install unrar, 7z/7zz, or bsdtar with RAR5 support before extraction."
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        if args.manifest:
            Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(2)

    if args.extract:
        out_dir.mkdir(parents=True, exist_ok=True)
        tool_name = tools[0]["name"]
        template = next(template for name, template in RAR_TOOLS if name == tool_name)
        cmd = render_command(template, archive, out_dir)
        manifest["command"] = cmd
        subprocess.run(cmd, check=True)
        manifest["extracted"] = True
    else:
        manifest["next_action"] = "Re-run with --extract to unpack the archive."

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
