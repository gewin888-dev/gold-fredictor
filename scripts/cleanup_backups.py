"""备份文件自动清理：保留最近N个，删除旧文件。"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP_COUNT = 3


def _parse_db_backup_date(filename: str) -> str:
    """从 gold_monitor.backup.YYYYMMDD_HHMMSS.db 提取时间戳"""
    m = re.search(r"(\d{8}_\d{6})", filename)
    return m.group(1) if m else ""


def _parse_code_backup_date(filename: str) -> str:
    """从 xxx.backup.YYYYMMDD_HHMMSS.py 提取时间戳"""
    m = re.search(r"(\d{8}_\d{6})", filename)
    return m.group(1) if m else ""


def cleanup_backups(keep: int = KEEP_COUNT, dry_run: bool = False) -> list[str]:
    """清理旧备份文件，每种类型保留最近 keep 个。返回被删除的文件列表。"""
    deleted: list[str] = []

    # ── DB 备份 ──
    db_pattern = re.compile(r"gold_monitor\.backup\.(\d{8}_\d{6})\.db$")
    db_backups = []
    for f in ROOT.iterdir():
        m = db_pattern.match(f.name)
        if m:
            db_backups.append((m.group(1), f))

    db_backups.sort(key=lambda x: x[0], reverse=True)
    for _, path in db_backups[keep:]:
        if dry_run:
            deleted.append(f"[DRY-RUN] {path}")
        else:
            path.unlink()
            deleted.append(str(path))

    # ── 代码备份（app/ 和 dashboard/ 下） ──
    code_pattern = re.compile(r".*\.backup\.(\d{8}_\d{6})\.py$")
    code_backups = []
    for search_dir in [ROOT / "app", ROOT / "dashboard"]:
        if not search_dir.exists():
            continue
        for f in search_dir.rglob("*.backup.*.py"):
            m = code_pattern.match(f.name)
            if m:
                code_backups.append((m.group(1), f))

    code_backups.sort(key=lambda x: x[0], reverse=True)
    for _, path in code_backups[keep:]:
        if dry_run:
            deleted.append(f"[DRY-RUN] {path}")
        else:
            path.unlink()
            deleted.append(str(path))

    if deleted:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Backup cleanup: removed %d old files", len(deleted))
        for d in deleted:
            logger.debug("  %s", d)

    return deleted


if __name__ == "__main__":
    removed = cleanup_backups(dry_run=False)
    if removed:
        print(f"Removed {len(removed)} old backup(s):")
        for r in removed:
            print(f"  {r}")
    else:
        print("No old backups to remove.")
