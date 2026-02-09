import os
import sqlite3
from datetime import datetime
from typing import List


def list_unexpired_guest_dirs(db_path: str) -> List[str]:
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as con:
        rows = con.execute("SELECT folder_path FROM guest_access WHERE expires_ts > ?", (now,)).fetchall()
    return [r[0] for r in rows]


def list_identity_folders(known_dir: str, db_path: str) -> List[str]:
    dirs = []
    if os.path.isdir(known_dir):
        dirs.append(known_dir)
    dirs.extend(list_unexpired_guest_dirs(db_path))
    return dirs


def iter_person_folders(root_dir: str):
    # one person per folder
    if not os.path.isdir(root_dir):
        return
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p):
            yield name, p
