import sqlite3
from datetime import datetime


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def today_str():
    return datetime.now().strftime("%Y-%m-%d")


def add_log(db_path: str, level: str, message: str):
    with sqlite3.connect(db_path) as con:
        con.execute("INSERT INTO text_logs(ts, level, message) VALUES(?,?,?)", (now_iso(), level, message))


def add_alert(db_path: str, camera: str, alert_type: str, message: str,
              evidence_path: str = None, name: str = None, sim: float = None, face_prob: float = None):
    with sqlite3.connect(db_path) as con:
        con.execute("""
        INSERT INTO alerts(ts, camera, alert_type, message, name, sim, face_prob, evidence_path)
        VALUES(?,?,?,?,?,?,?,?)
        """, (now_iso(), camera, alert_type, message, name, sim, face_prob, evidence_path))


def update_alert_status(db_path: str, alert_id: int, status: str, false_alert: int):
    with sqlite3.connect(db_path) as con:
        con.execute("UPDATE alerts SET status=?, false_alert=? WHERE id=?", (status, false_alert, alert_id))


def add_entry_decision(
    con: sqlite3.Connection,
    alert_id: int,
    decision: str,
    name: str = None,
    contact: str = None,
    address: str = None,
    reason: str = None,
    notes: str = None,
    image_path: str = None,
):
    cur = con.execute(
        """
        INSERT INTO entry_decisions(
            alert_id, decision, name, contact, address, reason, notes, image_path, created_at
        )
        VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (alert_id, decision, name, contact, address, reason, notes, image_path, now_iso()),
    )
    return cur.lastrowid


def get_open_alerts(db_path: str, limit: int = 20):
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT * FROM alerts
            WHERE status='open'
            ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_recent_logs(db_path: str, limit: int = 200):
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM text_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def add_event(db_path: str, camera: str, event_type: str, name=None, sim=None, face_prob=None, evidence_path=None, message=None):
    ts = now_iso()
    date = today_str()
    with sqlite3.connect(db_path) as con:
        con.execute("""
        INSERT INTO events(ts, date, camera, event_type, name, sim, face_prob, evidence_path, message)
        VALUES(?,?,?,?,?,?,?,?,?)
        """, (ts, date, camera, event_type, name, sim, face_prob, evidence_path, message))


def set_inside_state(db_path: str, name: str, inside: int, camera: str):
    ts = now_iso()
    date = today_str()
    entry_ts = ts if camera == "entry" else None
    exit_ts = ts if camera == "exit" else None

    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT inside, last_entry_ts, last_exit_ts FROM attendance_state WHERE date=? AND name=?",
            (date, name)
        ).fetchone()

        if row is None:
            con.execute("""
            INSERT INTO attendance_state(date,name,inside,last_entry_ts,last_exit_ts,last_seen_camera,last_update_ts)
            VALUES(?,?,?,?,?,?,?)
            """, (date, name, int(inside), entry_ts, exit_ts, camera, ts))
        else:
            old_inside, old_entry, old_exit = row
            new_entry = entry_ts or old_entry
            new_exit = exit_ts or old_exit
            con.execute("""
            UPDATE attendance_state
            SET inside=?, last_entry_ts=?, last_exit_ts=?, last_seen_camera=?, last_update_ts=?
            WHERE date=? AND name=?
            """, (int(inside), new_entry, new_exit, camera, ts, date, name))
