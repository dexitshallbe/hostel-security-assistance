import os
import sqlite3


def init_db(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Alerts shown in Streamlit
    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        camera TEXT NOT NULL,                 -- entry/exit
        alert_type TEXT NOT NULL,             -- UNKNOWN_PERSON/TAILGATING/FACE_OCCLUSION/etc
        message TEXT,
        name TEXT,
        sim REAL,
        face_prob REAL,
        evidence_path TEXT,
        status TEXT NOT NULL DEFAULT 'open',  -- open/dealt/ignored
        false_alert INTEGER NOT NULL DEFAULT 0
    )
    """)

    # Text logs shown in Streamlit
    cur.execute("""
    CREATE TABLE IF NOT EXISTS text_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        level TEXT NOT NULL,
        message TEXT NOT NULL
    )
    """)

    # Attendance state updated in-place per day
    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance_state (
        date TEXT NOT NULL,
        name TEXT NOT NULL,
        inside INTEGER NOT NULL,          -- 1 inside, 0 outside
        last_entry_ts TEXT,
        last_exit_ts TEXT,
        last_seen_camera TEXT,
        last_update_ts TEXT,
        PRIMARY KEY (date, name)
    )
    """)

    # Guest access expiration
    cur.execute("""
    CREATE TABLE IF NOT EXISTS guest_access (
        guest_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_ts TEXT NOT NULL,
        expires_ts TEXT NOT NULL,
        folder_path TEXT NOT NULL
    )
    """)

    # Optional: keep a history of recognition events for audits
    cur.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        date TEXT NOT NULL,
        camera TEXT NOT NULL,
        event_type TEXT NOT NULL,     -- known_entry/known_exit/unknown/mismatch/...
        name TEXT,
        sim REAL,
        face_prob REAL,
        evidence_path TEXT,
        message TEXT
    )
    """)

    # Operator entry decisions linked to alerts
    cur.execute("""
    CREATE TABLE IF NOT EXISTS entry_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id INTEGER NOT NULL,
        decision TEXT NOT NULL,                 -- ENTRY_GRANTED / ENTRY_DENIED
        name TEXT,
        contact TEXT,
        address TEXT,
        reason TEXT,
        notes TEXT,
        image_path TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(alert_id) REFERENCES alerts(id)
    )
    """)

    con.commit()
    con.close()
