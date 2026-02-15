# streamlit_app.py
# LAN-only Streamlit dashboard with:
# - bcrypt password hashing
# - brute-force lockout
# - session-based "authed" flag (per browser session)
# - real-time alerts + logs + evidence images
# - ignore/dealt actions (ignore => false_alert=1)
# - temporary guest upload with expiry (hours) + worker reload flag

import os
import uuid
import sqlite3
from datetime import datetime, timedelta

import bcrypt
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.config import Config
from src.db.schema import init_db
from src.db.queries import (
    get_open_alerts,
    get_recent_logs,
    update_alert_status,
    add_log,
)

# ---------------------------
# Paths / config
# ---------------------------
CFG = Config()
DB_PATH = CFG.log_db_path
GUESTS_ROOT = CFG.guests_dir
RELOAD_FLAG = CFG.guests_reload_flag

# ---------------------------
# LAN-only demo auth config
# ---------------------------
DEMO_USER = "admin"

# Generate once, then paste here:
# python -c "import bcrypt; print(bcrypt.hashpw(b'demo123', bcrypt.gensalt()).decode())"
# NOTE: must be bytes: b"..."
DEMO_PASS_HASH = b"$2b$12$4hpLku0pTKn8lgXXFSmMa.qPL59Py8RH.PwsenY9x/OXFNMamS/36"

MAX_FAILS = 5
LOCKOUT_MINUTES = 10

# Auto refresh dashboard every N ms
AUTO_REFRESH_MS = 2000


def touch_reload_flag():
    os.makedirs(GUESTS_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(RELOAD_FLAG), exist_ok=True)
    with open(RELOAD_FLAG, "a") as f:
        f.write("")
    os.utime(RELOAD_FLAG, None)


def _init_session():
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if "fail_count" not in st.session_state:
        st.session_state.fail_count = 0
    if "locked_until" not in st.session_state:
        st.session_state.locked_until = None


def _is_locked() -> bool:
    locked_until = st.session_state.locked_until
    if locked_until is None:
        return False
    return datetime.now() < locked_until


def _lock_remaining_minutes() -> int:
    locked_until = st.session_state.locked_until
    if locked_until is None:
        return 0
    seconds = (locked_until - datetime.now()).total_seconds()
    return max(0, int(seconds // 60) + 1)


def check_login() -> bool:
    _init_session()

    if st.session_state.authed:
        return True

    st.title("Hostel Security Login")

    if _is_locked():
        st.error(f"Too many attempts. Try again in ~{_lock_remaining_minutes()} minute(s).")
        return False

    u = st.text_input("User ID", autocomplete="username")
    p = st.text_input("Password", type="password", autocomplete="current-password")

    # Safety hint if user forgot to replace the hash
    if b"REPLACE_THIS_WITH_GENERATED_HASH" in DEMO_PASS_HASH:
        st.warning(
            "You haven't set a real password hash yet.\n\n"
            "Run:\n"
            "`python -c \"import bcrypt; print(bcrypt.hashpw(b'demo123', bcrypt.gensalt()).decode())\"`\n"
            "Then paste it into `DEMO_PASS_HASH` in `streamlit_app.py`."
        )

    if st.button("Login"):
        ok_user = (u == DEMO_USER)
        ok_pass = False
        if ok_user:
            try:
                ok_pass = bcrypt.checkpw(p.encode("utf-8"), DEMO_PASS_HASH)
            except Exception:
                ok_pass = False

        if ok_user and ok_pass:
            st.session_state.authed = True
            st.session_state.fail_count = 0
            st.session_state.locked_until = None
            add_log(DB_PATH, "INFO", "Dashboard login successful")
            st.rerun()
        else:
            st.session_state.fail_count += 1
            remaining = MAX_FAILS - st.session_state.fail_count
            if remaining <= 0:
                st.session_state.locked_until = datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)
                add_log(DB_PATH, "WARN", "Login locked due to too many failed attempts")
                st.error(f"Locked for {LOCKOUT_MINUTES} minutes due to too many attempts.")
            else:
                add_log(DB_PATH, "WARN", f"Login failed. Attempts left: {remaining}")
                st.error(f"Invalid credentials. Attempts left: {remaining}")

    return False


def render_sidebar():
    with st.sidebar:
        st.write("✅ Logged in")
        st.caption("LAN-only mode (bcrypt + lockout)")

        # Optional: show where DB lives
        st.code(DB_PATH)

        if st.button("Logout"):
            st.session_state.authed = False
            add_log(DB_PATH, "INFO", "Dashboard logout")
            st.rerun()


def render_alerts():
    st.subheader("Real-time Alerts (Open)")
    alerts = get_open_alerts(DB_PATH, limit=30)

    if not alerts:
        st.info("No open alerts right now.")
        return

    for a in alerts:
        with st.container(border=True):
            st.write(f"**[{a['alert_type']}]** camera=`{a['camera']}`  time=`{a['ts']}`")

            meta_cols = st.columns(3)
            with meta_cols[0]:
                if a.get("name"):
                    st.write(f"name: `{a['name']}`")
            with meta_cols[1]:
                if a.get("sim") is not None:
                    try:
                        st.write(f"sim: `{float(a['sim']):.3f}`")
                    except Exception:
                        st.write(f"sim: `{a['sim']}`")
            with meta_cols[2]:
                if a.get("face_prob") is not None:
                    try:
                        st.write(f"face_prob: `{float(a['face_prob']):.2f}`")
                    except Exception:
                        st.write(f"face_prob: `{a['face_prob']}`")

            if a.get("message"):
                st.write(a["message"])

            # Evidence image
            ep = a.get("evidence_path")
            if ep and os.path.exists(ep):
                st.image(ep, caption="Evidence", use_container_width=True)
            elif ep:
                st.caption("Evidence path stored, but file not found on disk.")

            c1, c2 = st.columns(2)

            # Ignore => keep logs + evidence but tag false_alert=1
            if c1.button("Ignore (False alert)", key=f"ign_{a['id']}"):
                update_alert_status(DB_PATH, int(a["id"]), status="ignored", false_alert=1)
                add_log(DB_PATH, "INFO", f"Alert {a['id']} ignored (false_alert=1)")
                st.rerun()

            # Dealt => resolved
            if c2.button("Dealt", key=f"deal_{a['id']}"):
                update_alert_status(DB_PATH, int(a["id"]), status="dealt", false_alert=0)
                add_log(DB_PATH, "INFO", f"Alert {a['id']} marked dealt")
                st.rerun()


def render_logs():
    st.subheader("Live Text Logs")
    logs = get_recent_logs(DB_PATH, limit=200)

    if not logs:
        st.info("No logs yet.")
        return

    # Show newest first
    for l in logs[:80]:
        ts = l.get("ts", "")
        level = l.get("level", "")
        msg = l.get("message", "")

        if level == "ALERT":
            st.error(f"{ts} [{level}] {msg}")
        elif level == "WARN":
            st.warning(f"{ts} [{level}] {msg}")
        else:
            st.write(f"{ts} [{level}] {msg}")


import subprocess
import shutil

def render_guest_upload():
    st.subheader("Temporary Guest Access (Video)")
    st.caption(
        "Upload a short face-pose video for a temporary guest. "
        "We’ll extract face crops automatically and grant access for the chosen duration."
    )

    with st.form("guest_form"):
        guest_name = st.text_input("Guest name")
        hours = st.number_input("Access duration (hours)", min_value=1, max_value=168, value=24)
        video = st.file_uploader(
            "Upload pose video (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False
        )
        submitted = st.form_submit_button("Add guest")

    if not submitted:
        return

    if not guest_name:
        st.error("Guest name is required.")
        return
    if video is None:
        st.error("Upload a video file.")
        return

    # Basic name sanitization to avoid weird folder names
    safe_name = "".join(c for c in guest_name.strip() if c.isalnum() or c in ("_", "-", " ")).strip().replace(" ", "_")
    if not safe_name:
        st.error("Guest name contains no usable characters.")
        return

    gid = str(uuid.uuid4())[:8]
    folder_name = f"{safe_name}__{gid}"

    # Final destination (faces)
    final_folder = os.path.join(GUESTS_ROOT, folder_name)
    os.makedirs(final_folder, exist_ok=True)

    # Temp folder for video
    tmp_root = os.path.join(GUESTS_ROOT, "_tmp")
    tmp_folder = os.path.join(tmp_root, folder_name)
    os.makedirs(tmp_folder, exist_ok=True)

    video_path = os.path.join(tmp_folder, f"pose_video_{gid}.mp4")
    with open(video_path, "wb") as out:
        out.write(video.getbuffer())

    # Run extraction (call your script)
    # If your extractor is in same repo: extract_face_frames.py
    # Tune args as you like
    cmd = [
        "python", "extract_face_frames.py",
        video_path,
        "--out", final_folder,
        "--every", "3",
        "--max", "60",
        "--min_prob", "0.80",
        "--min_dist", "0.50",
        "--blur", "30",
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        add_log(DB_PATH, "INFO", f"Guest video processed: {guest_name} id={gid}. extractor_ok.")
    except subprocess.CalledProcessError as e:
        # Cleanup partial folder if extraction failed
        shutil.rmtree(final_folder, ignore_errors=True)
        add_log(DB_PATH, "ERROR", f"Guest video processing failed: {guest_name} id={gid}. {e.stderr[:300]}")
        st.error("Video processing failed. Check logs.")
        st.code(e.stderr or e.stdout)
        return
    finally:
        # Always delete temp video folder
        shutil.rmtree(tmp_folder, ignore_errors=True)

    # Count how many face images were produced
    produced = [p for p in os.listdir(final_folder) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(produced) == 0:
        shutil.rmtree(final_folder, ignore_errors=True)
        add_log(DB_PATH, "WARN", f"No faces extracted for guest: {guest_name} id={gid}")
        st.error("No faces could be extracted from the video. Try a clearer/closer video.")
        return

    created = datetime.now()
    expires = created + timedelta(hours=int(hours))

    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        INSERT INTO guest_access(guest_id, name, created_ts, expires_ts, folder_path)
        VALUES(?,?,?,?,?)
        """, (
            gid,
            guest_name,
            created.isoformat(timespec="seconds"),
            expires.isoformat(timespec="seconds"),
            final_folder
        ))

    touch_reload_flag()
    add_log(DB_PATH, "INFO", f"Guest added: {guest_name} id={gid} faces={len(produced)} expires={expires.isoformat(timespec='seconds')}")

    st.success(f"Guest '{guest_name}' added for {hours} hour(s). Faces extracted: {len(produced)} (id={gid})")
    st.rerun()



def main():
    # Ensure DB exists
    init_db(DB_PATH)

    st.set_page_config(page_title="Hostel Security", layout="wide")

    # Login gate
    if not check_login():
        return

    render_sidebar()

    # Auto refresh dashboard
    st_autorefresh(interval=AUTO_REFRESH_MS, key="dash_refresh")

    st.title("Hostel Security Dashboard")
    st.caption("No live camera feed shown. Alerts + logs are updated in real time.")

    colA, colB = st.columns([2.2, 1.0])

    with colA:
        render_alerts()

    with colB:
        render_logs()

    st.divider()
    render_guest_upload()


if __name__ == "__main__":
    main()
