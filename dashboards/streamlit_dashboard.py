"""
dashboards/streamlit_dashboard.py
───────────────────────────────────
Live analytics dashboard powered by Streamlit.

Install: pip install streamlit
Run:     streamlit run dashboards/streamlit_dashboard.py

Features:
  • Auto-refreshes every 10 seconds
  • Detection event timeline chart
  • Per-camera breakdown
  • Recent event table
  • Snapshot gallery (last 20 saved images)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta

# ── Streamlit import (graceful error if not installed) ───────
try:
    import streamlit as st
    from streamlit_autorefresh import st_autorefresh  # optional
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Run: pip install streamlit")
        sys.exit(1)

import pandas as pd

from config.settings import load_config
from services.analytics_service import AnalyticsService

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Platform Dashboard",
    page_icon="📷",
    layout="wide",
)

# ── Auto-refresh every 10 seconds ───────────────────────────
if _HAS_AUTOREFRESH:
    st_autorefresh(interval=10_000, key="dashboard_refresh")
else:
    if st.button("🔄 Refresh"):
        st.rerun()

# ── Load config + analytics ──────────────────────────────────
@st.cache_resource
def get_config():
    return load_config("config/config.yaml")

cfg = get_config()

# ── Sidebar controls ─────────────────────────────────────────
st.sidebar.title("📷 Vision Platform")
st.sidebar.markdown("---")

hours = st.sidebar.slider("Report period (hours)", 1, 168, 24)
camera_names = ["All cameras"] + [c.name for c in cfg.cameras]
selected_cam = st.sidebar.selectbox("Camera", camera_names)
cam_filter = None if selected_cam == "All cameras" else selected_cam

db_path = cfg.database.db_path
if not Path(db_path).exists():
    st.warning("⚠️ No detections database found yet. Run `python app.py` first.")
    st.stop()

svc = AnalyticsService(db_path)
report = svc.get_report(hours=hours, camera=cam_filter)

# ── Header metrics ────────────────────────────────────────────
st.title("Vision Platform — Live Dashboard")
st.caption(f"Updated: {report.generated_at:%Y-%m-%d %H:%M:%S}  |  Period: last {hours}h")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events", report.total_events)
col2.metric("Active Cameras", len(report.cameras))
col3.metric("Top Label", report.top_label or "—")
col4.metric("Period", f"{hours}h")

st.markdown("---")

# ── Hourly timeline chart ─────────────────────────────────────
if report.hourly:
    st.subheader("📈 Detections Over Time")
    rows = [{"Hour": h.hour, "Camera": h.camera, "Label": h.label, "Count": h.count}
            for h in report.hourly]
    df = pd.DataFrame(rows)
    pivot = df.groupby(["Hour", "Label"])["Count"].sum().unstack(fill_value=0)
    st.bar_chart(pivot)
else:
    st.info("No events in the selected period.")

# ── Per-camera breakdown ──────────────────────────────────────
st.subheader("📷 Camera Breakdown")
cam_cols = st.columns(max(len(report.cameras), 1))

for i, cam_summary in enumerate(report.cameras):
    with cam_cols[i % len(cam_cols)]:
        st.markdown(f"**{cam_summary.camera_name}**")
        st.metric("Events", cam_summary.total_events)
        label_df = pd.DataFrame(
            [{"Label": k, "Count": v} for k, v in cam_summary.by_label.items()]
        ).sort_values("Count", ascending=False)
        st.dataframe(label_df, use_container_width=True, hide_index=True)
        if cam_summary.last_seen:
            st.caption(f"Last seen: {cam_summary.last_seen:%H:%M:%S}")

st.markdown("---")

# ── Recent events table ───────────────────────────────────────
st.subheader("🗂️ Recent Detection Events")

import sqlite3
with sqlite3.connect(db_path) as conn:
    conn.row_factory = sqlite3.Row
    limit = 100
    cam_clause = f"AND camera_name = '{cam_filter}'" if cam_filter else ""
    rows = conn.execute(
        f"""SELECT id, camera_name, label, detector, confidence, timestamp
            FROM detections
            WHERE timestamp >= datetime('now', '-{hours} hours') {cam_clause}
            ORDER BY id DESC LIMIT {limit}"""
    ).fetchall()

if rows:
    df_events = pd.DataFrame([dict(r) for r in rows])
    df_events["confidence"] = df_events["confidence"].apply(lambda x: f"{x:.2f}")
    st.dataframe(df_events, use_container_width=True, hide_index=True)
else:
    st.info("No recent events.")

# ── Snapshot gallery ──────────────────────────────────────────
st.markdown("---")
st.subheader("🖼️ Recent Snapshots")

snap_dir = Path(cfg.storage.snapshots_path)
snapshots = sorted(snap_dir.glob("*.jpg"), reverse=True)[:20]

if snapshots:
    cols = st.columns(5)
    for i, snap in enumerate(snapshots):
        with cols[i % 5]:
            st.image(str(snap), caption=snap.name, use_column_width=True)
else:
    st.info("No snapshots saved yet.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption("Vision Platform · Built with OpenCV + SQLite · Auto-refreshes every 10s")
