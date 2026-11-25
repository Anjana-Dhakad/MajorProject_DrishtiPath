import io

import uuid

import json

import re

import tempfile

import subprocess

from pathlib import Path



import imageio_ffmpeg



import pandas as pd

import streamlit as st

from PIL import Image

from urllib.parse import urlencode

from urllib.request import Request, urlopen

import streamlit.components.v1 as components



from detector import detect_hazards

from video_detector import detect_hazards_in_video

from severity import calculate_severity

from database import (

    initialize_database,

    save_report,

    get_all_reports,

    get_report_statistics,

    update_report_status,

    get_report_status_statistics,

    REPORT_STATUSES,

)

from map_view import create_location_map





# =========================

# INITIALIZATION

# =========================

st.set_page_config(

    page_title="DrishtiPath",

    page_icon="🛣️",

    layout="wide",

)



initialize_database()



def convert_video_for_browser(input_path, output_path):

    """Convert video to H.264 MP4 so Chromium/Streamlit can play it reliably."""

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [ffmpeg_exe, "-y", "-i", str(input_path), "-map", "0:v:0", "-map", "0:a?",

               "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",

               "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k", str(output_path)]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0 or not Path(output_path).exists():

        raise RuntimeError((result.stderr or result.stdout or "FFmpeg conversion failed")[-1200:])

    return str(output_path)



# Keep results across Streamlit reruns

DEFAULT_STATE = {

    "detections": None,

    "annotated_image": None,

    "total_hazards": 0,

    "severity": None,

    "severity_score": 0,

    "image_name": None,

    "latitude": 26.2183,

    "longitude": 78.1828,

    "location_name": "",

    "report_location_name": "",

}

for state_key, default_value in DEFAULT_STATE.items():

    if state_key not in st.session_state:

        st.session_state[state_key] = default_value



def search_locations_geoapify(place, limit=8):
    """Search Indian places using Geoapify's free-tier geocoding endpoint."""
    if not place or not place.strip():
        return []
    try:
        api_key = str(st.secrets.get("GEOAPIFY_API_KEY", "")).strip()
    except Exception:
        api_key = ""
    if not api_key:
        st.warning("Geoapify key not found. Add GEOAPIFY_API_KEY to .streamlit/secrets.toml.")
        return []

    query = place.strip()
    params = {"text": query, "limit": limit, "filter": "countrycode:in", "lang": "en", "apiKey": api_key}
    url = "https://api.geoapify.com/v1/geocode/search?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "DrishtiPath_RoadHazard_Project/1.0"})
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        results, seen = [], set()
        for feature in payload.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {}).get("coordinates", [])
            if len(geometry) < 2:
                continue
            try:
                longitude, latitude = float(geometry[0]), float(geometry[1])
            except (TypeError, ValueError):
                continue
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                continue
            address = str(properties.get("formatted") or properties.get("name") or query)
            key = (round(latitude, 6), round(longitude, 6))
            if key in seen:
                continue
            seen.add(key)
            results.append({"label": address, "latitude": latitude, "longitude": longitude,
                         "address": address, "name": properties.get("name", ""),
                         "place_type": properties.get("result_type", "location"),
                         "score": properties.get("rank", {}).get("confidence", 0)})
            if len(results) >= limit:
                break
        return results
    except Exception as error:
        print("Geoapify location search failed:", repr(error))
        st.warning("Location search couldn't connect. You can still click the map to set the location manually.")
        return []


def reverse_geocode_geoapify(latitude, longitude):
    """Get an address for a selected map point using Geoapify reverse geocoding."""
    try:
        api_key = str(st.secrets.get("GEOAPIFY_API_KEY", "")).strip()
    except Exception:
        api_key = ""
    if not api_key:
        return "Manual map location"
    params = {"lat": float(latitude), "lon": float(longitude), "lang": "en", "apiKey": api_key}
    url = "https://api.geoapify.com/v1/geocode/reverse?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "DrishtiPath_RoadHazard_Project/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        features = payload.get("features", [])
        if features:
            return features[0].get("properties", {}).get("formatted", "Manual map location")
        return "Manual map location"
    except Exception as exc:
        print("Geoapify reverse geocoding failed:", repr(exc))
        return "Manual map location"


def extract_stamped_gps_coordinates(image):

    """Read GPS coordinates printed on GPS Map Camera-style photo stamps using OCR.

    This is a fallback only; EXIF GPS metadata remains the preferred source.

    """

    try:

        import pytesseract

        # OCR the lower portion first, where GPS Camera stamps usually appear.

        width, height = image.size

        crop = image.crop((0, int(height * 0.55), width, height)).convert("RGB")

        text = pytesseract.image_to_string(crop, config="--psm 6")

        # Normalize OCR punctuation and search for latitude/longitude labels.

        cleaned = text.replace("º", "°").replace("\u00b0", "°")

        lat_match = re.search(r"(?:lat(?:itude)?\s*[:=]?\s*)([-+]?\d{1,2}(?:\.\d+)?)", cleaned, re.IGNORECASE)

        lon_match = re.search(r"(?:lon(?:g(?:itude)?)?\s*[:=]?\s*)([-+]?\d{1,3}(?:\.\d+)?)", cleaned, re.IGNORECASE)

        if not lat_match or not lon_match:

            # More permissive fallback for stamps like: 26.26461° 78.136791°

            numbers = re.findall(r"[-+]?\d{1,3}\.\d{4,}", cleaned)

            if len(numbers) >= 2:

                latitude, longitude = float(numbers[0]), float(numbers[1])

            else:

                return None

        else:

            latitude, longitude = float(lat_match.group(1)), float(lon_match.group(1))

        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):

            return None

        return latitude, longitude, text.strip()

    except Exception as exc:

        # OCR is optional; absence of Tesseract should not prevent detection.

        print("Stamped GPS OCR unavailable/failed:", repr(exc))

        return None





def haversine_distance_m(lat1, lon1, lat2, lon2):

    """Return the distance between two GPS coordinates in metres."""

    from math import radians, sin, cos, sqrt, atan2

    earth_radius_m = 6371000

    d_lat = radians(lat2 - lat1)

    d_lon = radians(lon2 - lon1)

    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2

    return earth_radius_m * 2 * atan2(sqrt(a), sqrt(1 - a))



def get_hazard_names(detections):

    """Extract normalized hazard labels from a detection list."""

    names = set()

    for item in detections or []:

        if isinstance(item, dict):

            label = item.get("hazard", item.get("class", "Unknown"))

            names.add(str(label).strip().lower())

    return names



def find_possible_duplicate_reports(current_detections, latitude, longitude, saved_reports, radius_m=100):

    """Find saved reports within radius that share at least one hazard type."""

    current_hazards = get_hazard_names(current_detections)

    possible_duplicates = []

    for old_report in saved_reports:

        try:

            old_lat = float(old_report.get("latitude"))

            old_lon = float(old_report.get("longitude"))

        except (TypeError, ValueError):

            continue

        distance = haversine_distance_m(float(latitude), float(longitude), old_lat, old_lon)

        if distance > radius_m:

            continue

        old_hazards_data = old_report.get("hazard_data", [])

        if isinstance(old_hazards_data, str):

            try:

                old_hazards_data = json.loads(old_hazards_data)

            except (json.JSONDecodeError, TypeError):

                old_hazards_data = []

        if isinstance(old_hazards_data, dict):

            old_hazards_data = old_hazards_data.get("detections", [])

        shared_hazards = current_hazards & get_hazard_names(old_hazards_data)

        if shared_hazards:

            possible_duplicates.append({

                "report": old_report,

                "distance_m": distance,

                "shared_hazards": sorted(shared_hazards),

            })

    return sorted(possible_duplicates, key=lambda item: item["distance_m"])







# =========================

# CUSTOM CSS

# =========================

st.markdown(
    """
    <style>
    :root { --rg-bg:#0b1018; --rg-surface:#131c28; --rg-surface2:#182433; --rg-border:#29394b; --rg-text:#edf4fc; --rg-muted:#9aacc0; --rg-accent:#38bdf8; }
    .stApp { background: radial-gradient(ellipse at 15% 0%, #13283a 0, transparent 32%), var(--rg-bg); color:var(--rg-text); }
    [data-testid="stAppViewContainer"] > .main { background:transparent; }
    [data-testid="stHeader"] { background:rgba(10,15,22,.92); backdrop-filter:blur(10px); }
    [data-testid="stMainBlockContainer"] { padding-top:3.15rem !important; padding-bottom:3rem; max-width:1500px; }
    /* Leave Streamlit native page scrolling in control. Avoid forced heights/overflow on its containers. */
    /* Large selectable cards for image/video detection mode */
    [data-testid="stRadio"]:has(input[name="road_detection_mode_cards"]) { margin:14px 0 24px; }
    [data-testid="stRadio"]:has(input[name="road_detection_mode_cards"]) > div { display:flex !important; flex-wrap:wrap !important; gap:14px !important; width:100% !important; }
    [data-testid="stRadio"]:has(input[name="road_detection_mode_cards"]) label { flex:1 1 calc(50% - 14px) !important; min-width:220px !important; min-height:128px !important; box-sizing:border-box !important; border:1px solid #344a60 !important; border-radius:18px !important; background:linear-gradient(145deg,#172637,#121d2a) !important; color:#d9e9f8 !important; padding:24px 20px !important; transition:all .18s ease !important; box-shadow:0 8px 22px #0002 !important; }
    [data-testid="stRadio"]:has(input[name="road_detection_mode_cards"]) label:hover { border-color:#38bdf8 !important; background:linear-gradient(145deg,#1e354a,#16283a) !important; transform:translateY(-2px); }
    [data-testid="stRadio"]:has(input[name="road_detection_mode_cards"]) label:has(input:checked) { border-color:#38bdf8 !important; background:linear-gradient(145deg,#0e4665,#1b3450) !important; color:#f4fbff !important; box-shadow:0 0 0 1px #38bdf855,0 8px 20px #0002 !important; }
    [data-testid="stRadio"]:has(input[name="road_detection_mode_cards"]) label p { font-size:18px !important; font-weight:800 !important; }
    [data-testid="stRadio"]:has(input[name="road_detection_mode_cards"]) label input { accent-color:#38bdf8 !important; }
    [data-testid="stMainBlockContainer"] h1, [data-testid="stMainBlockContainer"] h2, [data-testid="stMainBlockContainer"] h3 { scroll-margin-top:5rem; }
    .rg-topbar { margin-top:12px; }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-left:1rem; padding-right:1rem; }
    .main-title { font-size:clamp(30px,3vw,42px); font-weight:850; letter-spacing:-1.4px; color:#f7fbff; line-height:1.12; margin-top:0.15rem; }
    .subtitle { color:var(--rg-muted); font-size:15px; margin-top:5px; }
    .rg-topbar { display:flex; align-items:center; gap:12px; padding:15px 18px; border-radius:16px; margin:6px 0 24px; position:relative; z-index:0; overflow:hidden; background:linear-gradient(120deg,rgba(17,39,58,.96),rgba(19,31,45,.96)); border:1px solid #29465d; box-shadow:0 12px 28px rgba(0,0,0,.16); }
    .rg-logo { width:46px;height:46px;display:grid;place-items:center;border-radius:14px;background:linear-gradient(135deg,#0ea5e9,#2563eb);font-size:24px;box-shadow:0 7px 18px #0284c733; }
    .rg-topname { color:#f4f9ff;font-size:20px;font-weight:850;letter-spacing:-.4px; }
    .rg-topdesc { color:#a7bdd0;font-size:12px;margin-top:2px; }
    .rg-topstatus { margin-left:auto; border:1px solid #23634f; background:#102c25;color:#8ee3b6;padding:6px 10px;border-radius:999px;font-size:11px;font-weight:750;white-space:nowrap; }
    /* Sidebar: preserve Streamlit default fixed layout and internal scroll behavior */
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#121a25,#101620)!important; border-right:1px solid #273445; }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
    [data-testid="stSidebar"] h1 { color:#f4f9ff;font-size:25px;letter-spacing:-.6px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius:11px; padding:8px 10px; transition:all .18s ease; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#1c2b3b; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size:14px;font-weight:600; }
    /* Buttons */
    .stButton > button, .stDownloadButton > button { border-radius:11px; min-height:42px; border:1px solid #334a60; background:linear-gradient(180deg,#1a2b3c,#152334); color:#eaf5ff; font-weight:700; transition:all .18s ease; box-shadow:0 3px 10px #0002; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color:#38bdf8;color:#fff; transform:translateY(-1px); background:linear-gradient(180deg,#203a50,#183047); }
    .stButton > button:focus, .stDownloadButton > button:focus { box-shadow:0 0 0 2px #38bdf855; }
    /* Inputs */
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] { background:#151f2c!important; border-radius:10px!important; border-color:#304257!important; color:#edf5ff!important; }
    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus { border-color:#38bdf8!important; box-shadow:0 0 0 1px #38bdf833!important; }
    [data-testid="stFileUploader"] section { background:#141f2d; border:1px dashed #3b526a; border-radius:15px; padding:12px; }
    /* Metrics and data surfaces */
    div[data-testid="stMetric"] { background:linear-gradient(145deg,#1a2635,#141d29); padding:16px 18px; border-radius:14px; border:1px solid #2c3d50; box-shadow:0 8px 22px #0002; }
    div[data-testid="stMetric"] label { color:#aabbd0!important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#f1f7ff; font-weight:800; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #29394b;border-radius:12px;overflow:hidden; }
    [data-testid="stExpander"] { border:1px solid #29394b!important; border-radius:12px!important; background:#121b26; }
    [data-testid="stTabs"] button { font-weight:700; }
    hr { border-color:#263647!important; }
    h1,h2,h3 { letter-spacing:-.4px; }
    h2 { color:#f2f8ff!important; } h3 { color:#e3effb!important; }
    p,li { line-height:1.55; }
    [data-testid="stAlert"] { border-radius:12px; }
    .rg-page-intro { color:#9fb1c6;font-size:13px;margin-top:0;margin-bottom:18px; }
    @media(max-width:900px) { .rg-topstatus { display:none; } .main-title { font-size:30px; } }
    </style>
    """, unsafe_allow_html=True
)


# =========================

# SIDEBAR

# =========================

st.sidebar.title("🛣️ DrishtiPath")

st.sidebar.caption("AI Road Hazard Detection & Geo-Tagged Reporting")
st.sidebar.markdown("<div style='height:8px'></div><div style='font-size:11px;color:#6f8aa5;letter-spacing:1.4px;font-weight:800'>SAFETY INTELLIGENCE</div>", unsafe_allow_html=True)



page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "🔍 Detect Road Hazards",

        "📍 Map & Reports",

        "ℹ️ About Project",

    ],

)





# =========================

# HEADER

# =========================

st.markdown("<div class='rg-topbar'><div class='rg-logo'>🛣️</div><div><div class='rg-topname'>DrishtiPath</div><div class='rg-topdesc'>AI-powered road hazard detection & geo-tagged reporting</div></div><div class='rg-topstatus'>● SYSTEM READY</div></div>", unsafe_allow_html=True)
st.markdown('<div class="main-title">DrishtiPath</div>', unsafe_allow_html=True)

st.markdown(

    '<div class="subtitle">AI-Based Road Hazard Detection and Geo-Tagged Reporting System</div>',

    unsafe_allow_html=True,

)

st.divider()







# =========================

# DASHBOARD

# =========================

if page == "🏠 Dashboard":

    stats = get_report_statistics()

    reports = get_all_reports()



    total_reports = int(stats.get("total_reports", 0) or 0)

    total_hazards = int(stats.get("total_hazards", 0) or 0)

    high_count = int(stats.get("high", 0) or 0)

    medium_count = int(stats.get("medium", 0) or 0)

    low_count = int(stats.get("low", 0) or 0)



    # Dashboard-only styling: other app pages remain untouched.

    st.markdown("""

    <style>

      .rg-dashboard { margin-top: 3px; }
      .rg-detect-intro { padding:20px 22px; border:1px solid #2d4961; border-radius:18px; background:linear-gradient(135deg,#142438,#172b3c 60%,#183443); margin:8px 0 18px; }
      .rg-detect-title { font-size:23px; font-weight:850; color:#f1f8ff; margin-bottom:6px; }
      .rg-detect-sub { color:#a9c0d5; font-size:13px; }
      .rg-detect-panel { background:linear-gradient(160deg,#121e2c,#142233); border:1px solid #2a4056; border-radius:16px; padding:16px 18px; margin:8px 0 16px; }
      .rg-result-panel { background:#111c29; border:1px solid #2b4258; border-radius:16px; padding:15px; margin:12px 0; }
      [data-testid="stRadio"] { padding:9px 12px; border-radius:12px; background:#111e2b; border:1px solid #2b4055; }
      [data-testid="stFileUploader"] { margin-top:6px; }

      .rg-hero { position:relative; overflow:hidden; padding:28px 30px; border-radius:22px; margin:4px 0 22px;

        background:linear-gradient(115deg,#142337 0%,#18364b 58%,#1d4852 100%); border:1px solid #34546b; }

      .rg-hero:after { content:''; position:absolute; width:230px; height:230px; right:-75px; top:-105px; border-radius:50%;

        background:rgba(110,190,210,.10); border:1px solid rgba(110,190,210,.18); }

      .rg-hero-kicker { color:#8fd7e8; font-size:11px; letter-spacing:2px; font-weight:800; text-transform:uppercase; margin-bottom:9px; }

      .rg-hero-title { color:#f6fbff; font-size:30px; font-weight:850; line-height:1.15; margin:0; }

      .rg-hero-sub { color:#bdccdc; font-size:14px; margin-top:8px; max-width:650px; }

      .rg-hero-chip { display:inline-flex; margin-top:17px; padding:7px 12px; border-radius:999px; color:#9be6bf;

        background:rgba(24,117,76,.22); border:1px solid rgba(89,191,137,.42); font-size:12px; font-weight:700; }

      .rg-kpi { min-height:130px; position:relative; overflow:hidden; border-radius:17px; padding:17px 18px 15px;

        background:linear-gradient(145deg,#202938,#171e29); border:1px solid #323f51; box-shadow:0 8px 22px rgba(0,0,0,.12); }

      .rg-kpi:before { content:''; position:absolute; top:0; left:0; width:100%; height:3px; background:var(--accent,#69b9e7); }

      .rg-kpi-top { display:flex; align-items:center; justify-content:space-between; gap:8px; color:#b6c2d3; font-size:12px; font-weight:750; letter-spacing:.4px; }

      .rg-kpi-icon { display:flex; align-items:center; justify-content:center; width:34px; height:34px; border-radius:10px; background:rgba(255,255,255,.055); font-size:18px; }

      .rg-kpi-value { font-size:29px; font-weight:850; color:#f4f7fb; line-height:1.2; margin-top:12px; }

      .rg-kpi-foot { font-size:12px; color:#8f9db1; margin-top:4px; }

      .rg-section-title { color:#eef4fb; font-size:20px; font-weight:800; margin:3px 0 4px; }

      .rg-section-sub { color:#91a0b5; font-size:12px; margin-bottom:12px; }

      .rg-panel { border:1px solid #303b4d; background:#171e29; border-radius:17px; padding:16px 18px; }

      .rg-panel-title { color:#e8eef6; font-size:15px; font-weight:750; margin-bottom:8px; }

      .rg-sev-row { display:flex; align-items:center; gap:10px; margin:12px 0 3px; color:#dce4ef; font-size:13px; }

      .rg-sev-label { width:70px; font-weight:700; }

      .rg-sev-track { flex:1; height:9px; border-radius:20px; background:#303846; overflow:hidden; }

      .rg-sev-fill { height:9px; border-radius:20px; }

      .rg-sev-count { width:34px; text-align:right; font-weight:800; }

      .rg-sev-percent { font-size:11px; color:#8998ac; margin:0 0 7px 80px; }

      .rg-empty { padding:19px; border:1px dashed #3b4658; border-radius:14px; color:#9eacc0; background:#171d27; }

      .rg-note { color:#9ba9bc; font-size:12px; padding:11px 13px; border-radius:12px; background:#151d29; border:1px solid #2b3749; }

    </style>

    """, unsafe_allow_html=True)





    # Hero section

    st.markdown("""

    <div class="rg-hero">

      <div class="rg-hero-kicker">ROADGUARD AI · LIVE OVERVIEW</div>

      <div class="rg-hero-title">Road Safety Command Center</div>

      <div class="rg-hero-sub">A consolidated view of road hazard detections, severity levels and geo-tagged reports recorded in your system.</div>

      <div class="rg-hero-chip">● Detection engine · YOLOv8</div>

    </div>

    """, unsafe_allow_html=True)





    # KPI cards

    metric_cols = st.columns(4, gap="medium")

    metric_cards = [

        ("REPORTS LOGGED", total_reports, "Geo-tagged reports saved", "📍", "#63b9e9"),

        ("HAZARDS FOUND", total_hazards, "Total detections across reports", "🚧", "#f3b75e"),

        ("HIGH SEVERITY", high_count, "Reports marked high priority", "⚠️", "#f16f78"),

        ("AI ENGINE", "YOLOv8", "Road damage detection model", "🤖", "#8b91f5"),

    ]

    for col, (label, value, foot, icon, accent) in zip(metric_cols, metric_cards):

        with col:

            st.markdown(f"""

            <div class="rg-kpi" style="--accent:{accent}">

              <div class="rg-kpi-top"><span>{label}</span><span class="rg-kpi-icon">{icon}</span></div>

              <div class="rg-kpi-value">{value}</div>

              <div class="rg-kpi-foot">{foot}</div>

            </div>

            """, unsafe_allow_html=True)





    # Report status overview (new)

    status_stats = get_report_status_statistics()

    st.markdown('<div class="rg-section-title">Report Status Overview</div>', unsafe_allow_html=True)

    st.markdown('<div class="rg-section-sub">Track the progress of reports from submission to resolution.</div>', unsafe_allow_html=True)

    status_cols = st.columns(5, gap="small")

    status_cards = [

        ("Pending", status_stats.get("Pending", status_stats.get("pending", 0)), "🕒"),

        ("Verified", status_stats.get("Verified", status_stats.get("verified", 0)), "✅"),

        ("In Progress", status_stats.get("In Progress", status_stats.get("in_progress", 0)), "🔧"),

        ("Resolved", status_stats.get("Resolved", status_stats.get("resolved", 0)), "🟢"),

        ("Rejected", status_stats.get("Rejected", status_stats.get("rejected", 0)), "❌"),

    ]

    for col, (status_label, status_value, status_icon) in zip(status_cols, status_cards):

        with col:

            st.metric(f"{status_icon} {status_label}", int(status_value or 0))



    st.markdown("<br>", unsafe_allow_html=True)



    # Severity analytics

    st.markdown('<div class="rg-section-title">Severity & Risk Snapshot</div>', unsafe_allow_html=True)

    st.markdown('<div class="rg-section-sub">Understand how saved reports are distributed by preliminary severity.</div>', unsafe_allow_html=True)

    sev_col1, sev_col2 = st.columns([1.15, 1], gap="large")

    with sev_col1:

        st.markdown('<div class="rg-panel-title">Severity distribution</div>', unsafe_allow_html=True)

        severity_df = pd.DataFrame({"Severity": ["High", "Medium", "Low"], "Reports": [high_count, medium_count, low_count]})

        st.bar_chart(severity_df.set_index("Severity"), height=280)

    with sev_col2:

        st.markdown('<div class="rg-panel-title">Report severity breakdown</div>', unsafe_allow_html=True)

        denominator = total_reports if total_reports else 1

        severity_items = [("High", high_count, "#f16f78"), ("Medium", medium_count, "#f3b75e"), ("Low", low_count, "#58c99a")]

        for label, count, color in severity_items:

            pct = (count / denominator * 100) if total_reports else 0

            st.markdown(f"""

            <div class="rg-sev-row"><div class="rg-sev-label">{label}</div>

              <div class="rg-sev-track"><div class="rg-sev-fill" style="width:{pct:.1f}%;background:{color};"></div></div>

              <div class="rg-sev-count">{count}</div></div>

            <div class="rg-sev-percent">{pct:.1f}% of saved reports</div>

            """, unsafe_allow_html=True)

        if not total_reports:

            st.caption("Severity metrics will populate after you save reports.")



    st.divider()





    # Hazard type analytics

    st.markdown('<div class="rg-section-title">Hazard Intelligence</div>', unsafe_allow_html=True)

    st.markdown('<div class="rg-section-sub">Breakdown of the detected road-damage categories from report records.</div>', unsafe_allow_html=True)

    hazard_counts = {}

    for report in reports:

        hazard_data = report.get("hazard_data", [])

        if isinstance(hazard_data, str):

            try:

                hazard_data = json.loads(hazard_data)

            except Exception:

                hazard_data = []

        if isinstance(hazard_data, dict):

            hazard_data = hazard_data.get("detections", [])

        if isinstance(hazard_data, list):

            for hazard in hazard_data:

                if isinstance(hazard, dict):

                    hazard_name = str(hazard.get("hazard", hazard.get("class", "Unknown"))).title()

                    hazard_counts[hazard_name] = hazard_counts.get(hazard_name, 0) + 1



    if hazard_counts:

        hazard_df = pd.DataFrame(list(hazard_counts.items()), columns=["Hazard Type", "Count"]).sort_values("Count", ascending=False)

        haz_col1, haz_col2 = st.columns([1.6, 1], gap="large")

        with haz_col1:

            st.markdown('<div class="rg-panel-title">Detection count by hazard type</div>', unsafe_allow_html=True)

            st.bar_chart(hazard_df.set_index("Hazard Type"), height=320)

        with haz_col2:

            top_hazard = hazard_df.iloc[0]

            st.markdown('<div class="rg-panel-title">Top detected hazard</div>', unsafe_allow_html=True)

            st.metric("Most frequent category", str(top_hazard["Hazard Type"]), f"{int(top_hazard['Count'])} detections")

            st.markdown('<div class="rg-panel-title" style="margin-top:16px">Hazard category totals</div>', unsafe_allow_html=True)

            st.dataframe(hazard_df, use_container_width=True, hide_index=True, height=260)

    else:

        st.markdown('<div class="rg-empty">No detection analytics yet. Analyze a road image and save a report to start building this overview.</div>', unsafe_allow_html=True)





    st.divider()

    # Recent report activity

    st.markdown('<div class="rg-section-title">Recent Report Activity</div>', unsafe_allow_html=True)

    st.markdown('<div class="rg-section-sub">Latest saved reports for a quick review.</div>', unsafe_allow_html=True)

    if reports:

        recent_reports = reports[:5]

        recent_df = pd.DataFrame([

            {"Report ID": r.get("id", "—"), "Media": str(r.get("media_type", "image")).title(), "Date": r.get("report_date", "—"),

             "Location": r.get("location_name", "Unknown location"), "Hazards": r.get("total_hazards", 0),

             "Severity": r.get("severity", "—"), "Status": r.get("status", "Pending")}

            for r in recent_reports

        ])

        st.dataframe(recent_df, use_container_width=True, hide_index=True)

    else:

        st.markdown('<div class="rg-empty">No reports saved yet. Your recent activity will appear here.</div>', unsafe_allow_html=True)





    st.markdown('<div class="rg-note">Tip: Analyze a road image in <b>Detect Road Hazards</b> to generate a detection result and save it as a geo-tagged report. Visit <b>Map & Reports</b> to explore reports on the map.</div>', unsafe_allow_html=True)





# =========================

# DETECTION PAGE

# =========================

elif page == "🔍 Detect Road Hazards":

    st.markdown("""
    <div class="rg-detect-intro">
      <div class="rg-detect-title">🧠 AI Hazard Analysis</div>
      <div class="rg-detect-sub">Analyze road images or video clips, review detected damage, and turn findings into geo-tagged safety reports.</div>
    </div>
    """, unsafe_allow_html=True)



    detection_mode = st.radio(
        "Choose Detection Mode",
        options=["🖼️ Image Detection", "🎬 Video Detection"],
        index=0,
        key="road_detection_mode_cards",
        horizontal=True,
    )
    detection_mode = "Image Detection" if detection_mode == "🖼️ Image Detection" else "Video Detection"



    if detection_mode == "Image Detection":

        st.markdown("<div class='rg-detect-panel'><b>🖼️ Image Analysis</b><br><span style='color:#9fb4c9'>Upload a clear road photo to detect potholes, cracks, debris and surface damage.</span></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)



        uploaded_file = st.file_uploader(

            "Upload Road Image",

            type=["jpg", "jpeg", "png"],

            key="road_image_upload",

        )



        if uploaded_file is not None:

            try:

                image = Image.open(uploaded_file).convert("RGB")

                st.image(image, caption="Uploaded Road Image", use_container_width=True)



                if st.button("🚀 Analyze Road Image", use_container_width=True):

                    with st.spinner("AI is analyzing the image..."):

                        upload_dir = Path("uploads")

                        upload_dir.mkdir(parents=True, exist_ok=True)



                        safe_original_name = Path(uploaded_file.name).name

                        unique_name = f"{uuid.uuid4().hex}_{safe_original_name}"

                        image_path = upload_dir / unique_name

                        image.save(image_path)



                        # If GPS coordinates are printed on the photo stamp (but not in EXIF),

                        # try OCR and use them as the initial report/map location.

                        stamped_gps = extract_stamped_gps_coordinates(image)

                        if stamped_gps:

                            gps_lat, gps_lon, gps_text = stamped_gps

                            st.session_state.latitude = gps_lat

                            st.session_state.longitude = gps_lon

                            st.session_state.location_name = f"GPS photo coordinates ({gps_lat:.6f}, {gps_lon:.6f})"

                            st.session_state.report_location_name = st.session_state.location_name

                            st.session_state.gps_source = "OCR photo stamp"

                            st.session_state.location_map_revision = st.session_state.get("location_map_revision", 0) + 1

                        else:

                            st.session_state.gps_source = "No stamped GPS coordinates detected"



                        annotated_image, detections, total = detect_hazards(str(image_path))

                        image_width, image_height = image.size

                        severity, severity_score = calculate_severity(

                            detections, image_width, image_height

                        )



                        st.session_state.detections = detections

                        st.session_state.annotated_image = annotated_image

                        st.session_state.total_hazards = total

                        st.session_state.severity = severity

                        st.session_state.severity_score = severity_score

                        st.session_state.image_name = unique_name



                    st.success("Analysis completed successfully!")

            except Exception as error:

                st.error(f"Could not process this image: {error}")



        # Show the most recent detection results after Streamlit reruns

        if st.session_state.detections is not None:

            detections = st.session_state.detections

            total = st.session_state.total_hazards

            severity = st.session_state.severity

            severity_score = st.session_state.severity_score



            st.markdown("<div class='rg-result-panel'><h3 style='margin:0'>📊 Detection Summary</h3><p style='color:#9fb4c9;margin:4px 0 0'>AI analysis results and severity overview</p></div>", unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4

            )

            col1.metric("Hazards Detected", total)

            col2.metric("Preliminary Severity", severity)

            col3.metric("Severity Score", severity_score)

            if total > 0:

                avg_conf = sum(d["confidence"] for d in detections) / total

                col4.metric("Average Confidence", f"{avg_conf:.2f}%")

            else:

                col4.metric("Average Confidence", "N/A")



            st.markdown("### 🖼️ AI Detection Result")

            st.image(st.session_state.annotated_image, use_container_width=True)

            gps_source = st.session_state.get("gps_source")

            if gps_source == "OCR photo stamp":

                st.success("GPS coordinates were read from the location stamp printed on the photo. The map is centered on those coordinates; click the map to adjust manually.")

            elif gps_source:

                st.info("GPS coordinates were not read from the photo stamp. You can still set the report location by clicking the map manually.")





            if detections:

                detection_df = pd.DataFrame(

                    [

                        {

                            "Hazard": d["hazard"].title(),

                            "Confidence (%)": d["confidence"],

                            "Bounding Box": str(d["box"]),

                        }

                        for d in detections

                    ]

                )

                st.markdown("### 📋 Detection Details")

                st.dataframe(detection_df, use_container_width=True, hide_index=True)

            else:

                st.warning("No hazards were detected in this image.")



            # Download annotated image

            buffer = io.BytesIO()

            Image.fromarray(st.session_state.annotated_image).save(buffer, format="JPEG")

            st.download_button(

                "⬇️ Download AI Detection Image",

                data=buffer.getvalue(),

                file_name="road_hazard_detection.jpg",

                mime="image/jpeg",

            )



            # =========================

            # LOCATION REPORTING WITH GEOCODING + MANUAL MAP REFINEMENT

            # =========================

            st.divider()

            st.markdown("## 📍 Geo-Tagged Reporting")

            st.write("Search a road, area, landmark or address. Select a result to move the map, then click the map to fine-tune the exact point.")



            search_place = st.text_input(

                "🔎 Search Location",

                placeholder="Example: Gola Ka Mandir, Gwalior, Madhya Pradesh",

                key="location_search_input",

            )



            if "google_location_results" not in st.session_state:

                st.session_state.google_location_results = []

            if "location_map_revision" not in st.session_state:

                st.session_state.location_map_revision = 0

            if "last_google_selected_label" not in st.session_state:

                st.session_state.last_google_selected_label = None



            if st.button("📍 Search Locations", use_container_width=True):

                if search_place.strip():

                    with st.spinner("Searching location..."):

                        st.session_state.google_location_results = search_locations_geoapify(search_place.strip())

                    st.session_state.pop("google_location_selection", None)

                    st.session_state.last_google_selected_label = None

                    st.session_state.pop("last_manual_map_click", None)

                    st.session_state.location_map_revision += 1

                    if st.session_state.google_location_results:

                        st.success(f"Found {len(st.session_state.google_location_results)} location result(s). Select a result below.")

                    else:

                        st.warning("No results found. Try a more specific search, including city/state.")

                else:

                    st.warning("Please enter a location first.")



            search_results = st.session_state.google_location_results

            if search_results:

                result_labels = [f"{i + 1}. {item['label']}" for i, item in enumerate(search_results)]

                selected_label = st.selectbox(

                    "Choose the correct location",

                    result_labels,

                    key="google_location_selection",

                    help="Select a result to move the map to that location. You can then click the map to refine it.",

                )

                if selected_label != st.session_state.last_google_selected_label:

                    selected_index = result_labels.index(selected_label)

                    selected_place = search_results[selected_index]

                    st.session_state.latitude = float(selected_place["latitude"])

                    st.session_state.longitude = float(selected_place["longitude"])

                    st.session_state.location_name = selected_place["label"]

                    st.session_state.report_location_name = selected_place["label"]

                    st.session_state.last_google_selected_label = selected_label

                    st.session_state.location_map_revision += 1



            reports = get_all_reports()

            location_map = create_location_map(

                latitude=st.session_state.latitude,

                longitude=st.session_state.longitude,

                reports=reports,

                selection_mode=True,

                selected_location_name=st.session_state.get("location_name") or st.session_state.get("report_location_name") or "Selected location",

            )

            st.markdown("### 🗺️ Selected Location")

            components.html(location_map, height=480, scrolling=False)

            st.caption("The map is interactive. Click the map to view coordinates in its overlay, then copy them into the coordinate fields below. Place search uses Geoapify; you can also adjust the point manually.")


            st.markdown("### 📌 Selected Place")

            st.info(st.session_state.get("location_name") or st.session_state.get("report_location_name") or "Map point selected — click the map to fine-tune the location.")

            st.markdown("### 📌 Location Coordinates")

            col1, col2 = st.columns(2)

            with col1:
                st.number_input("Latitude", min_value=-90.0, max_value=90.0, format="%.6f", key="latitude")

            with col2:
                st.number_input("Longitude", min_value=-180.0, max_value=180.0, format="%.6f", key="longitude")



            location_name = st.text_input(

                "Location / Road Name",

                key="report_location_name",

            )



            # =========================

            # DUPLICATE REPORT CHECK + SAVE REPORT

            # =========================

            existing_reports = get_all_reports()

            possible_duplicates = find_possible_duplicate_reports(

                detections,

                st.session_state.latitude,

                st.session_state.longitude,

                existing_reports,

                radius_m=100,

            ) if total > 0 else []



            allow_duplicate_save = False

            if possible_duplicates:

                st.warning("⚠️ Possible duplicate report(s) found nearby with matching hazard type(s).")

                duplicate_rows = []

                for match in possible_duplicates:

                    old = match["report"]

                    duplicate_rows.append({

                        "Report ID": old.get("id"),

                        "Location": old.get("location_name", "Unknown"),

                        "Distance (m)": round(match["distance_m"], 1),

                        "Matching Hazard(s)": ", ".join(match["shared_hazards"]).title(),

                        "Status": old.get("status", "Pending"),

                    })

                st.dataframe(pd.DataFrame(duplicate_rows), use_container_width=True, hide_index=True)

                allow_duplicate_save = st.checkbox(

                    "I checked the possible duplicates and still want to save this report.",

                    key="confirm_duplicate_report_save",

                )

                st.caption("Duplicate check uses a 100 m GPS radius and at least one matching hazard category. It is a warning, not proof that reports are identical.")



            if st.button("💾 Save Hazard Report", use_container_width=True):

                if not st.session_state.image_name:

                    st.warning("Please analyze an image before saving a report.")

                elif total <= 0:

                    st.warning("No hazards were detected. A hazard report cannot be saved as an empty report.")

                elif possible_duplicates and not allow_duplicate_save:

                    st.warning("Please confirm above if you want to save this report despite the possible duplicate warning.")

                else:

                    report_id = save_report(

                        image_name=st.session_state.image_name,

                        detections=detections,

                        total_hazards=total,

                        severity=severity,

                        latitude=st.session_state.latitude,

                        longitude=st.session_state.longitude,

                        location_name=location_name,

                    )

                    st.success(f"Report #{report_id} saved successfully!")

                    st.balloons()







    else:

        st.caption("Upload a road video, preview it, then run AI analysis and compare the original with the annotated result.")

        video_file = st.file_uploader("Upload Road Video", type=["mp4", "avi", "mov", "mkv"], key="road_video_upload")

        sample_interval = st.slider(

            "Analyze one frame every (seconds)", min_value=0.5, max_value=5.0, value=1.0, step=0.5,

            help="Smaller intervals analyze more frames and take longer to process.",

        )



        if video_file is not None:

            upload_dir = Path("uploads")

            upload_dir.mkdir(parents=True, exist_ok=True)

            video_bytes = video_file.getvalue()

            signature = f"{video_file.name}:{len(video_bytes)}"

            if st.session_state.get("video_upload_signature") != signature:

                source_path = upload_dir / f"source_{uuid.uuid4().hex}_{Path(video_file.name).stem}.mp4"

                preview_path = upload_dir / f"preview_{uuid.uuid4().hex}.mp4"

                source_path.write_bytes(video_bytes)

                try:

                    convert_video_for_browser(source_path, preview_path)

                    st.session_state.video_source_path = str(source_path)

                    st.session_state.video_preview_path = str(preview_path)

                    st.session_state.video_upload_signature = signature

                    st.session_state.video_summary = None

                    st.session_state.video_detections = []

                    st.session_state.processed_video_path = None

                except Exception as error:

                    st.error(f"Could not prepare video preview: {error}")



            source_path_value = st.session_state.get("video_source_path")

            preview_path_value = st.session_state.get("video_preview_path")

            if preview_path_value and Path(preview_path_value).exists():

                st.markdown("### 🎬 Original Video Preview")

                st.video(Path(preview_path_value).read_bytes(), format="video/mp4")



            if st.button("🎥 Analyze Road Video", use_container_width=True, key="analyze_road_video_btn"):

                if not source_path_value or not Path(source_path_value).exists():

                    st.error("Please upload the video again. Source video is missing.")

                else:

                    raw_output = upload_dir / f"processed_raw_{uuid.uuid4().hex}.mp4"

                    browser_output = upload_dir / f"processed_browser_{uuid.uuid4().hex}.mp4"

                    with st.spinner("AI is analyzing sampled video frames..."):

                        try:

                            raw_path, video_detections, video_summary = detect_hazards_in_video(

                                source_path_value, str(raw_output), sample_interval=sample_interval

                            )

                            convert_video_for_browser(raw_path, browser_output)

                            st.session_state.video_detections = video_detections

                            st.session_state.video_summary = video_summary

                            st.session_state.processed_video_path = str(browser_output)

                            st.session_state.source_video_name = Path(source_path_value).name

                            st.success("Video analysis completed!")

                        except Exception as error:

                            st.error(f"Could not process this video: {error}")



        if st.session_state.get("video_summary") is not None:

            video_summary = st.session_state.video_summary

            video_detections = st.session_state.get("video_detections", [])

            st.markdown("## 📊 Video Analysis Summary")

            duration = video_summary.get("source_duration_seconds", 0)

            mins, secs = divmod(int(duration), 60)

            metric_cols = st.columns(4)

            metric_cols[0].metric("Video Duration", f"{mins}:{secs:02d}")

            metric_cols[1].metric("Frames Analyzed", video_summary.get("analyzed_frames", 0))

            metric_cols[2].metric("Unique Hazards", video_summary.get("unique_hazard_total", len(video_detections)))

            metric_cols[3].metric("Sampling Interval", f"{video_summary.get('sample_interval_seconds', 1)} sec")



            if video_detections:

                st.markdown("### 🧱 Unique Hazard Counts")

                unique_counts = video_summary.get("unique_hazard_counts", {})

                if unique_counts:

                    st.dataframe(pd.DataFrame([{"Hazard": k.title(), "Unique Count": v} for k, v in unique_counts.items()]),

                                 use_container_width=True, hide_index=True)

                st.markdown("### 🧭 Tracked Hazards")

                tracked_df = pd.DataFrame(video_summary.get("unique_hazards", []))

                if not tracked_df.empty:

                    tracked_df = tracked_df.rename(columns={

                        "track_id": "Track ID", "hazard": "Hazard", "first_seen_seconds": "First Seen (sec)",

                        "last_seen_seconds": "Last Seen (sec)", "observations": "Frames Observed",

                        "max_confidence": "Max Confidence (%)"})

                    st.dataframe(tracked_df, use_container_width=True, hide_index=True)
                with st.expander("View all frame-level detections"):

                    st.dataframe(pd.DataFrame([{

                        "Timestamp (sec)": d.get("timestamp_seconds"), "Hazard": d.get("hazard", "").title(),

                        "Track ID": d.get("track_id"), "Confidence (%)": d.get("confidence")}

                        for d in video_detections]), use_container_width=True, hide_index=True)

                st.caption(f"Raw frame detections: {len(video_detections)}. Repeat detections associated with the same tracked hazard are counted once in Unique Hazards.")

            else:

                st.info("No hazards were detected in the analyzed frames.")



            st.markdown("### 🎞️ Original vs AI-Processed Video")

            original_col, processed_col = st.columns(2)

            with original_col:

                st.markdown("**Original Video**")

                preview_path_value = st.session_state.get("video_preview_path")

                if preview_path_value and Path(preview_path_value).exists():

                    st.video(Path(preview_path_value).read_bytes(), format="video/mp4")

            with processed_col:

                st.markdown("**AI-Processed Video**")

                processed_path_value = st.session_state.get("processed_video_path")

                if processed_path_value and Path(processed_path_value).exists():

                    processed_bytes = Path(processed_path_value).read_bytes()

                    st.video(processed_bytes, format="video/mp4")

                    st.download_button("⬇️ Download Processed Video", data=processed_bytes,

                                       file_name="roadguard_ai_processed_video.mp4", mime="video/mp4")

            st.caption("Both videos are converted to browser-compatible H.264 MP4. The processed video contains sampled frames only.")



            # Build one representative detection per tracked hazard for severity/reporting.

            unique_detections_by_track = {}

            for item in video_detections:

                track_id = item.get("track_id")

                if track_id is not None and track_id not in unique_detections_by_track:

                    unique_detections_by_track[track_id] = item

            unique_video_detections = list(unique_detections_by_track.values())

            video_width = int(video_summary.get("video_width", 0) or 0)

            video_height = int(video_summary.get("video_height", 0) or 0)

            video_severity, video_severity_score = calculate_severity(

                unique_video_detections, video_width, video_height

            ) if video_width > 0 and video_height > 0 else ("No Hazards", 0)



            st.divider()

            st.markdown("## 📍 Save Geo-Tagged Video Report")

            st.caption("Select the location where the road hazard was recorded. Video GPS metadata is not automatically extracted yet, so verify the map pin manually.")



            # Map click updates the coordinate widgets; user can fine-tune them below.

            if "video_report_latitude_input" not in st.session_state:

                st.session_state.video_report_latitude_input = 26.2183

            if "video_report_longitude_input" not in st.session_state:

                st.session_state.video_report_longitude_input = 78.1828

            if "video_report_location_name_input" not in st.session_state:

                st.session_state.video_report_location_name_input = ""



            video_location_map = create_location_map(

                latitude=float(st.session_state.video_report_latitude_input),

                longitude=float(st.session_state.video_report_longitude_input),

                selection_mode=True, selected_location_name=st.session_state.video_report_location_name_input or "Video hazard location"

            )

            components.html(video_location_map, height=380, scrolling=False)

            st.caption("Click the map to display coordinates in the map overlay. Copy them into the Latitude and Longitude fields below.")


            loc_col1, loc_col2 = st.columns(2)

            with loc_col1:

                video_location_name = st.text_input(

                    "Location name", key="video_report_location_name_input",

                    placeholder="e.g. City Centre Road, Gwalior"

                )

                video_latitude = st.number_input(

                    "Latitude", min_value=-90.0, max_value=90.0,

                    format="%.6f", key="video_report_latitude_input"

                )

            with loc_col2:

                video_longitude = st.number_input(

                    "Longitude", min_value=-180.0, max_value=180.0,

                    format="%.6f", key="video_report_longitude_input"

                )

                st.metric("Video Severity", video_severity, f"Score: {video_severity_score}")



            if st.button(

                "💾 Save Video Hazard Report", use_container_width=True,

                disabled=(video_summary.get("unique_hazard_total", 0) == 0),

                key="save_video_hazard_report_btn"

            ):

                if not video_location_name.strip():

                    st.warning("Please enter the road location name before saving the report.")

                else:

                    try:

                        report_id = save_report(

                            image_name=Path(st.session_state.get("video_source_path", "road_video.mp4")).name,

                            detections=unique_video_detections,

                            total_hazards=int(video_summary.get("unique_hazard_total", len(unique_video_detections))),

                            severity=video_severity,

                            latitude=float(video_latitude),

                            longitude=float(video_longitude),

                            location_name=video_location_name.strip(),

                            media_type="video",

                            media_path=st.session_state.get("video_source_path"),

                            processed_media_path=st.session_state.get("processed_video_path"),

                            report_metadata={

                                "source_duration_seconds": video_summary.get("source_duration_seconds"),

                                "analyzed_frames": video_summary.get("analyzed_frames"),

                                "frame_level_detection_total": video_summary.get("frame_level_detection_total"),

                                "unique_hazard_total": video_summary.get("unique_hazard_total"),

                                "unique_hazards": video_summary.get("unique_hazards", []),

                                "sample_interval_seconds": video_summary.get("sample_interval_seconds"),

                                "severity_score": video_severity_score,

                            },

                        )

                        st.success(f"Video report #{report_id} saved successfully! Open Map & Reports to view it.")

                    except Exception as error:

                        st.error(f"Could not save video report: {error}")





# =========================

# MAP & REPORTS

# =========================

elif page == "📍 Map & Reports":

    st.subheader("📍 Geo-Tagged Road Hazard Reports")

    reports = get_all_reports()



    if not reports:

        st.info("No reports saved yet. Detect a road hazard and save its report to see it here.")

    else:

        # ---------- Filters ----------

        st.markdown("### 🔎 Search & Filter Reports")

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:

            severity_filter = st.selectbox("Filter by Severity", ["All", "High", "Medium", "Low"], key="mr_severity_filter")

        with filter_col2:

            location_filter = st.text_input("Search Location", placeholder="Enter location name...", key="mr_location_filter")



        def parse_hazard_data(value):

            if isinstance(value, str):

                try:

                    value = json.loads(value)

                except (json.JSONDecodeError, TypeError):

                    value = []

            if isinstance(value, dict):

                value = value.get("detections", [])

            return value if isinstance(value, list) else []



        available_hazards = set()

        for report in reports:

            for item in parse_hazard_data(report.get("hazard_data", [])):

                if isinstance(item, dict):

                    name = item.get("hazard", item.get("class", "Unknown"))

                    available_hazards.add(str(name).title())

        with filter_col3:

            hazard_filter = st.selectbox("Filter by Hazard Type", ["All"] + sorted(available_hazards), key="mr_hazard_filter")



        # ---------- Apply filters ----------

        filtered_reports = []

        for report in reports:

            if severity_filter != "All" and report.get("severity") != severity_filter:

                continue

            if location_filter.strip() and location_filter.strip().lower() not in str(report.get("location_name", "")).lower():

                continue

            if hazard_filter != "All":

                matches = any(

                    isinstance(item, dict) and str(item.get("hazard", item.get("class", "Unknown"))).title() == hazard_filter

                    for item in parse_hazard_data(report.get("hazard_data", []))

                )

                if not matches:

                    continue

            filtered_reports.append(report)



        # ---------- Filter counts ----------

        st.markdown("### 📊 Filter Results")

        count_col1, count_col2 = st.columns(2)

        count_col1.metric("Matching Reports", len(filtered_reports))

        count_col2.metric("Total Reports", len(reports))

        st.divider()



        if not filtered_reports:

            st.warning("No reports match the selected filters. Try changing or clearing your filters.")

        else:

            # ---------- Map ----------

            st.markdown("### 🗺️ Report Locations")

            report_map = create_location_map(

                latitude=26.2183, longitude=78.1828,

                reports=filtered_reports, selection_mode=False,

                show_nearby=False,

            )

            components.html(report_map, height=520, scrolling=False)



            # ---------- Report table ----------

            st.markdown("### 📋 Report History")

            reports_df = pd.DataFrame([

                {

                    "Report ID": r.get("id"),

                    "Media Type": str(r.get("media_type", "image")).title(),

                    "Date": r.get("report_date"),

                    "Location": r.get("location_name"),

                    "Hazards": r.get("total_hazards"),

                    "Severity": r.get("severity"),

                    "Status": r.get("status", "Pending"),

                    "Latitude": r.get("latitude"),

                    "Longitude": r.get("longitude"),

                }

                for r in filtered_reports

            ])

            st.dataframe(reports_df, use_container_width=True, hide_index=True)

            csv_data = reports_df.to_csv(index=False).encode("utf-8")

            st.download_button(

                "⬇️ Download Filtered Reports CSV", data=csv_data,

                file_name="road_hazard_reports.csv", mime="text/csv",

            )

            st.divider()



            # ---------- Report details ----------

            st.markdown("## 🔍 Report Details")

            report_ids = [r.get("id") for r in filtered_reports]

            selected_id = st.selectbox(

                "Select a Report", report_ids,

                format_func=lambda value: f"Report #{value}", key="mr_selected_report",

            )

            selected_report = next(r for r in filtered_reports if r.get("id") == selected_id)

            st.markdown(f"### 📄 Report #{selected_report.get('id')}")

            if selected_report.get("media_type") == "video":

                st.markdown("**Video evidence**")

                evidence_path = selected_report.get("processed_media_path") or selected_report.get("media_path")

                if evidence_path and Path(evidence_path).exists():

                    st.video(Path(evidence_path).read_bytes())

                else:

                    st.info("Video evidence file is not available at the saved path on this device.")

            detail_col1, detail_col2, detail_col3 = st.columns(3)

            detail_col1.metric("Total Hazards", selected_report.get("total_hazards", 0))

            detail_col2.metric("Severity", selected_report.get("severity", "N/A"))

            detail_col3.metric("Report Date", selected_report.get("report_date", "N/A"))



            # ---------- Report Status Management (new) ----------

            st.markdown("### 📌 Report Status Management")

            current_status = selected_report.get("status", "Pending")

            if current_status not in REPORT_STATUSES:

                current_status = "Pending"

            status_col1, status_col2 = st.columns([2, 1])

            with status_col1:

                new_status = st.selectbox(

                    "Update report status",

                    REPORT_STATUSES,

                    index=REPORT_STATUSES.index(current_status),

                    key=f"mr_status_select_{selected_id}",

                )

            with status_col2:

                st.write("")

                st.write("")

                if st.button("💾 Update Status", key=f"mr_status_update_{selected_id}", use_container_width=True):

                    try:

                        update_report_status(selected_id, new_status)

                        st.success(f"Report #{selected_id} status updated to {new_status}.")

                        st.rerun()

                    except Exception as error:

                        st.error(f"Could not update report status: {error}")

            st.caption(f"Current status: **{current_status}**")



            st.markdown("### 📍 Location Information")

            st.write("**Location:**", selected_report.get("location_name", "Unknown"))

            coord_col1, coord_col2 = st.columns(2)

            with coord_col1:

                st.markdown(f"**Latitude:** {selected_report.get('latitude', 'N/A')}")

            with coord_col2:

                st.markdown(f"**Longitude:** {selected_report.get('longitude', 'N/A')}")



            # ---------- Hazard breakdown ----------

            st.markdown("### 🚧 Detected Hazard Breakdown")

            detail_hazards = parse_hazard_data(selected_report.get("hazard_data", []))

            hazard_rows = []

            for index, item in enumerate(detail_hazards, start=1):

                if not isinstance(item, dict):

                    continue

                try:

                    confidence = float(item.get("confidence", 0))

                except (TypeError, ValueError):

                    confidence = 0.0

                hazard_rows.append({

                    "No.": index,

                    "Hazard Type": item.get("hazard", item.get("class", "Unknown")),

                    "Confidence (%)": round(confidence, 2),

                })

            if hazard_rows:

                st.dataframe(pd.DataFrame(hazard_rows), use_container_width=True, hide_index=True)

                hazard_summary = {}

                for row in hazard_rows:

                    name = row["Hazard Type"]

                    hazard_summary[name] = hazard_summary.get(name, 0) + 1

                st.markdown("#### 📊 Hazard Type Summary")

                st.dataframe(pd.DataFrame(list(hazard_summary.items()), columns=["Hazard Type", "Count"]), use_container_width=True, hide_index=True)

            else:

                st.info("No hazard breakdown is stored for this report.")



            # ---------- Selected report map ----------

            st.markdown("### 🗺️ Selected Report Location")

            lat = selected_report.get("latitude")

            lon = selected_report.get("longitude")

            if lat is not None and lon is not None:

                try:

                    detail_map = create_location_map(

                        latitude=float(lat), longitude=float(lon),

                        reports=[selected_report], selection_mode=False,

                    )

                    components.html(detail_map, height=420, scrolling=False)

                except (ValueError, TypeError):

                    st.warning("Unable to display the map for this report.")

            else:

                st.warning("Location coordinates are not available for this report.")





            # ---------- Report media (image or video) ----------

            st.markdown("### 🖼️ Report Media")
            media_name = selected_report.get("image_name")

            if media_name:
                try:
                    if isinstance(media_name, Image.Image):
                        st.image(media_name, caption=f"Media for Report #{selected_id}", use_container_width=True)
                    elif isinstance(media_name, (bytes, bytearray, io.BytesIO)):
                        media_bytes = media_name.getvalue() if isinstance(media_name, io.BytesIO) else bytes(media_name)
                        with Image.open(io.BytesIO(media_bytes)) as report_img:
                            st.image(report_img.copy(), caption=f"Image for Report #{selected_id}", use_container_width=True)
                    else:
                        # Database may contain a Windows-style path or only a filename.
                        stored_value = str(media_name).strip().replace("\\", "/")
                        stored_path = Path(stored_value)
                        candidates = [stored_path]
                        if not stored_path.is_absolute():
                            candidates.extend([Path.cwd() / stored_path, Path("uploads") / stored_path.name])
                        # If a full uploads/... path was saved, also try the filename in uploads.
                        candidates.append(Path("uploads") / stored_path.name)
                        resolved_path = next((candidate for candidate in candidates if candidate.exists() and candidate.is_file()), None)

                        if resolved_path is None:
                            st.warning(f"Report media file not found: {stored_path.name}")
                        else:
                            suffix = resolved_path.suffix.lower()
                            video_suffixes = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
                            if suffix in video_suffixes:
                                st.video(str(resolved_path))
                                st.caption(f"Video attached to Report #{selected_id}")
                            else:
                                with Image.open(resolved_path) as report_img:
                                    st.image(report_img.copy(), caption=f"Image for Report #{selected_id}", use_container_width=True)
                except Exception as media_error:
                    st.warning(f"Could not display report media: {media_error}")
            else:
                st.info("No image or video file is attached to this report.")




# =========================

# ABOUT PROJECT

# =========================

elif page == "ℹ️ About Project":

    st.subheader("ℹ️ About DrishtiPath")

    st.write(

        """

        DrishtiPath is an AI-based road hazard detection and geo-tagged reporting application.

        The system uses a trained YOLO object detection model to identify road damage including

        potholes, cracks, edge damage and different road surface defects.

        Detected hazards can be associated with geographic coordinates and saved as reports for visualization.

        """

    )



    st.markdown("### Technology Stack")

    st.write(

        """

        - Python

        - Streamlit

        - YOLOv8

        - OpenCV

        - SQLite

        - Folium

        - Streamlit-Folium

        - Pandas

        - Pillow

        - OpenStreetMap (manual map selection)

        """

    )



    st.markdown("### Project Objectives")

    st.write(

        """

        - Detect road hazards from images

        - Identify multiple road damage categories

        - Display detection confidence

        - Calculate a rule-based preliminary visual severity score

        - Record geographical coordinates

        - Save reports in a database

        - Visualize reported hazards on a map

        """

    )
