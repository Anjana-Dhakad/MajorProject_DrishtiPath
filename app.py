import io
import uuid
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from streamlit_folium import st_folium

from detector import detect_hazards
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
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide",
)

initialize_database()

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
    """Search Geoapify, rank text matches, and prevent distant false matches for Gwalior queries."""
    if not place or not place.strip():
        return []

    try:
        api_key = st.secrets.get("GEOAPIFY_API_KEY", "").strip()
    except Exception:
        api_key = ""
    if not api_key:
        st.error("Geoapify API key is missing. Add GEOAPIFY_API_KEY to .streamlit/secrets.toml.")
        return []

    original = place.strip()
    lower = original.lower()
    city_names = ["gwalior", "delhi", "mumbai", "raipur", "bhopal", "indore", "jaipur", "agra", "lucknow", "pune", "chennai", "kolkata"]
    explicit_city = next((city for city in city_names if city in lower), None)
    local_terms = ["itm", "purani chhawani", "purani chawani", "morar", "thatipur", "gola ka mandir", "sithouli", "sitholi"]
    is_gwalior_local = any(term in lower for term in local_terms) or explicit_city == "gwalior"

    # For known Gwalior-area searches, query with city context and bias the search
    # toward Gwalior. Other explicitly named cities are not overridden.
    queries = [original]
    if is_gwalior_local and "gwalior" not in lower:
        queries.insert(0, f"{original}, Gwalior, Madhya Pradesh, India")
    elif not explicit_city:
        queries.append(f"{original}, India")
    queries = list(dict.fromkeys(queries))

    all_features = []
    try:
        for query in queries:
            params = {
                "text": query,
                "limit": limit,
                "filter": "countrycode:in",
                "lang": "en",
                "apiKey": api_key,
            }
            if is_gwalior_local:
                params["bias"] = "proximity:78.1828,26.2183"
            url = "https://api.geoapify.com/v1/geocode/search?" + urlencode(params)
            request = Request(url, headers={"User-Agent": "RoadGuardAI_RoadHazard_Project/1.0"})
            with urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            all_features.extend(payload.get("features", []))

        # Ignore generic tokens such as "university" and city/state words when
        # matching names; they caused unrelated universities to appear as hits.
        generic_tokens = {"india", "madhya", "pradesh", "gwalior", "university", "college", "road", "near", "area"}
        query_tokens = [w.lower() for w in original.split() if len(w) > 2 and w.lower() not in generic_tokens]
        results_by_key = {}
        for feature in all_features:
            props = feature.get("properties", {})
            try:
                lat, lon = float(props.get("lat")), float(props.get("lon"))
            except (TypeError, ValueError):
                continue
            # Hard geographic guardrail for local searches: discard results well
            # outside Gwalior rather than merely ranking them lower.
            if is_gwalior_local and not (25.95 <= lat <= 26.55 and 77.75 <= lon <= 78.55):
                continue

            name = props.get("name") or props.get("address_line1") or ""
            address = props.get("formatted") or props.get("address_line1") or original
            fields = ["name", "address_line1", "address_line2", "street", "suburb", "city", "county", "state", "formatted"]
            searchable = " ".join(str(props.get(k) or "") for k in fields).lower()
            matched_tokens = [token for token in query_tokens if token in searchable]
            score = len(matched_tokens)
            if name and original.lower() in name.lower():
                score += 5
            if is_gwalior_local and "gwalior" in searchable:
                score += 2
            # Local landmark/venue searches need at least one distinctive word
            # from the requested name to be present in the returned result.
            if query_tokens and score <= 0:
                continue
            place_type = props.get("result_type") or props.get("type") or "location"
            label = address
            if name and name.lower() not in address.lower():
                label = f"{name}, {address}"
            item = {"label": label, "latitude": lat, "longitude": lon, "address": address,
                    "name": name, "place_type": place_type, "score": score}
            key = (round(lat, 6), round(lon, 6))
            previous = results_by_key.get(key)
            if previous is None or item["score"] > previous["score"]:
                results_by_key[key] = item

        results = list(results_by_key.values())
        results.sort(key=lambda item: (-item["score"], item["label"].lower()))
        return results[:limit]
    except Exception as error:
        print("Geoapify location search failed:", repr(error))
        st.warning("Location search could not connect to Geoapify. Please try again.")
        return []


def reverse_geocode_geoapify(latitude, longitude):
    """Return a road/address label for a manually selected map point."""
    try:
        api_key = st.secrets.get("GEOAPIFY_API_KEY", "").strip()
    except Exception:
        api_key = ""
    if not api_key:
        return "Manual map location"
    params = {"lat": latitude, "lon": longitude, "format": "json", "apiKey": api_key, "lang": "en"}
    url = "https://api.geoapify.com/v1/geocode/reverse?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "RoadGuardAI_RoadHazard_Project/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        props = payload.get("results", [{}])[0]
        return props.get("name") or props.get("street") or props.get("formatted") or "Manual map location"
    except Exception as exc:
        print("Reverse geocoding failed:", repr(exc))
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
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #ffffff;
    }
    .subtitle {
        color: #aab4c3;
        font-size: 16px;
    }
    div[data-testid="stMetric"] {
        background-color: #1b2230;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #30394a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# SIDEBAR
# =========================
st.sidebar.title("🛣️ RoadGuard AI")
st.sidebar.caption("AI Road Hazard Detection System")

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
st.markdown('<div class="main-title">RoadGuard AI</div>', unsafe_allow_html=True)
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
            {"Report ID": r.get("id", "—"), "Date": r.get("report_date", "—"),
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
    st.subheader("🔍 AI Road Hazard Detection")
    st.write("Upload a road image to detect potholes, cracks and other road surface damage.")

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

        st.markdown("## 📊 Analysis Summary")
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

        if "geoapify_location_results" not in st.session_state:
            st.session_state.geoapify_location_results = []
        if "location_map_revision" not in st.session_state:
            st.session_state.location_map_revision = 0
        if "last_geoapify_selected_label" not in st.session_state:
            st.session_state.last_geoapify_selected_label = None

        if st.button("📍 Search Locations", use_container_width=True):
            if search_place.strip():
                with st.spinner("Searching location with Geoapify..."):
                    st.session_state.geoapify_location_results = search_locations_geoapify(search_place.strip())
                st.session_state.pop("geoapify_location_selection", None)
                st.session_state.last_geoapify_selected_label = None
                st.session_state.pop("last_manual_map_click", None)
                st.session_state.location_map_revision += 1
                if st.session_state.geoapify_location_results:
                    st.success(f"Found {len(st.session_state.geoapify_location_results)} location result(s). Select a result below.")
                else:
                    st.warning("No results found. Try adding city/state to your search.")
            else:
                st.warning("Please enter a location first.")

        search_results = st.session_state.geoapify_location_results
        if search_results:
            result_labels = [f"{i + 1}. {item['label']}" for i, item in enumerate(search_results)]
            selected_label = st.selectbox(
                "Choose the correct location",
                result_labels,
                key="geoapify_location_selection",
                help="Select a result to move the map to that location. You can then click the map to refine it.",
            )
            if selected_label != st.session_state.last_geoapify_selected_label:
                selected_index = result_labels.index(selected_label)
                selected_place = search_results[selected_index]
                st.session_state.latitude = float(selected_place["latitude"])
                st.session_state.longitude = float(selected_place["longitude"])
                st.session_state.location_name = selected_place["label"]
                st.session_state.report_location_name = selected_place["label"]
                st.session_state.last_geoapify_selected_label = selected_label
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
        map_data = st_folium(
            location_map,
            width=900,
            height=450,
            key=f"report_location_map_{st.session_state.location_map_revision}",
        )

        # Manual map click refines the selected search result or default location.
        if map_data and map_data.get("last_clicked"):
            clicked = map_data["last_clicked"]
            click_signature = (round(float(clicked["lat"]), 7), round(float(clicked["lng"]), 7))
            if st.session_state.get("last_manual_map_click") != click_signature:
                st.session_state.last_manual_map_click = click_signature
                st.session_state.latitude = float(clicked["lat"])
                st.session_state.longitude = float(clicked["lng"])
                clicked_name = reverse_geocode_geoapify(float(clicked["lat"]), float(clicked["lng"]))
                st.session_state.location_name = clicked_name
                st.session_state.report_location_name = clicked_name
                st.session_state.location_map_revision += 1
                st.rerun()

        st.markdown("### 📌 Selected Place")
        st.info(st.session_state.get("location_name") or st.session_state.get("report_location_name") or "Map point selected — click the map to fine-tune the location.")
        st.markdown("### 📌 Location Coordinates")
        col1, col2 = st.columns(2)
        col1.metric("Latitude", f"{st.session_state.latitude:.6f}")
        col2.metric("Longitude", f"{st.session_state.longitude:.6f}")

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
            )
            st_folium(report_map, width=1000, height=500, key="filtered_reports_map")

            # ---------- Report table ----------
            st.markdown("### 📋 Report History")
            reports_df = pd.DataFrame([
                {
                    "Report ID": r.get("id"),
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
                    st_folium(detail_map, width=1000, height=400, key=f"detail_map_{selected_id}")
                except (ValueError, TypeError):
                    st.warning("Unable to display the map for this report.")
            else:
                st.warning("Location coordinates are not available for this report.")


            # ---------- Report image ----------
            st.markdown("### 🖼️ Report Image")
            image_name = selected_report.get("image_name")
            if image_name:
                image_path = Path("uploads") / str(image_name)
                if image_path.exists():
                    st.image(str(image_path), caption=f"Image for Report #{selected_id}", use_container_width=True)
                else:
                    st.info(f"Image filename: {image_name}, but the image file was not found in uploads folder.")
            else:
                st.info("No image filename stored for this report.")


# =========================
# ABOUT PROJECT
# =========================
elif page == "ℹ️ About Project":
    st.subheader("ℹ️ About RoadGuard AI")
    st.write(
        """
        RoadGuard AI is an AI-based road hazard detection and geo-tagged reporting application.
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
