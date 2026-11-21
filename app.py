import io
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

from detector import detect_hazards
from severity import calculate_severity
from database import (
    initialize_database,
    save_report,
    get_all_reports,
    get_report_statistics,
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
    "latitude": 22.9734,
    "longitude": 78.6569,
    "location_name": "",
    "report_location_name": "",
}
for state_key, default_value in DEFAULT_STATE.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value

def search_location(place):
    if not place or not place.strip():
        return None

    try:
        geolocator = Nominatim(
            user_agent="RoadGuardAI_RoadHazard_Project"
        )

        result = geolocator.geocode(
            place.strip(),
            timeout=15,
            exactly_one=True,
            country_codes="in"
        )

        if result is None:
            return None

        return (
            result.latitude,
            result.longitude,
            result.address
        )

    except Exception as e:
        print("Location search failed:", repr(e))
        return None

    # Fallback to Photon only if Nominatim fails
    try:
        geolocator = Photon(
            user_agent="roadguard_ai_app_2026"
        )

        result = geolocator.geocode(
            f"{place}, India",
            timeout=15,
            exactly_one=True
        )

        if result:
            address = result.address or place

            # Avoid accepting an obviously unrelated result
            searched_words = [
                w.lower() for w in place.split()
                if len(w) > 2
            ]

            returned_address = address.lower()

            if any(word in returned_address for word in searched_words):
                return (
                    result.latitude,
                    result.longitude,
                    address
                )

    except Exception as e:
        print("Photon fallback error:", e)

    return None

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
    st.subheader("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reports", stats["total_reports"])
    col2.metric("Total Hazards", stats["total_hazards"])
    col3.metric("High Severity", stats["high"])
    col4.metric("AI Model", "YOLOv8", "Active")

    st.markdown("### 📈 Severity Statistics")
    severity_df = pd.DataFrame(
        {
            "Severity": ["High", "Medium", "Low"],
            "Reports": [stats["high"], stats["medium"], stats["low"]],
        }
    )
    st.bar_chart(severity_df.set_index("Severity"))
    st.info(
        "Use Detect Road Hazards to analyze a road image and create a geo-tagged report."
    )


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
        # LOCATION REPORTING
        # =========================
        st.divider()
        st.markdown("## 📍 Geo-Tagged Reporting")
        st.write(
            "Search a road, area, landmark or address. The map will automatically move to the searched location."
        )

        search_place = st.text_input(
            "🔎 Search Location",
            placeholder="Example: City Center, Gwalior, Madhya Pradesh",
            key="location_search_input",
        )

        if st.button("📍 Find Location", use_container_width=True):
            if search_place.strip():
                with st.spinner("Searching location..."):
                    result = search_location(search_place.strip())
                if result:
                    lat, lon, address = result
                    st.session_state.latitude = lat
                    st.session_state.longitude = lon
                    st.session_state.location_name = address
                    st.session_state.report_location_name = address
                    st.success("Location found successfully!")
                else:
                    st.error("Location not found or geocoding service unavailable. Try a more specific address or landmark.")
            else:
                st.warning("Please enter a location first.")

        reports = get_all_reports()
        location_map = create_location_map(
            latitude=st.session_state.latitude,
            longitude=st.session_state.longitude,
            reports=reports,
            selection_mode=True,
        )
        st.markdown("### 🗺️ Selected Location")
        map_data = st_folium(
            location_map,
            width=900,
            height=450,
            key="report_location_map",
        )

        # Manual map click can refine the searched location
        if map_data and map_data.get("last_clicked"):
            clicked = map_data["last_clicked"]
            st.session_state.latitude = clicked["lat"]
            st.session_state.longitude = clicked["lng"]

        st.markdown("### 📌 Location Coordinates")
        col1, col2 = st.columns(2)
        col1.metric("Latitude", f"{st.session_state.latitude:.6f}")
        col2.metric("Longitude", f"{st.session_state.longitude:.6f}")

        location_name = st.text_input(
            "Location / Road Name",
            key="report_location_name",
        )

        # =========================
        # SAVE REPORT
        # =========================
        if st.button("💾 Save Hazard Report", use_container_width=True):
            if not st.session_state.image_name:
                st.warning("Please analyze an image before saving a report.")
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

    if reports:
        st.markdown("### 🗺️ Report Locations")
        report_map = create_location_map(
            latitude=22.9734,
            longitude=78.6569,
            reports=reports,
            selection_mode=False,
        )
        st_folium(report_map, width=1000, height=500, key="saved_reports_map")

        st.markdown("### 📋 Report History")
        reports_df = pd.DataFrame(
            [
                {
                    "Report ID": r["id"],
                    "Date": r["report_date"],
                    "Location": r["location_name"],
                    "Hazards": r["total_hazards"],
                    "Severity": r["severity"],
                    "Latitude": r["latitude"],
                    "Longitude": r["longitude"],
                }
                for r in reports
            ]
        )
        st.dataframe(reports_df, use_container_width=True, hide_index=True)
        csv_data = reports_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Reports CSV",
            data=csv_data,
            file_name="road_hazard_reports.csv",
            mime="text/csv",
        )
    else:
        st.info("No reports saved yet. Detect a road hazard and save its report to see it here.")


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
        - Geopy (location search)
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
