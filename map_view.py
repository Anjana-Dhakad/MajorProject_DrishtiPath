import folium
from streamlit_folium import st_folium
import streamlit as st

# Default center: Gwalior, Madhya Pradesh
DEFAULT_LATITUDE = 26.2183
DEFAULT_LONGITUDE = 78.1828


def _get_geoapify_key():
    """Safely read the Geoapify key from Streamlit secrets."""
    try:
        return str(st.secrets.get("GEOAPIFY_API_KEY", "")).strip()
    except Exception:
        return ""


def create_location_map(
    latitude=DEFAULT_LATITUDE,
    longitude=DEFAULT_LONGITUDE,
    reports=None,
    selection_mode=True,
    selected_location_name=None,
):
    """Create a labeled street map with report markers and a selected hazard point."""
    zoom = 18 if selection_mode else 15
    road_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=zoom,
        control_scale=True,
        max_zoom=20,
        scrollWheelZoom=True,
        tiles=None,
    )

    # Prefer Geoapify's OSM-Carto tiles (road names, street detail and POI labels).
    # Use public OpenStreetMap tiles if the Geoapify key is missing.
    tile_key = _get_geoapify_key()
    if tile_key:
        folium.TileLayer(
            tiles=f"https://maps.geoapify.com/v1/tile/osm-carto/{{z}}/{{x}}/{{y}}.png?apiKey={tile_key}",
            attr='© OpenStreetMap contributors, © Geoapify',
            name="Street Map (Roads & Labels)",
            max_zoom=20,
            overlay=False,
            control=True,
        ).add_to(road_map)
    else:
        folium.TileLayer(
            tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr='© OpenStreetMap contributors',
            name="OpenStreetMap (Roads & Labels)",
            max_zoom=19,
            overlay=False,
            control=True,
        ).add_to(road_map)

    # Saved report markers
    if reports:
        for report in reports:
            report_lat = report.get("latitude")
            report_lon = report.get("longitude")
            if report_lat is None or report_lon is None:
                continue

            severity = report.get("severity", "Low")
            marker_color = "red" if severity == "High" else "orange" if severity == "Medium" else "blue"
            report_id = report.get("id", "N/A")
            location_name = report.get("location_name", "Unknown")
            popup_text = (
                f"<b>Report #{report_id}</b><br>"
                f"Hazards: {report.get('total_hazards', 'N/A')}<br>"
                f"Severity: {severity}<br>"
                f"Location: {location_name}"
            )
            folium.Marker(
                location=[report_lat, report_lon],
                popup=folium.Popup(popup_text, max_width=380),
                tooltip=f"Report #{report_id} | {severity} | {location_name}",
                icon=folium.Icon(color=marker_color, icon="warning", prefix="fa"),
            ).add_to(road_map)

    # Selected hazard marker: show place name visibly beside the pin, not only on hover.
    if selection_mode:
        place_name = (selected_location_name or "Selected hazard location").strip()
        folium.Marker(
            location=[latitude, longitude],
            tooltip=place_name,
            popup=folium.Popup(
                f"<b>Selected Place</b><br>{place_name}<br><br>"
                "Click another point on the map to fine-tune the exact hazard location.",
                max_width=380,
            ),
            icon=folium.Icon(color="green", icon="map-pin", prefix="fa"),
        ).add_to(road_map)

        # Persistent text label adjacent to the selected pin.
        folium.Marker(
            location=[latitude, longitude],
            icon=folium.DivIcon(
                html=(
                    '<div style="display:inline-block; white-space:nowrap; '
                    'background:#ffffff; color:#166534; border:1px solid #16a34a; '
                    'border-radius:5px; padding:3px 7px; font-size:12px; '
                    'font-weight:700; box-shadow:0 1px 4px rgba(0,0,0,.25);">'
                    f'{place_name}</div>'
                ),
                icon_size=(220, 30),
                icon_anchor=(-12, 18),
            ),
        ).add_to(road_map)

        folium.CircleMarker(
            location=[latitude, longitude],
            radius=7,
            color="#16a34a",
            fill=True,
            fill_color="#16a34a",
            fill_opacity=0.25,
            weight=2,
            tooltip="Selected coordinate",
        ).add_to(road_map)

    folium.LayerControl(collapsed=True).add_to(road_map)
    return road_map
