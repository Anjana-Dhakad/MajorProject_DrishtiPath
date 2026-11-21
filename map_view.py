
import folium
from streamlit_folium import st_folium


def create_location_map(
    latitude=22.9734,
    longitude=78.6569,
    reports=None,
    selection_mode=True
):

    # Zoom closer when selecting a location
    zoom = 15 if selection_mode else 5

    road_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=zoom,
        control_scale=True
    )

    # Saved report markers
    if reports:
        for report in reports:

            if report.get("latitude") is None or report.get("longitude") is None:
                continue

            folium.Marker(
                location=[
                    report["latitude"],
                    report["longitude"]
                ],
                popup=f"""
                Report #{report['id']}<br>
                Hazards: {report['total_hazards']}<br>
                Severity: {report['severity']}<br>
                Location: {report.get('location_name', 'Unknown')}
                """,
                tooltip=f"Report #{report['id']}",
                icon=folium.Icon(
                    color="red" if report["severity"] == "High"
                    else "orange" if report["severity"] == "Medium"
                    else "blue",
                    icon="warning",
                    prefix="fa"
                )
            ).add_to(road_map)

    # Current selected location
    if selection_mode:
        folium.Marker(
            location=[latitude, longitude],
            tooltip="Selected Hazard Location",
            popup="Selected reporting location",
            icon=folium.Icon(
                color="green",
                icon="map-pin",
                prefix="fa"
            )
        ).add_to(road_map)

    return road_map
