import json
import html
import streamlit as st

DEFAULT_LATITUDE = 26.2183
DEFAULT_LONGITUDE = 78.1828


def _get_google_maps_key():
    try:
        return str(st.secrets.get("GOOGLE_MAPS_API_KEY", "")).strip()
    except Exception:
        return ""


def create_location_map(latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE, reports=None, selection_mode=True, selected_location_name=None, show_nearby=True, nearby_radius=700):
    lat, lon = float(latitude), float(longitude)
    key = _get_google_maps_key()
    if not key:
        return '<div style="padding:16px;color:#b91c1c;background:#fff7f7;border:1px solid #c33;border-radius:8px"><b>Google Maps key missing.</b> Add GOOGLE_MAPS_API_KEY to .streamlit/secrets.toml and restart Streamlit.</div>'

    pins = []
    for report in reports or []:
        try:
            rlat, rlon = float(report.get("latitude")), float(report.get("longitude"))
        except (TypeError, ValueError):
            continue
        pins.append({
            "lat": rlat,
            "lng": rlon,
            "title": f"Report #{report.get('id', 'N/A')} | {report.get('severity', 'Unknown')} severity",
            "details": f"Hazards: {report.get('total_hazards', 'N/A')} | Location: {report.get('location_name', 'Unknown')}",
        })

    pins_json = json.dumps(pins, ensure_ascii=False).replace("</", "<\\/")
    name_json = json.dumps(str(selected_location_name or "Selected hazard location"), ensure_ascii=False).replace("</", "<\\/")
    select_mode = "true" if selection_mode else "false"

    html_doc = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>html,body,#map{height:100%;width:100%;margin:0;font-family:Arial,sans-serif}#coords{position:absolute;bottom:12px;left:12px;z-index:5;background:white;padding:9px;border-radius:7px;box-shadow:0 2px 8px #0003;font-size:12px;color:#202124}#copy{border:0;background:#1a73e8;color:white;border-radius:4px;padding:5px 8px;margin-left:6px;cursor:pointer}</style></head><body><div id="map"></div><div id="coords">Click map for coordinates: <span id="coordtext">LAT_INIT, LNG_INIT</span><button id="copy" onclick="copyCoords()">Copy</button></div><script>let map,selectedMarker;const initial={center};const pins=REPORT_PINS;const selectMode=SELECT_MODE;const selectedName=SELECT_NAME;function initMap(){map=new google.maps.Map(document.getElementById('map'),{center:initial,zoom:selectMode?17:14,mapTypeControl:true,mapTypeControlOptions:{mapTypeIds:['roadmap','satellite','hybrid','terrain']},streetViewControl:true,fullscreenControl:true,clickableIcons:true,gestureHandling:'greedy'});if(selectMode){selectedMarker=new google.maps.Marker({position:initial,map,title:selectedName,draggable:true,icon:{url:'https://maps.google.com/mapfiles/ms/icons/green-dot.png'}});selectedMarker.addListener('dragend',e=>setCoords(e.latLng));map.addListener('click',e=>{selectedMarker.setPosition(e.latLng);setCoords(e.latLng);});}pins.forEach(p=>{let marker=new google.maps.Marker({position:{lat:p.lat,lng:p.lng},map,title:p.title});let info=new google.maps.InfoWindow({content:'<b>'+esc(p.title)+'</b><br>'+esc(p.details)});marker.addListener('click',()=>info.open({map,anchor:marker}));});}function setCoords(ll){document.getElementById('coordtext').textContent=ll.lat().toFixed(6)+', '+ll.lng().toFixed(6);}function copyCoords(){navigator.clipboard.writeText(document.getElementById('coordtext').textContent).then(()=>{document.getElementById('copy').textContent='Copied!';setTimeout(()=>document.getElementById('copy').textContent='Copy',1000);}).catch(()=>{document.getElementById('copy').textContent='Copy failed';});}function esc(s){return String(s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));}</script><script async defer src="https://maps.googleapis.com/maps/api/js?key=GOOGLE_KEY&callback=initMap"></script></body></html>"""
    html_doc = (
        html_doc.replace("LAT_INIT", f"{lat:.6f}")
        .replace("LNG_INIT", f"{lon:.6f}")
        .replace("REPORT_PINS", pins_json)
        .replace("SELECT_MODE", select_mode)
        .replace("SELECT_NAME", name_json)
        .replace("GOOGLE_KEY", html.escape(key))
        .replace("{center}", json.dumps({"lat": lat, "lng": lon}))
    )
    return html_doc
