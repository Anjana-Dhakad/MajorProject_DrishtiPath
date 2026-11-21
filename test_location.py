from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut

geolocator = Nominatim(
    user_agent="RoadGuardAI_RoadHazard_Project"
)

place = "Purani Chawani, Gwalior, Madhya Pradesh, India"

try:
    print("Searching location...")

    result = geolocator.geocode(place, timeout=15)

    if result:
        print("\nLocation found!")
        print("Address:", result.address)
        print("Latitude:", result.latitude)
        print("Longitude:", result.longitude)
    else:
        print("\nLocation returned None.")

except Exception as e:
    print("\nERROR:", repr(e))