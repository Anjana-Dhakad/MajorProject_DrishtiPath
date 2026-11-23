import requests
import streamlit as st

place = "Purani Chawani, Gwalior, Madhya Pradesh, India"

# API key from .streamlit/secrets.toml
try:
    API_KEY = st.secrets["GEOAPIFY_API_KEY"]
except Exception:
    try:
        API_KEY = st.secrets["geoapify"]["api_key"]
    except Exception:
        print("API key not found in secrets.toml")
        raise

url = "https://api.geoapify.com/v1/geocode/search"

params = {
    "text": place,
    "apiKey": API_KEY,
    "limit": 5,
    "filter": "countrycode:in",
    "lang": "en"
}

try:
    print("Searching location using Geoapify...")

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()
    results = data.get("features", [])

    if results:
        print("\nLocation found!")

        for i, result in enumerate(results, start=1):
            properties = result["properties"]
            coordinates = result["geometry"]["coordinates"]

            longitude = coordinates[0]
            latitude = coordinates[1]

            print(f"\nResult {i}")
            print("Address:", properties.get("formatted"))
            print("Latitude:", latitude)
            print("Longitude:", longitude)
            print("Place Type:", properties.get("result_type"))

    else:
        print("\nNo location results found.")

except requests.exceptions.HTTPError as e:
    print("\nHTTP ERROR:", e)
    print("Response:", response.text)

except Exception as e:
    print("\nERROR:", repr(e))