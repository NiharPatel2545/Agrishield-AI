from __future__ import annotations

import requests

from agrishield.config import OPENWEATHER_API_KEY


def current_weather(latitude: float, longitude: float) -> dict:
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "your-openweather-key":
        raise ValueError("OPENWEATHER_API_KEY is not set in .env")

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "lat": latitude,
            "lon": longitude,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "temp_c": data.get("main", {}).get("temp"),
        "humidity": data.get("main", {}).get("humidity"),
        "pressure": data.get("main", {}).get("pressure"),
        "wind_speed": data.get("wind", {}).get("speed"),
        "rain_1h_mm": data.get("rain", {}).get("1h", 0),
        "description": (data.get("weather") or [{}])[0].get("description"),
    }
