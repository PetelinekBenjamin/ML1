import pandas as pd
import requests
import json
import os

def fetch_weather_data():
    # API za pridobivanje zgodovinskih podatkov o vremenu
    # Prilagodite URL in parametre za svoje potrebe
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 46.562695,
        "longitude": 15.62935,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability"],
        "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min", "sunrise", "sunset", "daylight_duration", "sunshine_duration", "uv_index_max", "uv_index_clear_sky_max", "precipitation_sum", "rain_sum", "showers_sum", "snowfall_sum", "precipitation_hours", "precipitation_probability_max", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"],
        "timezone": "Europe/Berlin",
        "past_days": 3,
        "forecast_days": 0
    }

    # Pridobivanje podatkov
    response = requests.get(url, params=params)

    # Preverjanje zahtevka
    if response.status_code == 200:
        # Pretvorba v DataFrame
        data = response.json()

        # Shranjevanje podatkov v mapi (data/raw/weather) v lepši obliki
        weather_data_path = 'C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/data/raw/Weather/weather_data.json'
        with open(weather_data_path, 'w') as f:
            json.dump(data, f, indent=4)

        print("Zgodovinski podatki o vremenu uspešno pridobljeni in shranjeni.")
    else:
        print("Napaka pri pridobivanju podatkov o vremenu.")

if __name__ == "__main__":
    fetch_weather_data()
