import pandas as pd
import requests
import json
import os
import datetime

# URL
url = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"

# Pridobivanje podatkov
response = requests.get(url)

# Preverjanje zahtevka
if response.status_code == 200:
    # json format
    data = response.json()

    # Pretvorba v DataFrame
    df_new = pd.DataFrame(data)

    # Pretvorba datetime v GMT+2 (CET)
    df_new['last_update'] = pd.to_datetime(df_new['last_update'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

    # Filtriranje podatkov za postajališče "GOSPOSVETSKA C. - TURNERJEVA UL."
    df_filtered = df_new[df_new['name'] == 'GOSPOSVETSKA C. - TURNERJEVA UL.']

    df_filtered['last_update'] = pd.to_datetime(df_filtered['last_update'], unit='ms').dt.tz_convert('Europe/Berlin')



    # Pot do datoteke
    file_path = 'data/raw/api_data.json'

    # Če datoteka že obstaja in ni prazna, preberi obstoječe podatke
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as f:
            existing_data = json.load(f)

        # Pretvorba obstoječih podatkov v DataFrame
        df_existing = pd.DataFrame(existing_data)

        # Pretvorba datetime v GMT+2 (CET) za obstoječe podatke
        df_existing['last_update'] = pd.to_datetime(df_existing['last_update']).dt.tz_convert('Europe/Berlin')

        # Dodaj nove podatke k obstoječim
        df_combined = pd.concat([df_existing, df_filtered], ignore_index=True)

    else:
        df_combined = df_filtered

    # Pretvorba stolpca 'last_update' v želeni format (ISO 8601) pred shranjevanjem v JSON
    df_combined['last_update'] = df_combined['last_update'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')

    # Shranjevanje vseh podatkov v datoteko
    with open(file_path, 'w') as f:
        f.write(df_combined.to_json(orient='records', indent=4))

    print("Podatki uspešno dodani.")





url = "https://api.open-meteo.com/v1/forecast"

# Določitev trenutnega časa
current_time = datetime.datetime.now()

# Začetni čas za pridobivanje podatkov (3 dni nazaj)
start_time = current_time + datetime.timedelta(days=3)

# Priprava parametrov za zahtevo
params = {
    "latitude": 46.562695,
    "longitude": 15.62935,
    "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability"],
    "timezone": "Europe/Berlin",
    "past_days": 30,
    "forecast_days": 2
}

# Pridobivanje podatkov
response = requests.get(url, params=params)


# Preverjanje zahtevka
if response.status_code == 200:
    # Pretvorba v DataFrame
    data = response.json()

    # Shranjevanje podatkov v mapi (data/raw/weather) v lepši obliki
    weather_data_path = 'data/raw/Weather/weather_data.json'
    with open(weather_data_path, 'w') as f:
        json.dump(data, f, indent=4)

    print("Zgodovinski podatki o vremenu uspešno pridobljeni in shranjeni.")
else:
    print("Napaka pri pridobivanju podatkov o vremenu.")