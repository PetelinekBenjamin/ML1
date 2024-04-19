import pandas as pd
import json
import datetime
import pytz

# Poti do datotek
station_data_path = 'data/raw/api_data.json'
weather_data_path = 'data/raw/Weather/weather_data.json'

# Naloži podatke o postajališču
with open(station_data_path, 'r') as f:
    station_data = json.load(f)

# Pretvori podatke o postajališču v dataframe
df_station = pd.DataFrame(station_data)

# Naloži podatke o vremenu
with open(weather_data_path, 'r') as f:
    weather_data = json.load(f)

# Pretvori podatke o vremenu v dataframe
df_weather = pd.DataFrame(weather_data['hourly'])

# Pretvori čas iz timestamp formata v datetime objekte
df_weather['time'] = pd.to_datetime(df_weather['time'])
df_station['last_update'] = pd.to_datetime(df_station['last_update'])

# Pretvori nize vrednosti v stolpcih s temperaturo v številske vrednosti
df_weather['temperature_2m'] = df_weather['temperature_2m'].astype(float)
df_weather['dew_point_2m'] = df_weather['dew_point_2m'].astype(float)
df_weather['apparent_temperature'] = df_weather['apparent_temperature'].astype(float)

# Iteriraj skozi vsako postajališče
for index, row in df_station.iterrows():
    print("Postajališče:", row["name"])
    print("Število koles:", row["available_bikes"])
    print("Število stojal:", row["bike_stands"])
    print("Naslov:", row["address"])
    print("Položaj (latitude, longitude):", row["position"]["lat"], ",", row["position"]["lng"])
    print("Status:", row["status"])
    print("Zadnja posodobitev:", row["last_update"])

    # Pretvori čas zadnje posodobitve v objekt datetime
    last_update_datetime = pd.to_datetime(row["last_update"]).tz_localize(None)  # Remove timezone info

    # Poiščite najbližji čas v seznamu
    closest_time = min(df_weather['time'], key=lambda x: abs(x - last_update_datetime))

    # Poiščite indeks najbližjega časa
    indexTime = df_weather.index[df_weather['time'] == closest_time][0]

    # Uporabi indeks, da dobimo vrednosti metrik za najbližji čas
    temperature = df_weather.at[indexTime, 'temperature_2m']
    humidity = df_weather.at[indexTime, 'relative_humidity_2m']
    dew_point = df_weather.at[indexTime, 'dew_point_2m']
    apparent_temperature = df_weather.at[indexTime, 'apparent_temperature']
    precipitation_probability = df_weather.at[indexTime, 'precipitation_probability']

    df_station.at[index, 'temperature_2m'] = temperature
    df_station.at[index, 'relative_humidity_2m'] = humidity
    df_station.at[index, 'dew_point_2m'] = dew_point
    df_station.at[index, 'apparent_temperature'] = apparent_temperature
    df_station.at[index, 'precipitation_probability'] = precipitation_probability

    # Navedite pot in ime datoteke za shranjevanje
    csv_file_path = 'data/processed/GOSPOSVETSKA C - TURNERJEVA UL.csv'

    # Uporabite metodo to_csv() za shranjevanje DataFrame kot CSV
    df_station.to_csv(csv_file_path, index=False)  # Nastavite index=False, če ne želite shraniti indeksa




    # Natisni vrednosti metrik
    print("Metrike vremena za", closest_time)
    print("Temperatura:", temperature, "°C")
    print("Relativna vlaga:", humidity, "%")
    print("Rosna točka:", dew_point, "°C")
    print("Občutna temperatura:", apparent_temperature, "°C")
    print("Verjetnost padavin:", precipitation_probability, "%")
    print("\n")












