import pandas as pd
import json
import os
import requests

# URL
url = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"

# Pridobivanje podatkov
response = requests.get(url)

# Preverjanje zahtevka
if response.status_code == 200:
    # json format
    data = response.json()



    # Pretvorba
    df = pd.DataFrame(data)

    # Pretvorba datetime v centralno evropski čas (CET)
    df['last_update'] = pd.to_datetime(df['last_update'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('CET')


    # Shranjevanje
    output_directory = 'C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/data/processed'
    os.makedirs(output_directory, exist_ok=True)

    # Za vsako postajališče, obdelava in shranjevanje v CSV
    for name, group in df.groupby('name'):
        filename = os.path.join(output_directory, f"{name.replace(' ', '_').lower()}.csv")

        if os.path.exists(filename):
            # Preberi obstoječi DataFrame iz datoteke
            existing_df = pd.read_csv(filename)

            # Dodaj nove vrstice, če obstajajo
            existing_df = pd.concat([existing_df, group], ignore_index=False)

            # Shranimo nazaj v isto datoteko CSV
            existing_df.to_csv(filename, index=False)
        else:
            # Če datoteke še ni, preprosto shranimo trenutni podatkovni okvir
            group.to_csv(filename, index=False)
else:
    print("Napaka pri pridobivanju podatkov. Koda napake:", response.status_code)
