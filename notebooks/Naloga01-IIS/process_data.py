import pandas as pd
import json


# JSON datoteka
with open('C:/Users/benja/Desktop/Stuff/Šola/Strojno ucenje2/data/raw/api_data.json', 'r') as f:
    data = json.load(f)

# DataFrame
df = pd.DataFrame(data)

# Pretvorba v datetime
df['last_update'] = pd.to_datetime(df['last_update'], unit='ms')


# Grupiranje po urnem intervalu
hourly_grouped = df.groupby(pd.Grouper(key='last_update', freq='H')).count()

# Filtriranje podatkov
filtered_data = [station for station in data if station["name"] == "GOSPOSVETSKA C. - TURNERJEVA UL."]

# Shranjevanje podatkov
pot_do_procesiranih_podatkov = 'C:/Users/benja/Desktop/Stuff/Šola/Strojno ucenje2/data/processed/hourly_data.csv'
hourly_grouped.to_csv(pot_do_procesiranih_podatkov)

