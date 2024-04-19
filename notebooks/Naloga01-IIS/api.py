import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from flask import Flask, request, jsonify
import requests
import io
from flask_cors import CORS

# Pot do shranjenega modela
model_filename = "C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/models/naloga02_model.h5"
# Uvoz modela
model = load_model(model_filename)

# Pot do shranjenega scalerja
scaler_filename = "C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/models/naloga02_scaler1.pkl"
# Uvoz scalerja
scaler = joblib.load(scaler_filename)

# Pot do shranjenega scalerja
scaler_filename1 = "C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/models/naloga02_scaler2.pkl"
# Uvoz scalerja
scaler1 = joblib.load(scaler_filename1)

app = Flask(__name__)
CORS(app)

def pripravi_podatke_za_ucenje(vrednosti, okno_velikost):
    X = []
    for i in range(len(vrednosti) - okno_velikost + 1):
        X.append(vrednosti[i:i+okno_velikost, :])
    return np.array(X)

@app.route('/predict/naloga02', methods=['GET'])
def post_example():
    github_url = "https://raw.githubusercontent.com/PetelinekBenjamin/StrojnoUcenje2/master/data/processed/GOSPOSVETSKA%20C%20-%20TURNERJEVA%20UL.csv"
    response = requests.get(github_url)
    if response.status_code == 200:
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)
    else:
        print('Napaka pri pridobivanju podatkov:', response.status_code)
        return jsonify({"error": "Napaka pri pridobivanju podatkov"})

    # Filtriranje značilnic
    najdoprinosne_znacilnice = ['precipitation_probability', 'dew_point_2m', 'apparent_temperature', 'relative_humidity_2m', 'temperature_2m']
    ciljna_znacilnica = 'available_bike_stands'
    podatki = df[najdoprinosne_znacilnice + [ciljna_znacilnica]]

    podatki_standardized = scaler.fit_transform(podatki[['precipitation_probability', 'dew_point_2m', 'apparent_temperature', 'relative_humidity_2m', 'temperature_2m']])
    podatki_standardized1 = scaler1.fit_transform(podatki[['available_bike_stands']])
    podatki_standardized = podatki_standardized + podatki_standardized1

    test_data = podatki_standardized

    okno_velikost = 5
    X_test = pripravi_podatke_za_ucenje(test_data, okno_velikost)
    stevilo_podatkov = X_test.shape[0]

    # Napovedovanje za zadnjih 7 iteracij
    y_pred = []
    for i in range(stevilo_podatkov - 7, stevilo_podatkov):
        pred = model.predict(X_test[i:i+1])
        y_pred.append(pred)

    y_pred_unscaled = scaler1.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()

    rounded_y_pred = [round(num, 1) for num in y_pred_unscaled.tolist()]



    return jsonify({"prediction": rounded_y_pred})


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0')
