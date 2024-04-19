
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from flask import Flask, request
from flask import jsonify


# Pot do shranjenega modela
model_filename = "/app/naloga01_model01.h5"

# Uvoz modela
model = load_model(model_filename)

# Pot do shranjenega scalerja
scaler_filename = "/app/naloga01_scaler01.pkl"

# Uvoz scalerja
scaler = joblib.load(scaler_filename)


# Pot do shranjenega scalerja
scaler_filename1 = "/app/naloga01_scaler02.pkl"

# Uvoz scalerja
scaler1 = joblib.load(scaler_filename1)




app = Flask(__name__)




def pripravi_podatke_za_ucenje(vrednosti, okno_velikost):
    X = []
    for i in range(len(vrednosti) - okno_velikost + 1):
        X.append(vrednosti[i:i+okno_velikost, :])

    return np.array(X)





@app.route('/predict/naloga01', methods=['POST'])
def post_example():
    json_data = request.get_json()

    df = pd.DataFrame(json_data)
    stevilo_vrstic = df.shape[0]
    print(stevilo_vrstic)

    print(df.isnull().sum())


    # Filtriranje značilnic
    najdoprinosne_znacilnice = ['temperature', 'relative_humidity', 'apparent_temperature', 'dew_point']
    ciljna_znacilnica = 'available_bike_stands'
    podatki = df[najdoprinosne_znacilnice + [ciljna_znacilnica]]

    podatki_standardized = scaler.fit_transform(podatki[['temperature', 'relative_humidity', 'apparent_temperature', 'dew_point']])

    podatki_standardized1 = scaler1.fit_transform(podatki[['available_bike_stands']])

    podatki_standardized = podatki_standardized + podatki_standardized1


    test_data = podatki_standardized
    stevilo = test_data.shape[0]
    print("Število vrstic: ",stevilo)


    okno_velikost = 60

    X_test = pripravi_podatke_za_ucenje(test_data, okno_velikost)
    stevilo_podatkov = X_test.shape[0]
    print(stevilo_podatkov)

    y_pred = model.predict(X_test)

    y_pred_unscaled = scaler1.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return jsonify({"prediction": float(y_pred_unscaled[0])})



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0')

















