#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold 
from tensorflow.keras.optimizers import SGD
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from PIL import Image
import matplotlib.image as mpimg
import plotly.graph_objects as go
from keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import STL
from sklearn.feature_selection import f_regression
from sklearn.svm import SVR
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from keras.models import load_model
from flask import Flask, request
from flask import jsonify


# Pot do shranjenega modela
model_filename = r'C:\Users\benja\OneDrive\Desktop\Projektna naloga - strojno učenje\Koncni_model_scaler\lstmUcenNaVseh.h5'

# Uvoz modela
model = load_model(model_filename)

# Pot do shranjenega scalerja
scaler_filename = r'C:\Users\benja\OneDrive\Desktop\Projektna naloga - strojno učenje\Koncni_model_scaler\scalerKoncni.pkl'

# Uvoz scalerja
scaler = joblib.load(scaler_filename)




app = Flask(__name__)




def pripravi_podatke_za_ucenje(vrednosti, okno_velikost):
    X = []
    for i in range(len(vrednosti) - okno_velikost + 1):
        X.append(vrednosti[i:i+okno_velikost, :]) 
    
    return np.array(X)





@app.route('/predict/projektna', methods=['POST'])
def post_example():
    json_data = request.get_json()
    
    df = pd.DataFrame(json_data)
    stevilo_vrstic = df.shape[0]
    print(stevilo_vrstic)
    
    print(df.isnull().sum())

    
    # Odstranjevanje sezonskosti in trenda s pomočjo metode razlika
    df['PM10_deseasonalized'] = df['PM10'] - df['PM10'].rolling(window=12, min_periods=1).mean()  # Uporaba premikajočega povprečja za sezonsko odstranitev

    # Odstranjevanje trenda s prvo razliko
    df['PM10_deseasonalized'] = df['PM10_deseasonalized'].diff()

    # Odstranitev prvih manjkajočih vrednosti, ki nastanejo zaradi diferenciacije
    df.dropna(subset=['PM10_deseasonalized'], inplace=True)
    
    print(df.isnull().sum())
    
    
    # Filtriranje značilnic
    najdoprinosne_znacilnice = ['NO2', 'PM2.5', 'Latitude']
    ciljna_znacilnica = 'PM10_deseasonalized'
    podatki = df[najdoprinosne_znacilnice + [ciljna_znacilnica]]
    
    
    ciljna_spremenljivka_standardized = scaler.fit_transform(podatki[[ciljna_znacilnica]])
    podatki_standardized = np.hstack((podatki[najdoprinosne_znacilnice], ciljna_spremenljivka_standardized))
    
    
    test_data = podatki_standardized
    stevilo = test_data.shape[0]
    print("Število vrstic: ",stevilo)
    
    
    okno_velikost = 60
    
    X_test = pripravi_podatke_za_ucenje(test_data, okno_velikost)
    stevilo_podatkov = X_test.shape[0]
    print(stevilo_podatkov)
    
    y_pred = model.predict(X_test)
    
    # Inverzna transformacija rezultatov
    rezultati_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    koncni_rezultat = rezultati_original[0]
    
    koncni_rezultat = float(koncni_rezultat)
    
    
    return jsonify({"prediction": koncni_rezultat})
    

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)











# In[ ]:




