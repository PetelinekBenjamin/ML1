#!/usr/bin/env python
# coding: utf-8

# In[1]:


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




# Naložitev podatkov iz CSV datoteke
pot_do_datoteke = r'C:\Users\benja\OneDrive\Desktop\Projektna naloga - strojno učenje\RV2_UPP_IIR_SIPIA.csv'
df = pd.read_csv(pot_do_datoteke, parse_dates=['Date'], index_col='Date')

# Izpis števila manjkajočih vrednosti v vsakem stolpcu
print(df.isnull().sum())

# Sortiranje zapisov glede na čas
df.sort_index(inplace=True)

# Izris grafa vrednosti izposoje koles glede na čas
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['PM10'], label='PM10')
plt.xlabel('Datum')
plt.ylabel('PM10')
plt.legend()
plt.show()



# Predpostavka: df je vaš podatkovni okvir s podatki
# df_train so vrstice, kjer manjkajoče vrednosti niso prisotne
df_train = df.dropna()

# Ločitev atributov in ciljne spremenljivke
X_train = df_train.drop(['NO2', 'PM2.5', 'PM10'], axis=1)
y_train_NO2 = df_train['NO2']
y_train_PM25 = df_train['PM2.5']
y_train_PM10 = df_train['PM10']

# Ustvarjanje modelov Random Forest za vsak stolpec z manjkajočimi vrednostmi
rf_model_NO2 = RandomForestRegressor()
rf_model_PM25 = RandomForestRegressor()
rf_model_PM10 = RandomForestRegressor()

# Prileganje modelov
rf_model_NO2.fit(X_train, y_train_NO2)
rf_model_PM25.fit(X_train, y_train_PM25)
rf_model_PM10.fit(X_train, y_train_PM10)

# Izpolnitev manjkajočih vrednosti
X_missing = df[df.isnull().any(axis=1)].drop(['NO2', 'PM2.5', 'PM10'], axis=1)

# Napovedovanje manjkajočih vrednosti
predictions_NO2 = rf_model_NO2.predict(X_missing)
predictions_PM25 = rf_model_PM25.predict(X_missing)
predictions_PM10 = rf_model_PM10.predict(X_missing)

# Dodelitev napovedanih vrednosti nazaj v podatkovni okvir
df.loc[df['NO2'].isnull(), 'NO2'] = predictions_NO2[:len(df.loc[df['NO2'].isnull()])]
df.loc[df['PM2.5'].isnull(), 'PM2.5'] = predictions_PM25[:len(df.loc[df['PM2.5'].isnull()])]
df.loc[df['PM10'].isnull(), 'PM10'] = predictions_PM10[:len(df.loc[df['PM10'].isnull()])]

# Preverjanje, če so vse vrednosti zapolnjene
print(df.isnull().sum())

# Sestavljanje nove značilnosti NO2_ratio
df['NO2_ratio'] = df['NO2'] / df['PM2.5']

# Sestavljanje nove značilnosti O3_PM25_ratio
df['O3_PM25_ratio'] = df['O3'] / df['PM2.5']


# Pretvorba indeksa DataFrame-a df v datumske objekte
datum = pd.to_datetime(df.index, format='%d/%m/%Y')

# Dodajanje stolpcev day, month in year v DataFrame df na podlagi datumskega indeksa
df['day'] = datum.day
df['month'] = datum.month
df['year'] = datum.year


# Izpis zadnjih nekaj vrstic DataFrame-a, da preverimo nove značilnosti
print(df.tail())









# Izris grafa vrednosti izposoje koles glede na čas
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['PM10'], label='PM10')
plt.xlabel('Datum')
plt.ylabel('PM10')
plt.legend()
plt.show()




# Odstranjevanje sezonskosti in trenda s pomočjo metode razlika
df['PM10_deseasonalized'] = df['PM10'] - df['PM10'].rolling(window=12, min_periods=1).mean()  # Uporaba premikajočega povprečja za sezonsko odstranitev

# Odstranjevanje trenda s prvo razliko
df['PM10_deseasonalized'] = df['PM10_deseasonalized'].diff()

# Odstranitev prvih manjkajočih vrednosti, ki nastanejo zaradi diferenciacije
df.dropna(subset=['PM10_deseasonalized'], inplace=True)



# Izris prilagojene časovne vrste
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['PM10_deseasonalized'], label='PM10 (Deseasonalized)')
plt.xlabel('Datum')
plt.ylabel('PM10')
plt.legend()
plt.show()




print(df.columns)



print(df.isnull().sum())



# Izpis števila vrstic v DataFrameu
stevilo_vrstic = df.shape[0]
print("Število vrstic v DataFrameu:", stevilo_vrstic)



# Calculate feature importance using f_regression
doprinos, _ = f_regression(df.drop(['PM10_deseasonalized', 'PM10'], axis=1), df['PM10_deseasonalized'])


for zanc, dop in zip(df.columns,doprinos):
    print(f'{zanc} ima doprinos: {dop}')

    
    
    
# Filtriranje značilnic
najdoprinosne_znacilnice = ['NO2', 'PM2.5', 'Latitude']
ciljna_znacilnica = 'PM10_deseasonalized'
podatki = df[najdoprinosne_znacilnice + [ciljna_znacilnica]]

# Standardizacija samo ciljne spremenljivke
scaler = StandardScaler()
ciljna_spremenljivka_standardized = scaler.fit_transform(podatki[[ciljna_znacilnica]])
podatki_standardized = np.hstack((podatki[najdoprinosne_znacilnice], ciljna_spremenljivka_standardized))



#pot_do_scalerja = r'C:\Users\benja\OneDrive\Desktop\Projektna naloga - strojno učenje\Scaler\scalerKoncni1.pkl'
#joblib.dump(scaler, pot_do_scalerja)



# Ločitev na učno in testno množico
train_size = len(podatki) - 550
train_data, test_data = podatki_standardized[:train_size], podatki_standardized[train_size:]

# Preverjanje oblik podatkov
print("Oblika učnih podatkov:", train_data.shape)
print("Oblika testnih podatkov:", test_data.shape)
print()



def pripravi_podatke_za_ucenje(vrednosti, okno_velikost):
    X, y = [], []
    for i in range(len(vrednosti) - okno_velikost):
        X.append(vrednosti[i:i+okno_velikost, :])  # Vključi vse značilnice, vključno s ciljno značilnico
        y.append(vrednosti[i+okno_velikost, -1])  # Napovejte PM10_deseasonalized
    return np.array(X), np.array(y)




'''
# Seznam možnih velikosti okna
velikosti_okna = [10, 20, 30, 40, 50, 60, 70, 80,90,100,110]
    

# Inicializacija najboljše napake na visoko vrednost
najboljša_napaka = float('inf')
najboljše_okno = None

# Preizkus vsake velikosti okna
for okno_velikost in velikosti_okna:
    # Priprava podatkov z izbrano velikostjo okna
    X_train, y_train = pripravi_podatke_za_ucenje(train_data, okno_velikost)
    X_test, y_test = pripravi_podatke_za_ucenje(test_data, okno_velikost)
    
    # Izbris dimenzije 1 iz y_train in y_test
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # Ustvarjanje in učenje modela
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm1 = LSTM(128, return_sequences=True)(inputs)
    lstm2 = LSTM(128)(lstm1)
    dense1 = Dense(64, activation='relu')(lstm2)
    outputs = Dense(1)(dense1)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
    
    # Ocena uspešnosti modela na testnih podatkih
    napaka = model.evaluate(X_test, y_test)[0]
    
    # Posodobitev najboljšega okna, če je napaka boljša
    if napaka < najboljša_napaka:
        najboljša_napaka = napaka
        najboljše_okno = okno_velikost

print("Najboljše okno:", najboljše_okno)
'''


# Definirajte velikost okna
okno_velikost = 60

# Priprava učnih podatkov
X_train, y_train = pripravi_podatke_za_ucenje(train_data, okno_velikost)

# Priprava testnih podatkov
X_test, y_test = pripravi_podatke_za_ucenje(test_data, okno_velikost)



# Izbris dimenzije 1 iz y_train in y_test
y_train = y_train.flatten()
y_test = y_test.flatten()
        
    
'''
# Definicija modela z dodatnimi plasti DropOut
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm1 = LSTM(256, return_sequences=True)(inputs)
lstm2 = LSTM(256)(lstm1)
dense1 = Dense(64, activation='relu')(lstm2)
dropout = Dropout(0.2)(dense1)  # Dodamo plast Dropout
outputs = Dense(1)(dropout)

# Definicija modela
model_lstm = Model(inputs=inputs, outputs=outputs)

# Kompilacija modela
model_lstm.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Izvajanje učenja modela z dodanimi callbacki Dropout in Early Stopping
history = model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=0)


# Shranitev modela v mapo
model_lstm.save("C:\\Users\\benja\\OneDrive\\Desktop\\Projektna naloga - strojno učenje\\Modeli\\lstmUcenNaVseh1.h5")




# Preverjanje uspešnosti modela na testnih podatkih
y_pred = model_lstm.predict(X_test)



print("Oblika y_pred:", y_pred.shape)
print("Oblika y_test:", y_test.shape)


# Izračun metrike MSE na testnih podatkih brez uporabe inverse_transform
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Data:", mse)

# Izračun metrike MAE na testnih podatkih brez uporabe inverse_transform
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error on Test Data:", mae)

# Izračun R^2 na testnih podatkih brez uporabe inverse_transform
r2 = r2_score(y_test, y_pred)
print("R^2 on Test Data:", r2)

'''


'''
# Definicija modela
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
gru1 = GRU(128, return_sequences=True)(inputs)
gru2 = GRU(128)(gru1)
dense1 = Dense(64, activation='relu')(gru2)
dropout = Dropout(0.2)(dense1)
dense2 = Dense(32, activation='relu')(dropout)  # Dodatni gostoplastni sloj
outputs = Dense(1)(dense2)

# Definicija modela
model_gru = Model(inputs=inputs, outputs=outputs)

# Kompilacija modela
model_gru.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Izvajanje učenja modela z dodanimi callbacki Dropout in Early Stopping
history = model_gru.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=0)

# Shranitev modela v mapo
model_gru.save("C:\\Users\\benja\\OneDrive\\Desktop\\Projektna naloga - strojno učenje\\Modeli\\gru_model7.h5")

# Preverjanje uspešnosti modela na testnih podatkih
y_pred = model_gru.predict(X_test)

# Izračun metrike MSE na testnih podatkih brez uporabe inverse_transform
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Data:", mse)

# Izračun metrike MAE na testnih podatkih brez uporabe inverse_transform
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error on Test Data:", mae)

# Izračun R^2 na testnih podatkih brez uporabe inverse_transform
r2 = r2_score(y_test, y_pred)
print("R^2 on Test Data:", r2)

'''















print("MODELI IZ MAPE")
# Pot do mape, kjer so shranjeni modeli
pot_do_mape = "C:\\Users\\benja\\OneDrive\\Desktop\\Projektna naloga - strojno učenje\\Modeli"

# Naložitev vseh modelov v mapi
modeli = []
for datoteka in os.listdir(pot_do_mape):
    if datoteka.endswith(".h5"):
        model = load_model(os.path.join(pot_do_mape, datoteka))
        modeli.append(model)

# Za vsak model izračunaj napako na testnih podatkih
napake = []
for model in modeli:
    # Naredite napovedi za testne podatke
    y_pred = model.predict(X_test)
    
    # Izračunajte napako za model
    mse = mean_squared_error(y_test, y_pred)
    # Shrani napako
    napake.append(mse)

# Izpišite rezultate
for i, mse in enumerate(napake):
    print(f"Napaka modela {i+1}: {mse}")
    
    
    

indeksi_modelov = np.arange(len(napake)) + 1  # Dodamo 1, da se modeli štejejo od 1 naprej

# Narišemo graf
plt.figure(figsize=(10, 6))
plt.plot(indeksi_modelov, napake, marker='o', linestyle='-')
plt.xlabel('Modeli')
plt.ylabel('Napaka (MSE)')
plt.title('Napake modelov')
plt.grid(True)
plt.show()   
    
    
    
    
    
    
    
    
    
    
    
    

y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

#Izpis napovedanih vrednosti
print("Napovedane vrednosti (neskalirane):", y_pred_unscaled)


# In[ ]:




