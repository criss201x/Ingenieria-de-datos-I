# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# %%
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError

# %%
START_DATE = '2009-01-01'
#END_DATE = str(datetime.now.strftime('%Y-%m%d'))
END_DATE = '2020-12-31'
AMAZON = 'AMZN'

# %%
def get_data(ticker):
    try:
        stock_data = data.DataReader(ticker, 'yahoo', START_DATE, END_DATE)
    except RemoteDataError:
        print('No hay datos para {t}'.format(t=ticker))
    print('ORIGINAL')
    print(stock_data.head())
    return stock_data

# %%
def clean(stock):#esta funcion ayuda a la preparacion de datos removiendo aquellas anomalias en los datos como por ejemplo vacios 
    stock=get_data(AMAZON)
    dias=pd.date_range(start=START_DATE, end=END_DATE)
    stock = stock.reindex(dias)
    stock=stock.interpolate(method='time')
    stock.dropna(inplace=True)#quita las filas que tengan valores faltantes 
    print('CLEAN')
    print(stock.head())
    return stock.Close#devuelve un dataframe limpio mas en particular la columna de precio de este dataframe 


# %%
stock = get_data(AMAZON)
stock = clean(stock)#llama mi funcion clean y le envia datos por parametro 

# %%
stock.head(10)#dias y precio de cierre 

# %%
"""
## se crea un dataset de entrenamiento
"""

# %%
x=stock.values#definimos un array numpy
x.shape#devuelve las el numero de tuplas de nuestro dataset

# %%
"""
### convertimos la serie en un dataframe 
"""

# %%
x=x.reshape(x.shape[0],1)#array de dos dimenciones
x.shape#los datos ahora son una unica columna de muchas tuplas esto hace operable nuestro data set con numpy

# %%
def seriesTiempo_to_supervisado(X,timesteps,n_target):#elegimos el numero de dias que va tomar de imput nuestro dataframe, n_target el numero de dias a evaluar 
    
    x = np.zeros([len(X)-(timesteps+n_target), timesteps, X.shape[1]])#iniciamos dos arrays x y  donde x son las variables a entrenar
    y = np.zeros([len(X)-(timesteps+n_target), n_target])# y corresponde al array que contiene las variables de n_target 
    
    for t in range(timesteps):#iteramos la construccion del array de x y 
        x[:,t] = X[t:-(timesteps+n_target)+t,:]
    for i in range(n_target):
        y[:,i] = X[timesteps+i:-(n_target-i),0]
    return x,y

# %%
X,y=seriesTiempo_to_supervisado(x,30,5)
x.shape#x tiene menos casos que antes 

# %%
y.shape

# %%
"""
## Entrenamiento y test
"""

# %%
tt = int(0.9*len(x))#variables necesarias para evaluar el modelo
x_train,x_test = X[:tt], X[tt:]
y_train,y_test = y[:tt], y[tt:]

# %%
x_train.shape

# %%
x_test.shape

# %%
y_train.shape

# %%
y_test.shape

# %%
y_train[10,:]#tenemos un data set para crear un modelo de regresion para un modelo de clasificacion debemos clasificar los precios en funcion del precio de una variable input

# %%
x_train[10,:]

# %%
