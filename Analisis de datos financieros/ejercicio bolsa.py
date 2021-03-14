# %%
"""
## Ejercicio Bolsa Solucionado

El objetivo es agrupar un conjunto de valores de bolsa de acuerdo a su comportamiento.
"""

# %%
"""
### Obtención de datos desde Google

Utilizamos Pandas y la API de Yahoo para obtener datos de bolsa
"""

# %%
!pip install yfinance --upgrade --no-cache-dir

# %%
import datetime
#agrupamiento basado en clustering es decir agrupacion de empresas en caracteristicas similares 
## from pandas.io.data import DataReader #OLD
##sudo pip install pandas-datareader
import pandas_datareader.data as web
import yfinance as yf#accedemos a la api de yahoo
lista_val=["MSFT","AAPL","T","GOOGL", "HPQ","VZ","CVX","ORAN","VOD","JPM","BBVA","RBS","BK"]#lista de empersas a analizar 


def lista_google (dstart,dend):#recibe los valores de la funcion de arriba
 lis=[]
 for i in range(0,len(lista_val)):
     f = yf.download(lista_val[i], dstart,dend)
     #f= web.DataReader(lista_val[i], 'stooq', dstart, dend)  
     print( lista_val[i] +" " + str(len(f['Close'])) +" " + "valores")
     lis.append(f['Close']) #creamos una lista donde guardamos los precios de cierre 
 return lis

lis=lista_google(datetime.datetime(2018, 1, 1),datetime.datetime(2019, 12, 31))	

# %%
print()

# %%
import datetime

## from pandas.io.data import DataReader #OLD
##sudo pip install pandas-datareader
import pandas_datareader.data as web
import fix_yahoo_finance as yf
lista_val=["MSFT","AAPL","T","GOOGL", "HPQ","VZ","CVX","ORAN","VOD","JPM","BBVA","BK"]
#lista_val=["BK"]

def lista_google (dstart,dend):
 lis=[]
 for i in range(0,len(lista_val)):
     f = yf.download(lista_val[i], dstart,dend)
     #f= web.DataReader(lista_val[i], 'stooq', dstart, dend)  
     print( lista_val[i] +" " + str(len(f['Close'])) +" " + "valores")
     lis.append(f['Close']) 
 return lis

lis=lista_google(datetime.datetime(2018, 1, 1),datetime.datetime(2019, 12, 31))	

# %%
print(type(datetime.datetime(2018, 1, 1)))

# %%


# %%


# %%
!pip install datetime

# %%
"""
El objetivo es utilizar el valor "close" de 2017 y 2018 de los valores de "lista_val" para construir 3 grupos de valores. Así como agrupamiento jerárquico.
Creamos Dataframe Pandas y volcamos info
"""

# %%

import pandas as pd
StockValues= pd.DataFrame(columns=lista_val)#creamos un dataframe donde vamos a volvar toda nuestra informacion extraida de la web 
for i in range (len(lista_val)):#iteramos para hacer el volcado 
    StockValues[lista_val[i]]=lis[i]
    


# %%
StockValues

# %%
"""
Almacenamos el fichero, como csv y como pickle
"""

# %%
export_csv = StockValues.to_csv ('dataframe_stock.csv', index = None, header=True) #exportamos toda nuestra info a un fichero csv

import pickle
pickle.dump( StockValues, open( "StockValues.p", "wb" ) )#lo subimos a pickle para manejarlo de una manera mas eficiente 

StockValues2 = pickle.load( open( "StockValues.p", "rb" ) )
stv=pd.read_csv('dataframe_stock.csv') 

# %%
print(StockValues2)

# %%


# %%
