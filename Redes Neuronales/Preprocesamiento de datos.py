# %%
"""
## Se extrae la informacion
"""

# %%
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from datetime import datetime 

# %%
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError

# %%
START_DATE = '2010-01-01'
#END_DATE = str(datetime.now().strftime('%Y-%m-%d'))
END_DATE = '2020-12-31'
AMAZON = 'AMZN'

# %%
def get_data(ticker):
    try:
        stock_data = data.DataReader(ticker, 'yahoo', START_DATE, END_DATE)
    except RemoteDataError:
        print('No hay datos para {t}'.format(t=ticker))
    print('originales')
    return stock_data

# %%
"""
## Se transforma y limpia la informacion
"""

# %%
stock=get_data(AMAZON)
print(stock.shape)

# %%
stock.describe()#informacion general de la empresa 

# %%
"""
## Se eliminan columnas que no se requieren para la construccion de la red neuronal
"""

# %%
stock.drop(columns=['High','Low','Open','Volume','Adj Close'], inplace=True)

# %%
stock.head()#nuestro dataframe tiene unicamente la fecha y su respectivo precio de cierre 

# %%
"""
## Elegimos un rango de fechas para nuestro caso de estudio
"""

# %%
dias = pd.date_range(start=START_DATE, end=END_DATE)
print(dias)
stock = stock.reindex(dias)

# %%
"""
## No se aceptan valores nulos!
en el dataset hay dias en los que quiza nunca hubo actividad estos quedan
como valores nulos en el dataset y para su adecuado entrenamiento estos valores se deben suprimir
"""

# %%
stock.head()

# %%
"""
### para solucionar este inconveniente se crea una columna llamada año
Esta columna utiliza estos indices para tener los precios de cierrte por año 
"""

# %%
stock['anio']=stock.index.year

# %%
stock.head()

# %%
"""
## Despues se hace una interpolacion
"""

# %%
stock=stock.interpolate(method='time')

# %%
stock.head(10)#los primeros valores de un dataset no se pueden interpolar

# %%
"""
## contamos y eliminamos los valores nulos 
"""

# %%
stock.isnull().sum()

# %%
stock=stock.fillna(method='bfill')#el metodo bill reemplaza el valor nulo por el valor anterior 

# %%
"""
### Agrupamos por año
"""

# %%
annio=stock.groupby('anio')
annio.head(10)

# %%
"""
## se crea un data set con todas las fechas del precio por año
"""

# %%
dataframe = pd.DataFrame(index=pd.date_range(start='2020-01-01', end='2020-12-31'))
for i,data_anio in annio:
    colonna=list(data_anio.Close.values)
    if len(colonna)!=366:
        colonna.append(np.nan)
    dataframe[i]=(colonna)
dataframe.tail(10)

# %%
"""
### contamos los valores nulos de este dataset 
"""

# %%
dataframe.isnull().sum()

# %%
"""
## NORMALIZACION DE DATOS 
"""

# %%
"""
lo que se desea es contabilizar cuanto se ha ganado en cada año y cuanto porcentaje
"""

# %%
rolling_data_porcentual=dataframe.apply (lambda x: ((x-dataframe.iloc[0])/dataframe.iloc[0])*100,  axis=1)

# %%
rolling_data_porcentual.isnull().sum()

# %%
"""
### Obtenemos un dataframe de la ganancia calculada por dia 
"""

# %%
rolling_data_porcentual.head(10)

# %%
"""
## vamos a unir los valores en una misma columna ya que para nuestra red neuronal se requiere un array unidimencional  
"""

# %%
df_anio=rolling_data_porcentual.iloc[:,0:11] 
df_anio_nuevo= df_anio.iloc[:,0].append(df_anio.iloc[:,1])

# %%
for i in range(0,9):#se rreccorre el array para su concatenacion
    df_anio_nuevo = df_anio_nuevo.append(df_anio.iloc[:,i+2])

# %%
"""
## Se comprueba la transfromacion de los datos 
"""

# %%
df_anio_nuevo.shape

# %%
df_anio_nuevo.isnull().sum()

# %%
"""
### Se eliminan valores nulos 
"""

# %%
df_anio_nuevo.dropna(inplace=True)
df_anio_nuevo.isnull().sum()

# %%
"""
## se agregan estos valores al dataframe original
"""

# %%
stock=get_data(AMAZON)
print(stock.shape)

# %%
stock.describe()

# %%
dias=pd.date_range(start=START_DATE, end=END_DATE)
stock = stock.reindex(dias)

# %%
stock=stock.interpolate(method='time')
stock=stock.fillna( method='bfill')

# %%
stock['Adj Close']=df_anio_nuevo.values

# %%
stock.head(370)

# %%
"""
## exportamos nuestros datos a csv
"""

# %%
stock.to_csv('datos_stock.csv')

# %%
