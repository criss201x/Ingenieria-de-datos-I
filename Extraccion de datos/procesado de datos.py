# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# %%
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError 

# %%
#zona temporal para extraer datos 
START_DATE = '2019-01-01'
#END_DATE = str(datetime.now().strftime('%y-%m-%d'))
END_DATE = '2019-12-31'
AMAZON = 'AMZN'#nombre de la empresa la cual queremos saber los datos 

# %%
def get_data(ticker):#elegir desde cual api deseamos sacar los datos 
    try:
        stock_data = data.DataReader(ticker,'yahoo',START_DATE ,END_DATE)
    except RemoteDataError:
        print('no hay datos para {t}'.format(t=ticker))
    return stock_data

# %%
stock=get_data(AMAZON)#dataframe con los datos de amazon del 2019
print(stock.shape)
stock.describe()#estadisticas basicas del dataframe de amazon

# %%
stock.head()#5 primeras tuplas

# %%
dias = pd.date_range(start=START_DATE, end=END_DATE)
stock = stock.reindex(dias)
stock.head(10)

# %%
stock=stock.interpolate(method='time')#rellena los nulos con lo calculado en la funcion interpolate
stock.head(10)#escogimos la funcion time ya que trabajamos con datos de tipo fecha 
#NOTA: la funcion interpolate trabaja con los valores anteriores y posteriores por eso no puede interpolar la primera tupla 

# %%
stock.isnull().sum()

# %%
stock=stock.fillna(method='bfill')#rellena hacia atras se moveran hacia arriba

# %%
"""
## calculando la media movil
"""

# %%
rolling_data_20=stock.Close.rolling(window=20,min_periods=1).mean()#calculamos la media movil de este dataset
rolling_data_10=stock.Close.rolling(window=10,min_periods=1).mean()#inicialmente para el precio de cierre, el comando rolling ejecuta una operacion en este caso la media
plt.plot(rolling_data_20)
plt.plot(rolling_data_10)
plt.plot(stock.Close)
plt.legend(['media de 20 dias','media de 10 dias','price'])#etiquetas del panel superior derecho 
plt.show()

# %%
