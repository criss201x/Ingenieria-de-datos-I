# %%
!pip install pykalman

# %%
from pykalman import KalmanFilter#usamos el filtro de kalman para suprimir fluctuaciones de precios del dataset a trabajar 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import poly1d

# %%
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError

START_DATE = '2019-01-01'
#END_DATE = str(datetime.now().strftime('%Y-%m-%d'))
END_DATE = '2019-12-31'
AMAZON = 'AMZN'

# %%
def get_data(ticker):
    try:
        stock_data = data.DataReader(ticker, 'yahoo', START_DATE, END_DATE)
    except RemoteDataError:
        print('no hay datos para {t}'.format(t=ticker))
    return stock_data

# %%
x=get_data(AMAZON)#se trabajara con el precio de cierre
x=x.Close

# %%
#se define el filtro de kalman
kf = KalmanFilter(transition_matrices=[1],#este metodo sirve para calcular el valor siguiente una vez que sabemos el valor anterior 
                 observation_matrices=[1],
                 initial_state_mean=0,
                 initial_state_covariance=1,
                 observation_covariance=1,
                 transition_covariance=.05)#

#utilizar los valores observados del precio para obtener una media móvil
state_means, _= kf.filter(x.values)
state_means = pd.Series(state_means.flatten(), index=x.index)#se crea una serie temporal, el metodo flatten define una variable para la serie temporal

#calcular la media móvil con varias ventanas al pasado
mean10 = x.rolling(window=10).mean()
mean30 = x.rolling(window=30).mean()
mean60 = x.rolling(window=60).mean()
mean90 = x.rolling(window=90).mean()



# %%
#visualizamos los datos originales y la media estimada
plt.plot(state_means)
plt.plot(x)
plt.plot(mean30)
plt.plot(mean60)
plt.plot(mean90)
plt.title('Estimación de filtro de kalman del promedio')
plt.legend(['Estimado Kalman', 'X', 'Media móvil de 30 días', 'Media móvil de 60 días','Media móvil de 90 días'])
plt.xlabel('Dia')
plt.ylabel('Precio')
#en el grafico se puede observar que empieza con una muy mala estimacion y luego se ajusta 
#finalmente el resultado de kalman tiene mas precicion que la media en 90 dias

# %%
plt.plot(state_means[-200:])#medias moviles con el parametro  -200
plt.plot(x[-200:])#
plt.plot(mean30[-200:])#
plt.plot(mean60[-200:])#
plt.plot(mean90[-200:])#
plt.title('Estimación de filtro de kalman del promedio')#
plt.legend(['Estimado Kalman', 'X', 'Media móvil de 30 días', 'Media móvil de 60 días','Media móvil de 90 días'])#
plt.xlabel('Dia')#
plt.ylabel('Precio')#
#observece que el min period no viene dado desde 1 


# %%


# %%
