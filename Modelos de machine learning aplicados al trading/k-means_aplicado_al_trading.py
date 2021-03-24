# %%
"""
## se extrae la informacion
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
START_DATE = '2020-01-01'
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
def clean(stock):
    dias = pd.date_range(start=START_DATE, end=END_DATE)
    stock = stock.reindex(dias)
    stock = stock.interpolate(method='time')
    stock.dropna(inplace=True)
    print('LIMPIOS!')
    return stock

# %%
def trasnformar(stock):#esta funcion define el modelo al definir cuales variables tener en cuenta y cuales no 
    stock['diff_Open_Close']=stock.Open-stock.Close#precio de inicio menos precio de cierre 
    print('pre transformada')
    df=pd.DataFrame(stock['Adj Close'])
    df['diff open close']=stock.Open-stock.Close
    df['Volume']=stock.Volume#volumen diario
    print('transformada')
    return df

# %%
!pip install pykalman

# %%
from pykalman import KalmanFilter
def kalman_filter(df):#removemos las fluctuaciones de los precios ajustados de cierre por medio del filtro de kalman
    x=df['Adj Close']
# construimos el filtro de kalman
    kf = KalmanFilter(transition_matrices=[1],
                     observation_matrices=[1],
                     initial_state_mean=0,
                     initial_state_covariance=1,
                     observation_covariance=1,
                     transition_covariance=.05)
    state_means, _ = kf.filter(x.values)#utilizar los valores observados del precio para obtener una media móvil
    df['Adj Close']=state_means.flatten()
    
    df=df.drop(df.index.values[:10])
    return df #devuelve el mismo dataframe pero en la columna de precio de cierre ajustado se ha usado el filtro de kalman 

# %%
stock=get_data(AMAZON)

# %%
stock=clean(stock)
stock=trasnformar(stock)
stock=kalman_filter(stock)

# %%
stock.head(10)

# %%
x=stock.values
x.shape

# %%
def timeseries_to_supervised(X,timesteps,n_target):
    x = np.zeros([len(X)-(timesteps+n_target), timesteps, X.shape[1]])
    y = np.zeros([len(X)-(timesteps+n_target), n_target])
    for t in range(timesteps):
        x[:,t]=X[t:-(timesteps+n_target)+t,:]
    for i in range(n_target):    
        y[:,i]=X[timesteps+i:-(n_target-i),0]
    return x,y

# %%
X,y=timeseries_to_supervised(x,30,2)#dataframe de 30 dias

# %%
X[-1,:,0]#es posible saber si en dos dias el precio sube o baja 

# %%
"""
## Variable a predecir
"""

# %%
precio_ultimo_dia = X[:,-1,0]#se extrae la fila donde esta el precio del ultimo de cada ventana de 30 dias 
precio_dos_dias_despues=y[:,1]#aqui se guarda el precio dos dias despues de este ultimo 
target=precio_dos_dias_despues-precio_ultimo_dia#agrego las diferencias 
target[target<=0]=0#se evalua si es mayor o menor 
target[target>0]=1
target[:10]#tenemos un vector de 0 y 1  0 cuando baja y 1 cuando sube

# %%
"""
## Visualizacion
"""

# %%
for i in range(target.shape[0]):
    if target[i]==0:
        plt.scatter(X[i,0,0],X[i,1,0],color='r')#rojo la primera clase es decir cuando baja
    else:
        plt.scatter(X[i,0,0],X[i,1,0],color='b')#azul la segunda clase es decir cuando el precio sube
        #es complicado clasificar con kmeans 

# %%
X.shape

# %%
"""
## variable input
"""

# %%
X_flatten=np.zeros((X.shape[0],X.shape[1]*X.shape[2]))#convertimos a un array de una dimencion
for i in range(X.shape[0]):
    X_flatten[i,:]=X[i,:].flatten()

# %%
X_flatten.shape#con esta transformacion tenemos esta dimencionalidad donde cada ejemplo ahora tiene 90

# %%
tt=int(0.9*len(x))

# %%
x_train,x_test = X_flatten[:tt], X_flatten[tt:]
y_train,y_test = target[:tt], target[tt:]

# %%
x_train.shape
x_test.shape
y_train.shape
y_test.shape

# %%
"""
## modelo
"""

# %%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(x_train)
y_pred = kmeans.predict(x_test)
y_pred

# %%
#con kmeans no se puede hacer una buena prediccion, entonces dembemos plantear un algoritmo de aprendizaje supervisado donde se pueden ajustar pesos 
#para las diferentes lineas de separacion 

# %%
