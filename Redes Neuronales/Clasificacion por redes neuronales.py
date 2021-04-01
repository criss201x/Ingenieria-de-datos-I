# %%
"""
# problema de clasificacion resulto con redes neuronales
"""

# %%
#!pip install tensorflow # para instalar keras se requiere tensorflow
#!pip install keras #se requiere para nuestro modelo de red neuronal
#!pip install pykalman
#!pip install pandas
#!pip install numpy
#!pip install matplotlib
#!pip install datetime
#!pip install sklearn

from pykalman import KalmanFilter#usamos el filtro de kalman para suprimir fluctuaciones de precios del dataset a trabajar 
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Reshape, Flatten, Activation
from keras.layers import Dropout, BatchNormalization, GaussianNoise
from keras.layers import Dense, LSTM
from keras.constraints import maxnorm
from keras.optimizers import RMSprop, SGD, Adadelta, Adagrad, Adam

# %%
def tranform(stock):
    stock['diff_Open_Close']=stock.Open-stock.Close
    print('PRETRANSFORMADA')
    df=pd.DataFrame(stock['Adj Close'])
    df['diff Open Close']=stock.Open-stock.Close
    df['volume']=stock.Volume
    print('TRANSFORMADA')
    return df

# %%
#se define el filtro de kalman
def kalman_filter(df):
    x=df['Adj Close']
    kf = KalmanFilter(transition_matrices=[1],#este metodo sirve para calcular el valor siguiente una vez que sabemos el valor anterior 
                      observation_matrices=[1],
                      initial_state_mean=0,
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=.05)#
    state_means, _= kf.filter(x.values)#utilizar los valores observados del precio para obtener una media móvil
    state_means = pd.Series(state_means.flatten(), index=x.index)#se crea una serie temporal, el metodo flatten define una variable para la serie temporal
    df=df.drop(df.index.values[:6])
    return df

# %%
stock=pd.read_csv('datos_stock.csv')#

# %%
"""
### esta vez en lugar de importar los datos desde el api de yahoo vamos a usar el dataframe usado en el preprosesamiento de datos
"""

# %%
"""
es decir que tenia los datos normalizados como el porcentaje de subida y bajada con respecto al primer dia de cada año, este dataframe ya viene limpio y transformado
"""

# %%
stock=tranform(stock)
stock=kalman_filter(stock)

# %%
stock.head(10)#observece el procentaje de subida o bajada 

# %%
x=stock.values#se transforma en un array numpy

# %%
x.shape

# %%
"""
### las redes neuronales trabajan mejor con datos normalizados
es decir si tienen valores input presentes entre 0 y 1 
"""

# %%
scaler0=StandardScaler()#se normalizan los datos 
scaler0.fit(x[:,0].reshape(x[:,0].shape[0],1))#entrenamiento
x[:,0]=(scaler0.transform(x[:,0].reshape(x[:,0].shape[0],1))).flatten()#aplicar

# %%
scaler1=StandardScaler()#se hace para loas distintas columnas del dataset
scaler1.fit(x[:,1].reshape(x[:,1].shape[0],1))
x[:,0]=(scaler1.transform(x[:,1].reshape(x[:,1].shape[0],1))).flatten()

# %%
scaler2=StandardScaler()
scaler2.fit(x[:,2].reshape(x[:,2].shape[0],1))
x[:,0]=(scaler2.transform(x[:,2].reshape(x[:,2].shape[0],1))).flatten()

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
X,y=seriesTiempo_to_supervisado(x,30,2)#la ventanas de dias siempre es 30 
x.shape#x tiene menos casos que antes 

# %%
y.shape

# %%
X[-1,:,0]

# %%
y[-1,:]

# %%
"""
## variable a predecir
"""

# %%
precio_ultimo_dia=X[:,-1,0]
precio_dos_dias_despues=y[:,1]
target=precio_dos_dias_despues-precio_ultimo_dia
target[target<=0]=0
target[target>0]=1
target[:10]

# %%
"""
### variable input
"""

# %%
X.shape

# %%
from sklearn.utils import shuffle#el metodo shuffle hace una mezcla del data set y de la mezcla se recogen datos de training y los datos de set
X, target = shuffle(X, target)#de manera de que son distribuidos de manera mas uniforme

# %%
tt=int(0.9*len(x))

# %%
"""
## training y test
"""

# %%
x_train,x_test = X[:tt], X[tt:]
y_train,y_test = target[:tt], target[tt:]

# %%
x_train.shape

# %%
x_test.shape

# %%
y_train.shape

# %%
y_test.shape

# %%
y_test=y_test.reshape(y_test.shape[0],1)#convertirmos la dimencion de nuestro dataframe con el objetivo de que sea mas eficiente comparar los resultados obtenidos
y_train=y_train.reshape(y_train.shape[0],1)

# %%
"""
## Modelo
vamos a definir nuestro input y su respectiva forma, tambien se van a definir tambien capas ocultas de la red neuronal 
"""

# %%
"""
## se pueden hacer diferentes pruebas cambiando el ruido, variando el numero de neuronas por cada capa 
"""

# %%
_tensor_ = input_tensor = Input(shape=(x_train.shape[1],x_train.shape[2]), name='main_input')
_tensor_ = GaussianNoise(0.15)(_tensor_)
_tensor_ = LSTM(51, activation='relu', kernel_constraint=maxnorm(3.0), return_sequences=True)(_tensor_)#agrega ruido al dataset  para que la red neuronal no aprenda 
_tensor_ = Flatten()(_tensor_)#se usa el metodo flatten luego de la capa oculta ya que el input no es un vector si no una matriz despues se obtiene un vector de 90 componentes
_tensor_ = Dense(80, activation='relu', kernel_constraint=maxnorm(3.0))(_tensor_)#capas ocultas
_tensor_ = Dense(50, activation='relu', kernel_constraint=maxnorm(3.0))(_tensor_)#capas ocultas
_tensor_ = Dense(20, activation='relu', kernel_constraint=maxnorm(3.0))(_tensor_)#capas ocultas
ouput_tensor = Dense(1, activation='sigmoid', kernel_constraint=maxnorm(3.0))(_tensor_)#una sola neurona de salida con funcion de activacion sigmoidea
# esta salida de una sola neurona nos dara un valor cercano a 1 si la varialbe a target es 1 y cercano a 0 si clasifica este ejemplo como 0 

# %%
"""
## resumen de la red neuronal 
"""

# %%
model=Model(input_tensor,ouput_tensor)
model.compile(loss='mse', optimizer=Adagrad())
model.summary()

# %%
"""
## entrenamiento de la red neuronal
"""

# %%
model.fit(x_train, y_train, batch_size=300, epochs=150, shuffle=True, verbose=1)

# %%
"""
## variable de prediccion
es importante afirmar que la red no clasifica, se va a hacer la clasificacion segun una regla, es decir si la variable pred tiene un valor menos de 0.5 se clasifica como clase 0 en otros casos la clasificamos como clase 1
"""

# %%
y_pred = model.predict(x_test)

# %%
"""
### esta regla de clasificacion es totalmente arbitraria 
"""

# %%
y_pred_class=np.copy(y_pred)
y_pred_class[y_pred<0.5]=0#el valor optimo es 0.6
y_pred_class[y_pred>=0.5]=1

# %%
"""
## matriz de confucion 
se pasa la variable targe y la variable prediccion 
"""

# %%
"""
### en la parte izquiera superior tenemos un acierto verdadero de bajada y en la parte inferior derecha tenemos un acierto verdadero de subida 
### en las otras diagonales, en la derecha superior tenemos una falsa subida y en la izquierda inferior una falsa bajada
### se podrian cambiar las reglas para cambiar estos valores 
"""

# %%
confusion_matrix(y_test,y_pred_class)

# %%
(y_test==y_pred_class).sum()

# %%
y_test.shape

# %%
"""
no es optimo resolver el problema con metodos de aprendizaje no supervisado 
"""

# %%
import tensorflow as tf

# %%
tf.__version__

# %%
import keras

# %%
keras.__version__

# %%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# %%
