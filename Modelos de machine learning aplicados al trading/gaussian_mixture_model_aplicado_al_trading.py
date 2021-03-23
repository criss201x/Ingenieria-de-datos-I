# %%
"""
# GMM
"""

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
stock=get_data(AMAZON)
stock['Adj Close']=stock['Adj Close'].apply(lambda x: ((x-stock['Adj Close'].iloc[0])/stock['Adj Close'].iloc[0]*100))
stock.head()#esta fluctuacion no puede ser mejor para predecir cada clase

# %%
stock=clean(stock)
stock=trasnformar(stock)

# %%
stock.head(10)

# %%
x=stock.values

# %%
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
for i in range(target.shape[0]):#esto en nuestro dataset son dos variables de la distribucion de los datos
    if target[i]==0:
        plt.scatter(X[i,0,0],X[i,1,0],color='r')#rojo la primera clase es decir cuando baja
    else:
        plt.scatter(X[i,0,0],X[i,1,0],color='b')#azul la segunda clase es decir cuando el precio sube
        #es complicado clasificar con kmeans 

# %%
X.shape

# %%
X_flatten=np.zeros((X.shape[0],X.shape[1]*X.shape[2]))#convertimos a un array de una dimencion
for i in range(X.shape[0]):
    X_flatten[i,:]=X[i,:].flatten()

# %%
X_flatten.shape#con esta transformacion tenemos esta dimencionalidad donde cada ejemplo ahora tiene 90

# %%
"""
## Training y test
"""

# %%
from sklearn.utils import shuffle#el metodo shuffle hace una mezcla del data set y de la mezcla se recogen datos de training y los datos de set
X_flatten, target = shuffle(X_flatten, target)#de manera de que son distribuidos de manera mas uniforme

# %%
tt=int(0.9*len(x))

# %%
x_train,x_test = X_flatten[:tt], X_flatten[tt:]
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
for i in range(y_train.shape[0]):#entrenamiento
    if y_train[i]==0:
        plt.scatter(x_train[i,range(0,90,3)[0]],x_train[i,range(0,90,3)[1]],color='r')#rojo la primera clase es decir cuando baja
    else:
        plt.scatter(x_train[i,range(0,90,3)[0]],x_train[i,range(0,90,3)[1]],color='b')#azul la segunda clase es decir cuando el precio sube
    plt.title('train')

# %%
for i in range(y_test.shape[0]):#test
    if y_test[i]==0:
        plt.scatter(x_test[i,range(0,90,3)[0]],x_test[i,range(0,90,3)[1]],color='r')#rojo la primera clase es decir cuando baja
    else:
        plt.scatter(x_test[i,range(0,90,3)[0]],x_test[i,range(0,90,3)[1]],color='b')#azul la segunda clase es decir cuando el precio sube
    plt.title('test')

# %%
from sklearn import mixture#algoritmo GMM para cada clase 

# %%
targets=np.unique(y_train)#saber cuantas clases distintas tenemos 
num_classes = len(targets)#numero de clases 
samples_pers_class = []

for c in range(num_classes):#se divide en trainig set entre las dos clases que tenemos 
    samples_pers_class.append(x_train[y_train==targets[c]])
    print(samples_pers_class[-1].shape)
    print(len(samples_pers_class[-1]))

# %%
#train 
num_subclases=4#indica el numero de gausionos que queremos para cada clase en este caso 8 
mixtures = []
for c in range(num_classes):
    mixtures.append(mixture.GaussianMixture(n_components=num_subclases,covariance_type='tied'))
    mixtures[c].fit(samples_pers_class[c])

# %%
#prediccion
densities = np.zeros([len(x_test),num_classes])#entrenamos la gausiana
prioris = np.zeros(num_classes)
for c in range(num_classes):
    #p(c)
    prioris[c]= np.log(len(samples_pers_class[c]))-np.log(len(x_test))
    #p(x|c)
    densities[:,c] = mixtures[c].score_samples(x_test)#el score devuelve la probabilidad de cada ejemplo perternece a un GMM o a otro 

# %%
y_pred = np.zeros(len(x_test),dtype=type(targets[0]))

for n in range(len(x_test)):
    k = 0
    for c in range(num_classes):
        if densities[n,c] + prioris[c] > densities[n,k] + prioris[k]: k = c
    y_pred[n] = targets[k]

# %%
y_pred

# %%
y_test

# %%
#modelo de agrupacion natural para varias empresas

# %%
def mean_absolute_perrcentage_error(y_true, y_pred, epsilon=0.1):
    diff = abs(y_true - y_pred)/numpy.maximum(abs(y_true), epsilon)
    return diff

# %%



# %%
