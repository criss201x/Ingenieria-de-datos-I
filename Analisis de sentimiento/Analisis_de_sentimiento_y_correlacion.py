# %%
"""
# NO COMPILAR, SE REMOVIO LA LIBRERIA TWINT PORQUE NO ESTA FUNCIONANDO 
"""

# %%
 import sys,tweepy,csv,re

# %%
#ímport twint 
#ímport json
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data as wb
import numpy as np

# %%
"""
# Carga de datos
"""

# %%
data = pd.read_csv("/content/tweets.csv")#estos datos vienen de la librearia twint pero todo su codigo se removio 

# %%
data.head(10)
data_train = data[['date','tweet']]#nos quedamos con la fecha y el tweet
data_train.head(10)

# %%
tweets = data_train['tweet'].tolist()#tranforma  la columna tweet en una lista 

# %%
"""
# Limpieza de Datos
"""

# %%
def cleanTweet(tweet):#se hace limpieza del twwet

        # Eliminar enlaces, caracteres especiales, etc.del tweet

        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split(' '))

# %%
final = []
for tweet in tweets:#se itera la limpieza de los tweets
  final.append(cleanTweet(tweet))
print(final)

# %%
data_train['tweet'] = final#imprime un dataframe limpio con su respctiva fecha y tweet
data_train.head(10)

# %%
data_train.tail(20)#muestra 20 ultimos 

# %%
data_train['Sentimiento'] = 0.0#creamos una nueva columna que va a servir para guardar los valores del analisis de sentimiento 
data_train["Sentimiento"] = data_train["Sentimiento"].astype(float)#importante tener los valores en tipo float 

# %%
data_train.head(10)#se define y se observa la columna sentimiento 

# %%
for i in range (len(data_train)):
    data_train['Sentimiento'][i] = TextBlob(data_train['tweet'][i]).sentiment.polarity#con la libreria textblob calculamos el sentimiento, se recorre el dataframe y a cada fila se le agrega su respectivo score de sentimiento 


    

# %%
data_train.tail(10)#tenemos un dataframe con los tweets y su semtimiento 

# %%
df_new = data_train.drop(['tweet'], axis=1)#dejamos nuestro datrame unicamente con la fecha y el sentimiento 

# %%
df_new.head(10)

# %%
df_new_2 = df_new.groupby('date')#ahora lo que se busca es hacer un nuevo dataframe donde tengamos agrupados los tweets por fecha promediados 

# %%
dict1 = {}
for nombre,data in df_new_2:
  dict1[nombre]=data['Sentimiento'].mean()#creamos un diccionario de datos donde guardamos en cada posicion el sentimiento medio 



# %%
print(dict1)
df = pd.DataFrame(dict1.values(),index=dict1.keys())
df.head()
#convertimos nuestro diccionario de datos otra vez a dataframe 

# %%
df.head(10)

# %%
df.tail(10)

# %%
"""
# Estudio de la correlación con Análisis de sentimiento
"""

# %%
df.head(10)

# %%
stock = wb.DataReader('AMZN',data_source='yahoo', start='2020-05-06',end ='2020-05-28')#obtenemos los datos de amazon entre estas fechas para su comparacion 
stock.head(10)

# %%
days=pd.date_range(start='2020-05-06', end='2020-05-28')#reindexamos para tener un conjunto de fechas 
stock = stock.reindex(days)
stock=stock.interpolate( method='time')#interpolamos para tapar los vacios 
stock=stock.fillna( method='bfill')#en caso de no encontrar un valor lo llena basado en el anterior o el siguiente 

# %%
stock.head(10)#recordemos que nuestro datframe tiene precios de cierre, de apertura, el mas alto etc 

# %%
 a = df.values.tolist()#obtenemos los valores de la parte derecha 
 print(a)

# %%
sentimiento_list = []#lo transformamos a una lista 
#flatten the lis
for x in a:
    for y in x:
        sentimiento_list.append(y)

# %%
print(sentimiento_list)

# %%
stock['Sentimiento Medio'] = sentimiento_list#creamos una nueva columna con el sentimiento nuevo y le agregamos y le agregamos la lista anterior 


# %%
stock.head(10)#te tiene el dataframe en perfectas condiciones para un analisis de correlacion 

# %%
stock ['Diferencia del dia '] = stock['High'] - stock['Low']#diferencia entre el precio mas alto y el precio mas bajo 

# %%
stock.corr()#obtenemos la correlacion entre las diferentes variables 

# %%
"""
## cuando hay un valor positivo el mercado reduce su volatilidad.       
"""

# %%
"""
## los algoritmos los hace las librerias, para un analisis mas detallado se recomienda implementar nuestros propios algoritmos 
"""

# %%
