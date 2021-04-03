# %%
"""
# **Estudio de la correlación entre Empresas**
"""

# %%
import bs4 as bs#esta libreria permite hacer scraping 
import datetime as dt
import os
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import pickle
import requests#esta libreria maneja todas las peticiones http cuando se accede a una pagina web 
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline  

# %%
!pip install bokeh.plotting

# %%
"""
## se van a analizar las 500 empresas norteamericanas mas importantes 
"""

# %%
from bokeh.io import show
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.sampledata.unemployment1948 import data

# %%
def sp500():
  resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")#accedemos a la pagina web de wikipedia donde tenemos las 500 empresas
  soup = bs.BeautifulSoup(resp.text,"lxml")#convertimos la informacion en lxml para que la data sea mas tratable 
  table = soup.find('table',{'class':'wikitable sortable'})#buscamos la tabla html que hay detras de la pagina web que almacena esa informacion 
  #print(table)
  empresas = []
  for row in table.findAll('tr')[1:]:#nos quedamos con la etiqueta tr, esta es muy comun en html ya que representa el comienzo de una tabla 
    empresa = row.findAll('td')[0].text#nos quedamos con la etiqueta td, esta es muy comun en html ya que representa el comienzo de una tupla
    empresas.append(empresa)#nos quedamos unicamente con el nombre
  with open("sp500.pickle","wb") as f:#guarde los archivos en modo escritura 
    pickle.dump(empresas,f)#el formato pickle es similar a csv pero la lectura se hace por columnas 
  return empresas
  #print(empresas)
    #imprimimos y devolvera un conjunto de etiquetas html con sus respectivos valores y nos quedamos con su abreviatura 

# %%
empresas = sp500()#la barra n se debe eliminar 
print(empresas)

# %%
"""
## Procesamiento de datos 
"""

# %%
def take_data():#esta funcion remueve el \n
  with open("sp500.pickle","rb") as f:#abrimos el archivo pickle en modo lectura 
    empresas = pickle.load(f)#guaramos la info en una variable normal 
    res = [] 
    for sub in empresas: 
      res.append(sub.replace("\n", ""))#remueve el caracter \n 
    return res
    #print(res) 
  

# %%
res = take_data()#podemos ver los datos limpios 
res

# %%
res = take_data()
print(res[400:])#usamos una funcion para enviar este string 
#print(type(res))

# %%
def collect_data(data):
  mydata = pd.DataFrame()
  for t in data:
    mydata[t] =wb.DataReader(t,data_source='yahoo',start='01-10-2020')['Adj Close']#le agregamos los datos de siempre 
  print(mydata)
  return mydata


# %%
my_data = collect_data(res[450:])#le pasamos las ultimas 50 empresas con las fechas del dataframe de mydata
#print(my_data)
#print(type(my_data))

# %%
print(type(my_data))#procesaremos la informacion en formato dataframe

# %%
def visualize_data(my_data):#recibe los datos anteriores para su posterior visualizacion 
  df_corr = my_data.corr()
  #print(df_corr.head())
  data = df_corr.values#obtenemos los valores de la correlacion 
  #print(data)
  
  fig = plt.figure()#dibujamos algunas figuras 
  ax = fig.add_subplot(1,1,1)
  heatmap = ax.pcolor(data,cmap=plt.cm.RdYlGn)#definimos los colores que tendra nuestro mapa de calor 
  fig.colorbar(heatmap)
  ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)#se define una escala adecuada de visualizacion
  ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
  ax.xaxis.tick_top()#
  column_labels = df_corr.column_labels
  row_labels = df_corr.index
  ax.set_xtickslabels(column_labels)
  ax.set_ytickslabels(row_labels)
  plt.xticks(rotation=90)
  heatmap.set_clim(-1,1)#definimos el escalado es decir la parte de la derecha del mapa de calor 
  
  fig = plt.figure(figsize=(20, 20))
  
  heat_map = sb.heatmap(data)
  plt.show()#visualizamos el mapa


# %%

'AMZN' in my_data.columns

# %%
"""
# MAPA DE CALOR 
"""

# %%
visualize_data(my_data)

# %%
"""
# NOTA: esta visualizacion es incompatible con jupyter al menos sin extenciones adicionales
"""

# %%
"""
# Se recomienda usar visual studio code para tener la posibilidad de visualizar mejor el mapa y tener acceso a las opciones de visualizacion como por ejemplo zoom
"""

# %%
"""
#### observe que los colores mas verdosos son 1 es decir que cuando una empresa sube, la otra tambien sube y el color rojo corresponde a una correlacion negativa es decir que cuando una empresa cae la otra tambien cae
"""

# %%
