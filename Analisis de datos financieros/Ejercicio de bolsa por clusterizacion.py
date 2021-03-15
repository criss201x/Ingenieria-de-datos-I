# %%
"""
## Ejercicio Bolsa Solucionado

El objetivo es agrupar un conjunto de valores de bolsa de acuerdo a su comportamiento.
"""

# %%
import pickle

StockValues = pickle.load( open( "StockValues.p", "rb" ) )


# %%
print(StockValues)
#lis=lista_ibex()


# %%
lista_val=list(StockValues.columns.values)#creamos un cluster y lo manipulamos con kmean de skilearn, transformamos los valores a una lista 
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1)#creamos 3 cluster y los recorremos por medio de un algoritmo iterado 100 veces
import numpy as np
X = np.array(StockValues.T) ### we need to transpose
#for i in range (len(X)): ### Normalizar, no es necesario
#    X[i]=X[i][0]#los valores se deben transponer 
km.fit(X)#utilizamos la funcion fit para entrenarlo
print(lista_val)#esta lista corresponde a la lista de empresas que tenemos 
print(km.labels_)
#microfost, apple, google tienen coportontamientos distintos

# %%
lista_val
X

# %%
%matplotlib inline 
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt#vamos a graficar un dendograma, este nos da informacion de cuan similar es una empresa 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
ytdist = euclidean_distances(X)
#ytdist = 1-cosine_similarity(X)


#el dendograma esta mal debido a las distancias euclidianas 
#estas distancias no tienen en cuenta el valor de inversion 
Z = hierarchy.linkage(ytdist, 'single')

plt.figure(figsize=(25, 15))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(  Z ,labels=lista_val )
plt.show()#se puede observar erroneamente que microsoft es similar a jp morgan

# %%
from sklearn.metrics.pairwise import cosine_similarity

ytdistcos = 1 - cosine_similarity(X)

Z = hierarchy.linkage(ytdistcos, 'single')#el problema del dendograma anterior lo arreglamos basando la distancia en cosenos es decir los angulos del vector 
#los valores vienen de matrices de similitud 

plt.figure(figsize=(25, 15))
plt.title('Hierarchical Clustering Dendrogram- Cosine')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z ,labels=lista_val )
plt.show()#el cluster nos esta agrupando comportamientos muy similares
#metricas basicas de inversion

# %%


# %%
