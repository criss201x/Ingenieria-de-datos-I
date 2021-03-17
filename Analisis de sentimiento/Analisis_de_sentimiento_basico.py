# %%
#!pip install tweepy
import sys,tweepy,csv,re#la libreria re funciona para el manejo de expresiones regulares 

# %%
#"!pip install textblob
from textblob import TextBlob#la libreria de textblob solo funciona en ingles :(
import matplotlib.pyplot as plt
import csv
import pandas as pd#la mejor opcion para almacenar los tweets

# %%
consumerKey = '9Gkj4ePNTtx7vqu0Wn9gQCN2D' # Cambiar las credenciales por las tuyas. ;) 
consumerSecret = 'A61Ms3UMZxendJISWdCrTFsfbRJ6gQ5NhsPALEJJhvnwZi7kOu'
accessToken = '1114714793372524546-R98RN9FTEsz0pdPLgi86pMwc0VgnOm'
accessTokenSecret = '5O9pQsA5i4X5CdIZ3dKdbFkMSRDZfXDdiY2i8zqMZutBs'

# %%
"""
# Extracción de Información
"""

# %%
"""
se creara una funcion donde se van a extraer los datos, esta funcion recibe dos parametros de entrada que es la palabra clave y un valor numerico 
"""

# %%
 def DownloadData(searchTerm,NoOfTerms):
        # autenticacion de credenciales twiter 
        auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
        auth.set_access_token(accessToken, accessTokenSecret)
        api = tweepy.API(auth)#si el objeto api se crea correctamente quiere decir que la autenticacion con twitter fue exitosa

        # buscando tweets
       #tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "es",).items(NoOfTerms)#esta busqueda se hace con los parametros que recibe y con el parametro de lenguaje 
        tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en",).items(NoOfTerms)#esta busqueda se hace con los parametros que recibe y con el parametro de lenguaje 
        return tweets

# %%
"""
se piden los dos parametros que recibe la funcion DownloadData por pantalla, posteriormente se guardan los tweets en una variable 
"""

# %%
"""
# Recolección de datos
"""

# %%
searchTerm = input("Ingrese palabra clave/hashtag a buscar: ")
NoOfTerms = int(input("Ingrese cuantos tweets va a buscar: "))
tweets = DownloadData(searchTerm,NoOfTerms)

# %%
 #Ejecutar solo para ver que todo funciona,luego no ejecutar esta linea()
for tweet in tweets:#no ejecutar !!!!!!
    print(tweet)  

# %%
"""
# Limpieza de Datos
"""

# %%
def cleanTweet(tweet):#recibe un tweet y lo limpia
        # Eliminar enlaces, caracteres especiales, etc.del tweet
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split(' '))#se da uso de la libreria re para el uso de expresiones regulares 
        #esta expresion regular traduce que se va a remover todo aquello que tenga caracteres distintos a A-Z y 0-9 ademas tambien remueve tabulaciones 
        # la funcion sub hace un sub string 
final = []        
for tweet in tweets:#no ejecutar !!!!!!        
    out = cleanTweet(tweet.text)
    final.append(out)

# %%
print(final)

# %%
"""
# Información Relevante en los Tweets
"""

# %%
#muestra el ultimo tweet
print("Fecha Publicación Tweet",tweet.created_at)#
print("Ubicación",tweet.coordinates)
print("Metadatos",tweet.metadata)#incluyen informacion del lenguaje entre otros...
print("Número de Reteweets", tweet.retweet_count)
print("URL en el Tweet", tweet.source_url)
print("Numeró de favoritos",tweet.favorite_count)#numero de likes

# %%
"""
#### devuele los datos de los ultimos n tweets mas recientes 
"""

# %%
"""
# Importación de los datos a formato CSV
"""

# %%
df = pd.DataFrame(final)#transformamos nuestro array de tweets a una lista de pandas
df.columns=["Twitter Text"]#agregamos una nueva columna 
#df.to_csv('tweets_es.csv',index = False)#pasamos los datos alli
df.to_csv('tweets_en.csv',index = False)#pasamos los datos alli

# %%
"""
### para estudiar el sentimiento se creaara una funcion que recibe el tweet limpio y lo que se va a hacer es calcular el sentmiento acumulativo de los tweets, se llama la libreria textblo y este algoritmo devolvera lo positivo del tweet es decir cuanto mayor sea el valor, mayor es lo positivo del tweet y luego se hace una subjetividad, esto nos da un poco de relatividad respecto al tweet, se acumula y finalmente se promedia el sentimiento total 
"""

# %%
"""
### se hace una clasificacion de los tweets en 3 niveles de positivo y en 3 niveles de negativo, basicamente lo que se va a hacer es en base al valor que obteniamos con la funcion textblob, se hacen una serie de comparaciones 
"""

# %%
def sentiment_analysis(tweet):
  #indices de polaridad 
  polarity = 0
  positive = 0
  wpositive = 0
  spositive = 0
  negative = 0
  wnegative = 0
  snegative = 0
  neutral = 0
  for tweet in final:
    analysis = TextBlob(tweet)
    print(analysis.sentiment.polarity)
    polarity += analysis.sentiment.polarity#se establecen umbrales 
    if (analysis.sentiment.polarity == 0):  # agregando reacción de cómo reacciona la gente para encontrar el promedio más tarde
        neutral += 1#cuantas veces aparece el sentimiento 0 
    elif (analysis.sentiment.polarity > 0 and analysis.sentiment.polarity <= 0.3):#lo mismo del anterior para los siguientes 
        wpositive += 1#se hacen comparaciones entre diferentes rangos 
    elif (analysis.sentiment.polarity > 0.3 and analysis.sentiment.polarity <= 0.6):
        positive += 1
    elif (analysis.sentiment.polarity > 0.6 and analysis.sentiment.polarity <= 1):
        spositive += 1
    elif (analysis.sentiment.polarity > -0.3 and analysis.sentiment.polarity <= 0):
        wnegative += 1
    elif (analysis.sentiment.polarity > -0.6 and analysis.sentiment.polarity <= -0.3):
        negative += 1
    elif (analysis.sentiment.polarity > -1 and analysis.sentiment.polarity <= -0.6):
        snegative += 1



  return neutral,wpositive,positive,spositive,negative,wnegative,snegative
lst_sentimientos = []
lst_sentimientos = sentiment_analysis(final)
print(lst_sentimientos)#obtenemos una lista de sentimientos ordenada 

# %%
"""
se obervan puntajes de polaridad por ejemplo el de 0.6 tiene la palara 'good' esto le da un impacto de positividad al tweet, tambien se podria calcular la polaridad media sumando y dividiendo...
"""

# %%
def percentage(part, whole):#se calcula el porcentaje que recibe es el valor y el numero de elementos en nuestro conjunto de datos es decir el total de tweets
  temp = 100 * float(part) / float(whole)

  return format(temp, '.2f')#retorna un valor de porcentaje 

# %%
num_elements = len(final)#calculamos la longitud de los valores almacenados y su polaridad 
neutral =percentage(lst_sentimientos[0],num_elements)
wpositivo = percentage(lst_sentimientos[1],num_elements)
positivo = percentage(lst_sentimientos[2],num_elements)
spositivo = percentage(lst_sentimientos[3],num_elements)
negativo = percentage(lst_sentimientos[4],num_elements)
wnegativo = percentage(lst_sentimientos[5],num_elements)
snegativo = percentage(lst_sentimientos[6],num_elements)

# %%
"""
# Visualización de los Tweets
"""

# %%
"""
### una vez categorizados los sentimientos, se crea una funcion que recibe estos valores calculados anteriormente y el numero de busquedas, posteriormente se definen una serie de banderitas y etiquetas del grafico correspondiente, se se definen en un rango de positivos,negativos y neutral, despues se crea una lista de los tamaños que contiene los valores de cada categoria, posteriormente los colores y despues el grafico que recibe colores, valores y dimenciones
"""

# %%
def visualizacion(neutral,positivo,wpositvo,spositvo,negativo,wnegativo,snegativo,searchTerm,NoOfTerms):
  labels = ['Positivo [' + str(positivo) + '%]', 'Debilmente positivo [' + str(wpositivo) + '%]','Fuertemente positivo [' + str(spositivo) + '%]', 'Neutral [' + str(neutral) + '%]',
  'Negativo [' + str(negativo) + '%]', 'Debilmente negativo [' + str(wnegativo) + '%]', 'Fuertemente negativo [' + str(snegativo) + '%]']
  sizes = [positivo, wpositivo, spositivo, neutral, negativo, wnegativo, snegativo]
  colors = ['yellowgreen','lightgreen','darkgreen', 'gold', 'red','lightsalmon','darkred']
  patches, texts = plt.pie(sizes, colors=colors, startangle=90)
  plt.legend(patches, labels, loc="best")
  plt.title('How people are reacting on ' + searchTerm + ' by analyzing ' + str(NoOfTerms) + ' Tweets.')
  plt.axis('equal')
  plt.tight_layout()
  plt.show()

visualizacion(neutral,positivo,wpositivo,spositivo,negativo,wnegativo,snegativo,searchTerm,NoOfTerms)

# %%
