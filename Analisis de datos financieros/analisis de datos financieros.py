# %%
import pandas as pd
import numpy as np
from pandas_datareader import data as wb
import matplotlib.pyplot as plt

# %%
apple = wb.DataReader('AAPL',data_source='yahoo', start='2015-1-1')#vamos a llamar al stock de apple y el proveedor de estos datos sera yahoo
apple.head(10)#podriamos  colocar un parametro de fecha final 
apple.tail()#traemos los ultimos datos 

# %%
"""
# Cálculo de la tasa de Retorno
"""

# %%
"""
Se va a crear una nueva columna en el datframe y vamos a calcular la tasa de retorno, se calcula como el precio de cierre del dia siguiente con la funcion shift hacia adelante
y se le resta uno para tener lo valores bien normalizados 
"""

# %%
apple['Tasa de retorno'] =(apple['Adj Close']/apple['Adj Close'].shift(1)) -1
print(apple['Tasa de retorno'])#por ejemplo en enero del 2015 hubo un 15% de rentabilidad 
#la mejor manera de calcular rentabilidad por dia 

# %%
"""
# Visualizaciones
"""

# %%
"""

"""

# %%
apple ['Tasa de retorno'].plot(figsize=(8,5))

# %%
avg_returns_d = apple['Tasa de retorno'].mean()
avg_returns_d

# %%


# %%
avg_returns_a = avg_returns_d*250
print(str(round(avg_returns_a,4)*100) +'%')

# %%
"""
# **Tasa de Retorno Logarítmica**
"""

# %%
apple['Tasa de retorno Logaritmica'] =np.log(apple['Adj Close']/apple['Adj Close'].shift(1))
print(apple['Tasa de retorno Logaritmica'])

# %%
avg_returns_d_log = apple['Tasa de retorno Logaritmica'].mean()
avg_returns_d_log
avg_returns_a_log = avg_returns_d_log*250
print(str(round(avg_returns_a_log,4)*100) +'%')

# %%
"""
# Seleccion Cartera de Emperesa
"""

# %%
tickers = ['MSFT','AMZN','TSLA','V','FB']
mydata = pd.DataFrame()#construimos un dataframe a raiz de estas empresas 
for t in tickers:
  mydata[t] =wb.DataReader(t,data_source='yahoo',start='01-01-2019')['Adj Close']#traer fechas de entrada y precios de cierre

mydata.info()#

# %%
"""

"""

# %%
mydata.head()#tenemos nuestro dataframe de fechas vs empresas 

# %%
mydata.tail()#ultimos valores de nuestro dataframe 

# %%
mydata.iloc[1]

# %%
"""
# **Visualización y Normalizacion**
"""

# %%
"""
Visualizamos nuestra ganancia promedio
"""

# %%
(mydata / mydata.iloc[0]*100).plot(figsize = (15,6));#normalizamos los datos respecto al primer valor 

# %%
"""
# Calculo de la Tasa de rentabilidad
"""

# %%
returns=(mydata/mydata.shift(1)) -1#tenemos una serie de empresas a invertir 
returns.head()

# %%
weights = np.array([0.20,0.20,0.20,0.20,0.20])#vamos simular si se hubiera invetido que gananza se hubiera obtenido 
anual_returns = returns.mean()*250
np.dot(anual_returns,weights)#60% de rentabilidad 

# %%
pf1 = str(round(np.dot(anual_returns,weights),4)*100)+'%'#valor redondeado
print(pf1)

# %%
"""
# **Calculo del Valor de los Índices**
"""

# %%


# %%
indices = ['^GSPC','^IXIC','^GDAXI'] 
#^GSPC --> SYP500
#^IXIC --> NASDAQ
#^GDAXI --> DAX Alemnan
mydata = pd.DataFrame()
for t in indices:
  mydata[t] =wb.DataReader(t,data_source='yahoo',start='01-01-1995')['Adj Close']

mydata.info()

# %%
(mydata / mydata.iloc[0]*100).plot(figsize = (15,6));

# %%
ind_returns=(mydata/mydata.shift(1)) -1
ind_returns.head()

# %%
ind_anual_returns = ind_returns.mean()*250
print(ind_anual_returns)

# %%


# %%
"""
# Cálculo del Riesgo de una Inversión
"""

# %%
tickers = ['MSFT','TWLO']
mydata = pd.DataFrame()
for t in tickers:
  mydata[t] =wb.DataReader(t,data_source='yahoo',start='01-01-2019')['Adj Close']

mydata.info()

# %%
log_returns = np.log(mydata/mydata.shift(1))
log_returns

# %%
"""
# Microsoft
"""

# %%

log_returns['MSFT'].mean()
log_returns['MSFT'].mean()*250
log_returns['MSFT'].std()
log_returns['MSFT']*250**0.5

# %%
"""
## Twilo
"""

# %%
log_returns['TWLO'].mean()
log_returns['TWLO'].mean()*250
log_returns['TWLO'].std()
log_returns['TWLO']*250**0.5

# %%
log_returns[['MSFT','TWLO']].mean() * 250

# %%


# %%
log_returns[['MSFT','TWLO']].std()*250 **0.5

# %%
"""
# Covarianza y Correlacion
"""

# %%
"""
Hablamos de matriz de covarianza cuando lo que se quiere es tener varios valores es decir cuando queremos relacionar una empresa con varias
"""

# %%
"""
hablamos de correlacion para identificar si las empresas tienen algun tipo de correlacion
"""

# %%
msft_var = log_returns['MSFT'].var()#para poder trabajar la covarianza y correlacion debemos teber primero las variables con informacion 
msft_var

# %%
twilo_var = log_returns['TWLO'].var()#
twilo_var

# %%
msft_var = log_returns['MSFT'].var()*250#calculamos varianza anual 
msft_var

# %%
twilo_var = log_returns['TWLO'].var()*250
twilo_var

# %%
con_matrix_v = log_returns.cov()*250#obtenemos una matriz de covarianza
con_matrix_v# observece que tienen el mismo valor en diagonal microsoft vs twlo tienen esto quiere deicr que microfoft tiene menos varianza 

# %%
corr_matrix = log_returns.corr()#matriz de correlacion 
corr_matrix#es una correlacion media es decir cuando una tiende a subir la otra tambien 

# %%


# %%
tickers = ['V','MA']#ejemplo con empresas muy correlacionadas para ver que valores nos ofrece 
mydata = pd.DataFrame()#las empresas son visa y mastercad 
for t in tickers:
  mydata[t] =wb.DataReader(t,data_source='yahoo',start='01-01-2019')['Adj Close']

mydata.info()


# %%
log_returns = np.log(mydata/mydata.shift(1))
log_returns

# %%
con_matrix_v = log_returns.cov()*250
con_matrix_v


# %%
corr_matrix = log_returns.corr()#tiene un 92% de indice de correlacion es decir muy alto 
corr_matrix

# %%


# %%
"""
# Calculo del riesgo de un Porfolio
"""

# %%
weights = np.array([0.5,0.5])

# %%
"""
# **Varianza**
"""

# %%
porfolio_var = np.dot(weights.T, np.dot(log_returns.cov()*250,weights))
porfolio_var

# %%
"""
# Volatilidad
"""

# %%
porfolio_vol = (np.dot(weights.T, np.dot(log_returns.cov()*250,weights)))**0.5
porfolio_vol

# %%
"""
### **Creación de un Porfolio de Acciones**
"""

# %%


# %%
tickers = ['PG','TSLA']
pf_data = pd.DataFrame()
for t in tickers:
  pf_data[t] =wb.DataReader(t,data_source='yahoo',start='01-01-2019')['Adj Close']

pf_data.info()#obtenemos la informacion

# %%
(pf_data/pf_data.iloc[0]*100).plot(figsize=(10,5))

# %%
log_returns =np.log(pf_data/pf_data.shift(1))

# %%
log_returns.mean()*250

# %%


# %%
log_returns.cov()*250#obtenemos la varianza de estas dos empresas 

# %%
log_returns.corr()#evaluamos que tan correlacionadas estan ambas empresas 

# %%
num_empresas =len(tickers)#longitud de empresas evaluadas 
num_empresas

# %%
#portafolio diversificado
weights = np.random.random(num_empresas)#veremos diferentes relaciones de peso y de ganancia respecto a la diversificacion 
weights /= np.sum(weights) # weigths = weight / suma(weights)
weights#generamos una serie de pesos de manera aleatorea 

# %%
weights[0] +weights[1]

# %%
"""
Calculo de los beneficios de nuestro **Portfolio**
"""

# %%
np.sum(weights*log_returns.mean()) *250#calculamos el beneficio 
#ganancia logaritmica multiplicada por los pesos 

# %%
"""
**Calculo de la varianza del Porfolio**
"""

# %%
np.dot(weights.T,np.dot(log_returns.cov()*250,weights))#calculamos la variancia 

# %%
"""
**Calculo de la voltalidad del portfolio**
"""

# %%
np.sqrt(np.dot(weights.T,np.dot(log_returns.cov()*250,weights)))

# %%
"""
**Simulaciones con diferentes Pesos de nuestro Portfolio**
"""

# %%
porfolio_returns = []
porfolio_volatility = []
for x in range(1000):#esta iteracion calcula la ganancia y esos valores van a un vector 
  weights = np.random.random(num_empresas)
  weights /= np.sum(weights)
  porfolio_returns.append(np.sum(weights*log_returns.mean())*250)
  porfolio_volatility.append(np.sqrt(np.dot(weights.T,np.dot(log_returns.cov()*250,weights))))#hacemos lo mismo para la volatilidad
  porfolio_returns 

# %%
porfolio_returns = np.array(porfolio_returns)#se convierte en un array de numpy porque para procesarlo y visualizarlo es mejor 

# %%
 porfolio_volatility = np.array(porfolio_volatility)#se hace lo mismo parala volatilidad 

# %%
porfolio_returns


# %%
porfolio_volatility

# %%
porfolio = pd.DataFrame({'Ganancia':porfolio_returns, 'Volatilidad':porfolio_volatility})
porfolio.head(10)
porfolio.tail()

# %%
porfolio.plot(x='Volatilidad', y ='Ganancia', kind ='scatter', figsize=(10,10));#lo visualizamos por medio de una nube de puntos 
plt.xlabel('Volatilidad Esperada')
plt.ylabel('Gananacia Esperada')#observece que cuando aumenta la volatilidad tambien aumenta la ganacia 

# %%
