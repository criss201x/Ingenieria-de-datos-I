# %%
import numpy as np
import pandas as pd
import pandas_datareader as wb
import matplotlib.pyplot as plt
from scipy.stats import norm
%matplotlib inline

# %%
#configuración de datos de activos de monte carlo, cuánto tiempo y cuántas previsiones
ticker = 'FB'#se estudiara facebook, #tablero de cotizaciones
t_intervals = 30 #pasos de tiempo previstos en el futuro
iterations = 25 # cantidad de simulaciones
#adquiriendo datos
data = pd.DataFrame()
data[ticker] = wb.DataReader(ticker, data_source='yahoo', start='2019-1-1', end='2019-8-10')['Adj Close']#leemos los valores de facebook y yahoo provee esos datos  en un intervalo de fechas
#preparación de registros de retorno de datos
log_returns = np.log(1 + data.pct_change())#calculamos la ganancia logaritmica 
#gráfico del precio de cierre histórico del activo
data.plot(figsize=(10,6));#se calcula la variancia como 1 + el valor recibido
#comportamiento historico de facebook en el 2019 

# %%
data.pct_change()#calcula el porcentaje de variacion de la empresa 

# %%
log_returns.plot(figsize =(10,6))#graficamos las ganancias de facebook 

# %%
u = log_returns.mean()#media
var = log_returns.var()#varianza
drift = u - (0.5 * var)#operamos un valor logaritmico
stdev = log_returns.std()#desviacion estandar 
daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(t_intervals, iterations)))#para calcular la simulacion usamos una funcion exponencial 
#la función de punto porcentual (ppf) si es la inversa de la función de distribución acumulativa.
#por esta razón, la función de punto porcentual también se conoce comúnmente como función de distribución inversa. es decir, para una función de distribución calculamos el
#Toma el último punto de datos como punto de partida para la simulación
S0 =data.iloc[-1]#extraemos el primer valor de la simulacion
price_list = np.zeros_like(daily_returns)
price_list[0] = S0
#Aplica simulación monte carlo en asset
for t in range(1, t_intervals):
    price_list[t] = price_list[t-1] * daily_returns[t]#calcula el precio actual en base al precio de ayer por las ganancias esperadas, estas se dan de manera aleatorea 
    #basadas en una distribucion acumulativa
#visualizamos las simulaciones 
plt.figure(figsize=(10,6))
plt.plot(price_list)

# %%
price_list.mean()

# %%
data.tail(n=20)#en los ultimos 20 dias podemos observar que la media es 196 es decir que vale la pena invertir es decir que tiene un potencial de subida grande 

# %%
price_list

# %%
