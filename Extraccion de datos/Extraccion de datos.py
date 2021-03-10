# %%
!pip install pandas_datareader

# %%
import pandas as pd
import numpy as np
from pandas_datareader import data as wb #datareader para extraer datos de yahoo

# %%
apple = wb.DataReader('AAPL', data_source='yahoo', start='2015-1-1', end='2017-1-1')#appl corresponde al stock al que se desea observar, intervalo de fechas
apple.head(10)
apple.tail()

# %%
type(apple)

# %%
apple['variacion'] = (apple.High - apple.Low)#variacion entre el maximo y el minimo 
apple['% variacion'] = (apple.High - apple.Low)/apple.Close#se podria hacer tambien con un precio de apertura 
apple.head(10)

# %%
apple.to_csv('aapl.csv')#exporta el dataframe a un archivo csv

# %%
apple.history(period='1y')#trae datos adicionales

# %%
"""
## Finnhub
"""

# %%
empresa = 'AAPL'
metrica = 'growth'
startdate = '2020-04-3'
endate = '2020-04-9'

# %%
import requests
r = requests.get('https://finnhub.io/api/v1/stock/metric?symbol={}&metric={}&token=bvtjj4n48v6pijne8ug0'.format(empresa,metrica))#variables por parametro en ()

# %%
print(r.json())
df = r.json()['metric']


# %%
d = requests.get('https://finnhub.io/api/v1/company-news?symbol={}&from={}&to={}&token=bvtjj4n48v6pijne8ug0'.format(empresa,startdate,endate))#variables por parametro en ()

# %%
d
print(d.json())

# %%
df = d.json()
df_new = json_normalize(df)
df_new.head(50)

# %%


# %%


# %%
