# Trading inteligente con algoritmos de aprendizaje automático

### Abstract


Actualmente las tecnologías de la información mediante toda su ingeniería subyacente han ayudado a la mayoría de sectores y áreas del conocimiento a solucionar problemas que hace décadas ni se imaginaban, el objetivo del informe está enfocado exclusivamente en el sector financiero y para ser más específicos en el mercado, el trading y la inversión, se llevara a cabo una serie una serie de etapas las cuales van desde la extracción de los datos hasta la implementación de algunos modelos de aprendizaje automático con el fin de los resultados del un análisis sirvan como principal apoyo en la toma de decisiones en el contexto de la inversión.


**Keywords:** Aprendizaje automático, algoritmos, Trading, web Scraping, Python, Montecarlo, K-Means, Kalman, GMM.

### 1. Introducción

La previsión de precios de las acciones siempre ha sido de interés para inversores y analistas profesionales. El éxito en el comercio de acciones depende de elegir el mejor momento para comprar o vender acciones. Sin embargo, la previsión precisa del precio de las acciones es muy difícil debido al entorno intrínsecamente ruidoso y la alta volatilidad del movimiento de precios.

Con la actualización de la tecnología y la exploración de nuevos modelos de aprendizaje automático, el análisis de datos del mercado de valores ha ganado atención, ya que estos modelos proporcionan una plataforma para que los empresarios y los comerciantes elijan acciones más rentables. “Como estos datos están en grandes volúmenes y son muy complejos, siempre se considera la necesidad de un modelo de aprendizaje automático más eficiente para las predicciones en determinados intervalos de tiempo”[1].

El eje central de este informe es el sector financiero en este caso se habla del mercado como caso estudio porque este tiene una serie de reglas y patrones que pueden ser hasta cierto punto aprendidos por una maquina y esta puede recaudar información de muchas fuentes y posteriormente implementar un modelo que reconozca patrones use los datos para extraer información y bajo esta información tomar una decisión de compra o venta.

Este documento se encuentra organizado en 6 secciones, y finalmente unos resultados y conclusiones, las secciones están descritas de la siguiente manera, inicialmente en la
Sección 1, se habla del estado de arte y como diferentes autores han implementado algoritmos de inteligencia artificial en problemas de inversión, para la sección 2 se plantea un marco conceptual donde se menciona toda la fundamentación teórica de la lógica de negocio correspondiente y la manera de extraer información relacionada, para la sección 3 se formula una serie de técnicas y métodos para procesar la información obtenida de acuerdo a un modelo de flujo de datos establecido, en la sección 4 y 5 se plantean dos algoritmos los cuales son el método de Montecarlo y el k-means clustering, donde se define su descripción y aplicación para lograr una inversión inteligente, para la sección 6 se plantea el mismo proceso pero esta vez se va a implementar en una serie temporal, finalmente en los resultados se concluirá cuál de los algoritmos implementados es el más aproximado  a un valor óptimo.


### 1.1 Estado del arte

Con el rápido desarrollo de las técnicas de aprendizaje automático en los últimos años, se ha propuesto una gran cantidad de algoritmos para brindar apoyo a las decisiones de los inversores y el análisis profesional. Estas técnicas se utilizan a menudo en el comercio de acciones de dos formas: predicción del precio de las acciones y predicción del punto de negociación de las acciones.

En la predicción del precio de las acciones, Huang y Tsai [2] aplicaron regresión vectorial de soporte (SVR) y mapa de características autoorganizado (SOFM) para pronosticar el precio de las acciones. Hsieh y col. [3] wavelet de Haar integrado, algoritmo de colonia de abejas artificial, selección de correlación de regresión escalonada y redes neuronales recurrentes juntas para predecir el precio del índice bursátil. Svalina y col. [4] aplicó un sistema de inferencia difusa basado en red adaptable (ANFIS) para predecir el precio de las acciones. Laboissiere y col. [5] utilizó redes neuronales artificiales (ANN) para modelar la relación entre el precio de las acciones y las características de entrada. Chen y col. [6] aplicó series de tiempo difusas y grupos de variación difusa para predecir el índice bursátil ponderado por capitalización de la bolsa de valores de Taiwán (TAIEX) diario.

En la predicción de cotizaciones bursátiles, Bekiros [7] propuso un sistema híbrido neuro difuso para la toma de decisiones. Wen y col. [8] propuso un nuevo sistema de comercio inteligente basado en la predicción de la caja de oscilación mediante la combinación de la teoría de la caja de valores y el algoritmo de la máquina de vectores de soporte (SVM). Chang y col. [9] propuso un modelo que utiliza las representaciones lineales por partes y ANN para detectar señales de negociación de acciones. Como ANN carece de capacidad explicativa y tiene problemas de sobreajuste, Luo y Chen [10] integraron la representación lineal por partes (PLR) y la SVM ponderada (WSVM) para pronosticar las señales de negociación de acciones y compararon el resultado con el método basado en PLR y ANN. Ng y col. [11] presentó un método que se basa en el uso de un algoritmo genético (GA) para minimizar un error de generalización localizado ponderado para predecir puntos comerciales.

En este documento, se propone un novedoso sistema para analizar las acciones de una determinada empresa y tomar una decisión en cuanto a inversión se refiere, para lograr dicho objetivo es necesario aplicar algunas técnicas de Webscraping y algunos algoritmos de aprendizaje automático como podrían ser k-means, clasificaciones y regresiones, algoritmos de clustering,  Gaussian mixture model (GMM), y una red neuronal, se plantean una serie de etapas que van desde la extracción de los datos hasta la evaluación de que modelo de aprendizaje es más idónea si una con aprendizaje supervisado o una con aprendizaje NO supervisado  ya casi finalizando se evalúa que tan acertada fue la predicción programada y tener una respuesta de inversión inteligente.


### 2. Marco conceptual

### 2.1 Fundamentos básicos de inversión

El mercado de acciones es un lugar donde las personas pueden comprar acciones de una empresa y venderlas con ganancias, en algún momento en el futuro. Luego está el comercio intradía, donde la compra y la venta se realizan en un día. Si bien el mercado de acciones es ideal para las personas a las que no les importa invertir a largo plazo, los inversores a corto plazo suelen dominar el mercado comercial. Pero estos inversores necesitan formar diversas estrategias y realizar sus operaciones basándose en la información derivada de gráficos técnicos, patrones y tendencias. 

Uno de los primeros conceptos fundamentales en la inversión son los gráficos, estos gráficos normalmente están representados por lo que se conocen como velas, estas en trading hacen referencia a las velas japonesas, una herramienta que refleja en un gráfico información sobre la cotización de una acción en un determinado momento y que determinan el comportamiento de dicha acción en el mercado. En base al movimiento de las velas japonesas y de su lapso de tiempo, el trader puede tomar una decisión sobre qué es más conveniente, si comprar o vender. 

Las velas en trading indican la puja de los precios del mercado, su valor en la apertura y en el cierre de cada sesión bursátil. Cuando el precio de apertura es menor que el de cierre la vela se pinta de verde o blanco, lo que indica que la cotización subió (vela alcista), y si es al contrario, que el precio de apertura sea mayor que el de cierre, la vela se refleja en rojo o negro, lo que significa que la cotización bajó (vela bajista). Cuando ambos precios son iguales se le conoce como vela neutral.  

### 2.2	Extracción de datos 

El objetivo principal de Web Scraping es extraer información de uno o varios sitios web y procesarla en estructuras simples como hojas de cálculo, bases de datos o archivos CSV. Sin embargo, además de ser una tarea muy complicada, Web Scraping consume tiempo y recursos, principalmente cuando se realiza manualmente. Estudios anteriores han desarrollado varias soluciones automatizadas. El presente informe no se enfoca en hacer descripciones de técnicas avanzadas de web scraping, mas bien se le da un enfoque absolutamente practico en la implementación para el objetivo del análisis.

### 2.2	Extracción de datos financieros 

Para poder realizar operaciones de análisis de datos financieros es una buena idea partir de la extracción de datos de fuentes fiables, se plantean varias opciones para poder trabajar con datos totalmente relacionados al objetivo de estudio, a continuación, algunas de ellas:

- **Web Scraping (clásico):** por ejemplo, para buscar noticias relevantes con las empresas en las que se está invirtiendo
- **IEX Cloud:** Plataforma de pago que dispone de una versión gratuita donde se pueden hasta recolectar 500000 datos cada mes, es un api y dispone de un módulo en Python para desarrolladores
- **Finnhub.io** dispone de un plan gratuito con 60 llamadas al api por minuto, tiene una documentación amigable y maneja un api rest, link de la documentación https://finnhub.io/docs/api 
- **Alphaventage:** plataforma de pago, 500 peticiones al día con un coste de 30$ al mes. Nos ofrece datos en tiempo real
- **Tingo:** el plan gratuito ofrece hasta 500 llamadas al api al día y 500 símbolos al mes
- **API de Twitter:** permite evaluar de cómo se está hablando de las empresas en redes sociales, se puede hacer un análisis de sentimiento y tomar decisiones según redes sociales.
- **Api de Yahoo Finances:** es una opción conocida y muy popular en este nicho, a continuación, puede ver que su implementación en un lenguaje de programación es muy sencilla además que los resultados obtenidos son muy fáciles de entender.

![Figura 0](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_0.png)

### 3. Procesamiento y transformación de datos

Los datos que extraemos a menudo no están listos para ser analizados, pueden contener valores faltantes, no pueden faltar fechas como los fines de semana y feriados cuando el mercado está cerrado, si se desea obtener un modelo basado en series temporales no es posible dejar huecos de días en un Dataset, contienen muchas fluctuaciones que dificultan la convergencia del modelo predictivo. A continuación, se va a representar mediante un diagrama de flujo, que manifiesta el comportamiento de los datos desde su fase de extracción hasta obtener el valor de previsión.

![Figura 1](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_1.png)

**Figura 1.** Flujo de datos 

- **primera etapa:** Agregar los días que faltan a través de los comandos ofrecidos por la librería pandas
![Figura 2](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_2.png)

- **Segunda etapa: Rellenar valores faltantes:** Método fillna, que rellena los valores faltantes utilizando los valores del día anterior o siguiente, por otro lado el Método interpolate, que utiliza los valores anteriores y siguientes para definir una función que asocia los valores de precio con los días

![Figura 3](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_3.png)
- **Tercera etapa: reducir fluctuaciones:** Calcular la media móvil, simplemente es sustituir cada día con la media calculada con los x días antes

![Figura 4](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_4.png)

**Filtro de Kalman:** esta herramienta sirve para estimar el estado de un sistema dinámico lineal perturbado por el ruido, basado en mediciones (u observaciones) linealmente dependientes del estado y corrompidas por el ruido.

Es un filtro recursivo eficiente que evalua el estado de un sistema dinamico a partir de una serie de mediciones sujetas a ruido. se puede usar en este contexto si se toman valores reales y fluctuantes del precio como valor medido.

![Figura 5](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_5.png)

**Figura 2.** Filtro de Kalman 

Básicamente está basado en realizar una serie de simulaciones y probar diferentes casos para establecer de forma aleatoria condiciones del mercado es decir que no podemos predecir el futuro, pero podemos hacer simulaciones en base a datos pasados para intentar predecir cómo se va a comportar el mercado. [12]

En términos técnicos el método de Montecarlo es una técnica que utiliza números aleatorios y probabilidad para resolver problemas complejos. La simulación de Monte Carlo, o simulación de probabilidad, es una técnica utilizada para comprender el impacto del riesgo y la incertidumbre en los sectores financieros, la gestión de proyectos, los costos y otros modelos de pronóstico por aprendizaje automático.

![Figura 6](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_6.png)

Estos son diferentes comportamientos que podría tener el mercado en base a los parámetros que reciban el número de intervalos e iteraciones, sería interesante calcular la media de estas líneas para saber si va a subir o bajar (ejercicio completo disponible en el cuaderno jupyter).

![Figura 7](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_7.png)

**Figura 3.** Simulación de Montecarlo aplicada

### 5. K-Means Clustering

La agrupación en clústeres de K-means es una técnica no supervisada que se utiliza principalmente para la previsión financiera. [15] Esta técnica de agrupación basada en clústeres presenta la eliminación del peso mediante la normalización de atributos para obtener un resultado ideal.

La técnica de normalización se puede representar con la siguiente formula

X-Xmin/Xmax-Xmin

“La agrupación en clústeres de K-medias también implica el análisis de componentes principales que presenta la selección de solo aquellos componentes que tienen un efecto profundo en la clasificación. Los componentes restantes se eliminan.”[16]

Los pasos básicos de la agrupación en clústeres de K-medias son los siguientes:

1. Inicialice centroides que sean iguales a la cantidad de clases (objetivos) en los datos
2. Determine la distancia de cada objeto al centroide.
3. Seleccione la distancia mínima de las distancias obtenidas
4. Agrupar objetos sobre la base de distancias mínimas Continúe los pasos 3 y 4 hasta que se reasigne uno de los objetivos

### 5. Fases en un problema de aprendizaje automático aplicado al trading

