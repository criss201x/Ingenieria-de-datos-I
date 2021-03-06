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

### 6. Fases en un problema de aprendizaje automático aplicado al trading

![Figura 8](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_8.png)

**Figura 4.** Flujo de datos con series temporales

- **Definir una escala temporal:** se define cada cuando tiempo se va a tomar datos por ejemplo un mes
- **Seleccionar un conjunto de empresas:** no se podría hacer un modelo para todas las empresas, se podría hacer un modelo por empresas, por sectores, por empresas de determinado porcentaje de capitalización bursátil
- **Comprobar la existencia de valores nulos y faltantes:** se hace una limpieza de datos
- **Preparación de los datos para el modelo:** los datos sin transformar tienen un formato y el modelo a aplicar tiene otro, se requiere un proceso de preparación de datos
- **Entrenamiento del modelo:** se obtienen resultados

### 7. Pasos finales y resultados

- Evaluación del modelo
- Factores importantes: señal y predictibilidad, la señal pregunta si el precio va subir o va a bajar, la predictibilidad dice que tan fiable es el resultado de la predicción que ofrece el modelo
- ¿El precio sube o baja?
- ¿Cuánto de verdadero es ese resultado?
- El Precio de cierre es dividido por el precio de cierre predicho -1
- Interpretación de los resultados: de todas las empresas en el modelo de entrenamiento siempre hay empresas mejores que otras entonces un inversor trabaja sobre las mejores empresas

Un área clave que afecta las fluctuaciones del mercado es la psicología del mercado o del inversor. Esto se denomina "Finanzas conductuales". Aunque nos hemos vuelto cada vez más eficientes en la predicción de las tendencias del mercado utilizando métodos cuantitativos e Inteligencia Artificial, la psicología del mercado sigue confundiendo los modelos de predicción que hasta ahora nunca la han tenido en cuenta en el mismo nivel en el que afecta al mercado. [13]

Una solución adecuada para utilizar la psicología del mercado en nuestro beneficio es mediante el seguimiento de la interacción pública con y en las plataformas electrónicas de comercio financiero. Esto incluiría la minería de textos y el aprendizaje automático para establecer una escala de emociones de los inversores que podrían afectar directa o indirectamente al mercado.

**Resultados**

El valor optimo obtenido ya sea con un algoritmo u otro no es cuestión únicamente de aplicar el algoritmo al dataset en cuestión, en la fase de extracción y procesamiento de datos podemos observar que estos datos se han ido refinando de diversas maneras, la optimización que se hizo sobre el dataset previa al entrenamiento fue una reducción de fluctuaciones en los precios por medio del filtro de Kalman, que en secciones anteriores se explico, a continuación se muestran los resultados aplicados al dataset de yahoo.

![Figura 9](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_9.png)

**Figura 5.** Filtro de Kalman aplicado

Para el método de montecarlo podemos concluir aspectos respecto a tiempo y previsiones por medio de intervalos de tiempo y simulaciones para ello lo que se hizo fue estudiar datos financieros acerca de Facebook en un intervalo de fechas y se calcula su ganancia logarítmica, para su varianza se calcula el valor recibido más una unidad, en esencia lo que se calcula es un porcentaje de variación de la empresa y se grafican sus ganancias, todo esto para tener un análisis previo al entrenamiento deldataset.

El modelo propuesto también implementa un monitoreo constante de los comentarios e interacciones por parte de los usuarios y se incorporaría a una escala de calificación que representaría un factor en la toma de decisiones, esto se conoce como análisis de sentimiento, su aplicación en el contexto consiste en buscar un determinado tweet, después se hace lo que se conoce como minería de tweets, posteriormente se definen una serie de métodos de procesado de lenguaje natural y de acuerdo a unos puntajes obtenidos se definen unos umbrales y se grafican, para el caso de AMAZON los resultados serian los siguientes 10.

![Figura 10](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_10.png)

**Figura 7**. Análisis de opiniones obtenidas en los tweets de AMAZON

Por otro lado viendo los algoritmos de aprendizaje automático aplicados al presente contexto es importante aclarar que la eficiencia de estos va sujeto a una adecuada extracción y procesamiento de estos datos, por un lado tenemos el algoritmo GMM que Es una de las mejores técnicas de clustering de aprendizaje no supervisado para obtener una estimación de la densidad de un conjunto de muestras utilizadas para el aprendizaje del dataset.

Para la implementación de este algoritmo después de la fase de procesamiento de datos lo que se hace es extraer la fila donde está el precio del ultimo de cada ventana de 30 días, se evalúa su subida o su bajada de acuerdo a los valores en el vector, y para graficar se hace una iteración en el dataset y se visualizan las variables de distribución donde los puntos rojos traducen que el valor baja y azul cuando el valor sube.

![Figura 11](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_11.png)

**Figura 7.** Entrenamiento del dataset 

Teniendo el data set, este se divide en dos las dos clases que se tienen, se le agregan 8 variables gaussianas, estas se entrenan y su puntaje obtenido devuelve la probabilidad de que cada ejemplo pertenece a un GMM o a otro.

Por otro lado el algoritmo K-means que también es un algoritmo de clustering donde el objetivo es encontrar K vectores donde k es el numero de cluster y la media (u) tal que cada muestra será asignada al cluster correspondiente a la media mas cercana El modelo K-mean puede verse como un modelo GMM simplificado hasta el extremo, su implementación también es similar que el anterior.

![Figura 12](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_12.png)

**Figura 8.** Entrenamiento del dataset para K-means

Pero este dataset no serviría como modelo de entrenamiento porque no es posible hacer una buena predicción sin embargo GMM lo puede mejorar un poco aunque aún una solución por aprendizaje supervisado sigue quedándose corta.

Finalmente, para una solución por medio de clasificación por redes neuronales se traen los datos normalizados como el porcentaje de subida y bajada con respecto al primer día de cada año, este dataframe ya viene limpio y transformado, los datos normalizados para una red neuronal deben estar entre 0 y 1, también se requiere de una variable input y las capas de la red neuronal, la salida de una sola neurona dará como resultado un valor cercano a 1 si la variable a target es 1 y cercano a 0 si clasifica este ejemplo como 0, a continuación se puede observar un resumen de la red neuronal.

![Figura 12](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_13.png)

**Figura 9.** Resumen de la red neuronal

Ahora en el modelo se tiene que ponderarle de alguna forma errores específicos, un modelo financiero se puede equivocar en una compra por una mala estimación del riesgo, por ello es importante ponderar los diferentes tipos de errores que se pueden presentar. Por eso mismo se implementa una matriz de confusión donde esta recibe la variable target y la variable predicción.

en la parte izquierda superior tenemos un acierto verdadero de bajada y en la parte inferior derecha tenemos un acierto verdadero de subida en las otras diagonales, en la derecha superior tenemos una falsa subida y en la izquierda inferior una falsa bajada se podrían cambiar las reglas para cambiar estos valores.

![Figura 13](https://github.com/criss201x/Ingenieria-de-datos-I/blob/main/assets/Figura_14.png)

**Figura 10.** Matriz de confusión

### 8 Conclusiones

Con los datos presentados extraídos por medio de las diferentes técnicas de webscraping implementadas, y todas las observaciones e inferencias realizadas, es suficiente afirmar que la computación, el análisis y la predicción han recorrido un largo camino en la era moderna debido al uso del Machine Learning. Se tiene, para cualquier entusiasta de la inteligencia artificial, una gran cantidad de algoritmos que pueden proporcionarnos un rango variable de precisiones y eficiencias que se adaptan mejor a un determinado conjunto de datos.

Las técnicas antes mencionadas en el documento junto con los análisis de ondas de series de tiempo que fomentan las habilidades de estas técnicas “para generar conocimientos más precisos y avanzados sobre los mercados financieros de la actualidad".[14]

Los entrenamientos a los datasets anteriores y sus algoritmos de aprendizaje automatizo bien ya sea supervisado o no supervisado garantizan un 100% de éxito por ejemplo en una red neuronal es complejo hallar una combinación de capas, neuronas e iteraciones optima al modelo a predecir, finalmente hay que hacer un análisis técnico mas especializado o intentar con otros algoritmos como por ejemplo (SMV), sistemas de inferencia difusos, arboles de clasificación, etc…

### Referencias

1. S. Vazirani, A. Sharma and P. Sharma, &quot;Analysis of various machine learning algorithm and hybrid model for stock market prediction using python,&quot; 2020 International Conference on Smart Technologies in Computing, Electrical and Electronics (ICSTCEE),Bengaluru, India, 2020, pp. 203-207, doi: 10.1109/ICSTCEE49637.2020.9276859.
2. C.L. Huang and C.Y. Tsai, &quot;A hybrid SOFM-SVR with a filter-based feature selection for stock market forecasting&quot;, Expert Systems with Applications, vol. 36, pp. 1529-1539, Mar 2009.
3. T.J. Hsieh, H.F. Hsiao, and W.C. Yeh, &quot;Forecasting stock markets using wavelet transforms and recurrent neural networks: An integrated system based on artificial bee colony algorithm&quot;, Applied Soft Computing, vol. 11, pp. 2510-2525, Mar 2011.
4. I. Svalina, V. Galzina, R. Lujic, and G. Simunovic, "An adaptive network-based fuzzy inference system (ANFIS) for the forecasting: The case of close price indices", Expert Systems with Applications, vol. 40, pp. 6055-6063, Nov 1 2013.
5. L.A. Laboissiere, R.A.S. Fernandes, and G.G. Lage, "Maximum and minimum stock price forecasting of Brazilian power distribution companies based on artificial neural networks", Applied Soft Computing, vol. 35, pp. 66-74, Oct 2015.
6. S.M. Chen and C.D. Chen, "TAIEX Forecasting Based on Fuzzy Time Series and Fuzzy Variation Groups", IEEE Transactions on Fuzzy Systems, vol. 19, pp. 1-12, Feb 2011.
7. S.D. Bekiros, "Fuzzy adaptive decision-making for boundedly rational traders in speculative stock markets", European Journal of Operational Research, vol. 202, pp. 285-293, Apr 1 2010.
8. Q. Wen, Z. Yang, Y. Song, and P. Jia, "Automatic stock decision support system based on box theory and SVM algorithm", Expert Systems with Applications, vol. 37, pp. 1015-1022, Mar 2010.
9. P.C. Chang, T. W. Liao, J.J. Lin, and C.Y. Fan, "A dynamic threshold decision system for stock trading signal detection", Applied Soft Computing, vol. 11, pp. 3998-4010, Jul 2011.
10. L. Luo and X. Chen, "Integrating piecewise linear representation and weighted support vector machine for stock trading signal prediction", Applied Soft Computing, vol. 13, pp. 806-816, Feb 2013.
11. W.W.Y. Ng, X.L. Liang, J. Li, D.S. Yeung, and P.P.K. Chan, "LG-Trader: Stock trading decision support based on feature selection by weighted localized generalization error model", Neurocomputing, vol. 146, pp. 104-112, Dec 25 2014.
12. C. Lubritto et al., "Simulation analysis and test study of BTS power saving techniques," INTELEC 2009 - 31st International Telecommunications Energy Conference, Incheon, Korea (South), 2009, pp. 1-4, doi: 10.1109/INTLEC.2009.5351801.
13. Ye Shi-Qi and Peng Yong, “The Relation Between Risk And Return Of Portfolio Based On Standard Finance And Behavioral Finance,” IEEE International Conference On Control And Automation, June 2007.
14. P. Vats and K. Samdani, "Study on Machine Learning Techniques In Financial Markets," 2019 IEEE International Conference on System, Computation, Automation and Networking (ICSCAN), Pondicherry, India, 2019, pp. 1-5, doi: 10.1109/ICSCAN.2019.8878741.
15. Zhen Hu, Jie Zhu, and Ken Tse in “6th International Conference On Information Management, Innovation Management And Industrial Engineering,” November, 2013.
16. P. Vats and K. Samdani, "Study on Machine Learning Techniques In Financial Markets," 2019 IEEE International Conference on System, Computation, Automation and Networking (ICSCAN), Pondicherry, India, 2019, pp. 1-5, doi: 10.1109/ICSCAN.2019.8878741.

