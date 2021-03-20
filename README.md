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

