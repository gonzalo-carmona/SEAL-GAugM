# Predicción de Aristas para Data Augmentation en Problemas de Clasificación de Nodos con Graph Neural Networks
Repositorio de archivos correspondientes al TFM homónimo, con los cuales se han realizado los experimentos explicados en el mismo.

En este archivo damos una guía detallada de uso del repositorio y sus archivos, para reproducir dichos experimentos, así como el crédito a los repositorios originales
de cada una de las metodologías (SEAL para predicción de aristas, y GAug-M para el *data augmentation*).

En las carpetas **GAug_M_files** y **seal_files** se encuentran los archivos de tipo informativo (README, reglas de uso, licencia, etc.)
de los repositorios [GAUG](https://github.com/zhao-tong/GAug) y [SEAL_OGB](https://github.com/facebookresearch/SEAL_OGB), respectivamente, sobre los cuales se basa en su completitud este repositorio, que utiliza muchos de sus *scripts*, pero con ligeras
modificaciones para adaptar su uso.

## Entrenar un modelo predictor de aristas con SEAL
Lo primero que necesitaremos es entrenar un modelo de predicción de aristas mediante el *script* **seal_link_pred.py**, al que debemos proporcionar diversos argumentos. Explicamos a continuación los argumentos necesarios, ya que los valores por defecto del resto de argumentos del script son con los que se realizan los experimentos, por lo que los omitiremos:
- **--dataset**: El dataset (o grafo) sobre el que entrenar el modelo. Puede ser cualquiera de los que se utilizan en el repositorio [SEAL_OGB](https://github.com/facebookresearch/SEAL_OGB) (Cora, CiteSeer), pero además, se pueden utilizar los 4 restantes que se usan en [GAUG](https://github.com/zhao-tong/GAug) gracias al script *custom_datasets.py*; Flickr, BlogCatalog, PPI y AirUSA.
- **--num_hops**: Tamaño del subgrafo h-recubridor. Por defecto, su valor es h=1, que se puede utilizar si el *dataset* es demasiado denso (tiene demasiadas aristas), pero h=2 es adecuado cuando la capacidad de computación y memoria no supone un problema.
- **--use_feature**: Indica que se utilizarán las *features* explícitas como parte del *input* del modelo, por lo que se ha utilizado en todos los experimentos. Existe la opción de no utilizarlas ya que en algunos casos, mejora el rendimiento.
- **--dynamic_train, --dynamic_test** y **--dynamic_val**: Estos argumentos sirven para extraer los subgrafos recubridores "sobre la marcha", es decir, antes de realizar cada predicción, en lugar de extraerlos y guardarlos todos en memoria al comienzo de la ejecución. Esto es computacionalmente más lento, pero resuelve situaciones en las que no haya memoria suficiente para almacenar todos los subgrafos recubridores.
- **--train_node_embedding**: Análogo a --use_feature, pero relativo a los *node embeddings* (calculados por *node2vec*).
- **--save_appendix**: Los resultados se guardarán en la carpeta "*nombre del dataset**--data_appendix*". Los resultados de los experimentos realizados se han guardado en carpetas con apéndice *defaultparams*.

Por ejemplo, si quisiésemos entrenar el modelo para el dataset *Cora*, ejecutaríamos en un terminal, en la carpeta del repositorio, el siguiente comando:
```
python seal_link_pred.py --dataset Cora --num_hops 2 --use_feature --train_node_embedding --save_appendix defaultparams
```

Veríamos entonces el progreso del entrenamiento del modelo (que ocurre, por defecto, a lo largo de una *run* de 50 épocas). Además, la ejecución devolvería el valor k usado en la capa de *SortPooling*, y la época en la cual se obtuvo el mejor rendimiento. Es importante conocer estos valores para poder calcular la matriz de probabilidades.
```
Total number of parameters is 341730
SortPooling k is set to 48
```
```
Run 01:
Highest Valid: 90.62
Highest Eval Point: 4
   Final Test: 91.81
AUC
All runs:
Highest Valid: 90.62 ± nan
   Final Test: 91.81 ± nan
```

## Cálculo de la matriz de probabilidades

Una vez que tenemos un modelo entrenado, debemos utilizarlo para calcular la matriz de probabilidades del grafo (probabilidad de que exista cada arista entre dos nodos). Esto lo hacemos gracias al script **inference.py**. El procedimiento es más sencillo ahora; simplemente debemos proporcionar los siguientes argumentos:
- **--dataset**: *Dataset* sobre el que queremos realizar los cálculos.
- **--dynamic_train**: Si se utilizó --dynamic_train a la hora de entrenar el modelo, es necesario utilizarlo aquí también.
- **--num_hops**: Mismo valor que el que se utilizó en el entrenamiento (2 por defecto).
- **--best_run**: Época en la que se obtuvo el mejor rendimiento, denotada como **Highest Eval Point** en el *output* de *seal_link_pred.py*
- **--k**: Valor k de la capa *SortPooling*.

Siguiendo el ejemplo anterior, ejecutaríamos el siguiente comando para obtener la matriz de probabilidades de *Cora*:
```
python inference.py --dataset Cora --best_run 4 --k 48
```

La matriz se guarda en formato *pickle* en la carpeta *data/edge_probabilites*, lista para ser usada por GAug-M.

## Data Augmentation con GAug-M
El siguiente paso es entrenar y evaluar un modelo clasificador de nodos sobre el grafo modificado por GAug-M, y la matriz de probabilidades obtenida en el paso anterior. Para ello, ejecutamos el script **train_GAugM.py**, con los siguientes argumentos:
- **--dataset**: *Dataset* sobre el que queremos entrenar el modelo.
- **--gnn**: Arquitectura de GNN que se usará para clasificar los nodos; gcn, gat, gsage y jknet

Sólo con estos argumentos, el *script* utiliza la matriz calculada en el paso anterior, y tras los cálculos, nos proporciona la media y la desviación típica de la métrica micro-F1 a lo largo de 30 *runs* de 200 épocas cada una. Finalizando el ejemplo, si quisiéramos utilizar GSAGE, ejecutaríamos:
```
python train_GAugM.py --dataset Cora --gnn gsage
```

Gonzalo Carmona López, Universidad de Sevilla

goncarlop@alum.us.es

carmonalopezgonzalo@gmail.com
