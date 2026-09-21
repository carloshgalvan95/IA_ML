# Actividad Semanas 6 y 7: Riesgo Crediticio - South German Credit

## Documentacion detallada de la solucion

**Curso:** Inteligencia Artificial y Aprendizaje Automatico  
**Programa:** Maestria en Inteligencia Artificial Aplicada - Tecnologico de Monterrey  
**Profesor:** Luis Eduardo Falcon Morales

---

## Contexto general del problema

El problema abordado consiste en predecir el **riesgo crediticio** de clientes bancarios utilizando el dataset *South German Credit* del repositorio UCI Machine Learning. Este dataset contiene 1000 registros de clientes con 20 variables de entrada y una variable de salida (`credit_risk`) que indica si el cliente cumplio o no con su credito.

El articulo de referencia IEEE (*"An Investigation of Credit Card Default Prediction in the Imbalanced Datasets"*, Alam et al., 2020) propone que las tecnicas de sobremuestreo funcionan mejor que las de submuestreo para este tipo de datos desbalanceados, y utiliza normalizacion Min-Max para escalar las variables numericas. Se tomo informacion de este articulo (particularmente la Tabla 3 y la seccion de normalizacion) como guia para las decisiones de preprocesamiento.

---

## Ejercicio 1: Carga de datos y renombramiento de columnas

### Que se hizo

Se realizaron tres acciones principales:

1. **Importacion de librerias** (Cell 4): Se cargaron todas las dependencias necesarias para la actividad completa, organizadas por categoria funcional.
2. **Carga de datos** (Cell 5): Se leyo el archivo `SouthGermanCredit.asc` usando `pd.read_csv` con separador de espacio.
3. **Renombramiento de columnas** (Cell 6): Se tradujeron los nombres de columnas del aleman al ingles.

### Toma de decisiones

**Librerias seleccionadas y por que:**

| Categoria | Librerias | Justificacion |
|-----------|-----------|---------------|
| Datos | `pandas`, `numpy` | Manipulacion de DataFrames y operaciones numericas |
| Visualizacion | `matplotlib`, `seaborn` | Graficos estadisticos (histogramas, boxplots, heatmaps) |
| Preprocesamiento | `MinMaxScaler`, `OneHotEncoder`, `OrdinalEncoder`, `ColumnTransformer` | Transformaciones diferenciadas por tipo de variable (requerido por instrucciones) |
| Modelos | `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `XGBClassifier`, `MLPClassifier`, `SVC` | Los 7 modelos especificados en las instrucciones |
| Validacion | `cross_validate`, `RepeatedStratifiedKFold` | Evaluacion robusta con validacion cruzada repetida |
| Desbalance | `ImbPipeline`, `SMOTE` | Pipeline compatible con imblearn y tecnica de sobremuestreo |
| Metricas | `classification_report`, `confusion_matrix`, `recall_score` | Evaluacion orientada a recall como metrica principal |

**Mapeo de columnas:** Se utilizo la documentacion oficial del dataset en UCI para obtener el mapeo correcto aleman-ingles. Por ejemplo:

```
laufkont  -> status               (estado de cuenta corriente)
laufzeit  -> duration             (duracion del credito en meses)
moral     -> credit_history       (historial de cumplimiento crediticio)
hoehe     -> amount               (monto del credito en marcos alemanes)
kredit    -> credit_risk          (variable de salida: riesgo crediticio)
```

El mapeo completo cubre las 21 columnas (20 features + 1 target). Se uso `df.rename(columns=dict, inplace=True)` para aplicar la transformacion en sitio.

### Por que se hizo asi

- Se importaron todas las librerias al inicio para tener visibilidad completa de las dependencias y evitar errores de importacion durante la ejecucion posterior.
- Se uso `warnings.filterwarnings('ignore')` para suprimir advertencias de convergencia de modelos que pueden ser extensas pero no representan errores criticos.
- El renombramiento al ingles es necesario para que el codigo sea legible y consistente con la literatura del campo.

---

## Ejercicio 2: Transformacion de etiquetas 0 <-> 1

### Que se hizo

Se invirtieron los valores de la variable `credit_risk` usando la operacion `df['credit_risk'] = 1 - df['credit_risk']`.

**Antes de la transformacion:**
- `1` = Buen cliente (prestamo reembolsado)
- `0` = Mal cliente (prestamo no reembolsado)

**Despues de la transformacion:**
- `1` = Mal cliente (default / riesgo)
- `0` = Buen cliente (sin riesgo)

### Toma de decisiones

Se eligio `1 - df['credit_risk']` por su simplicidad algebraica: cuando el valor original es 1, se convierte en 0, y viceversa. Una alternativa habria sido `.map({0:1, 1:0})`, pero la operacion aritmetica es mas directa y eficiente.

### Por que se hizo asi

En clasificacion binaria, la **clase positiva** (etiquetada con 1) debe corresponder al **evento de interes** que queremos detectar. En riesgo crediticio, el evento critico es el incumplimiento de pago (mal cliente). Al hacer que `1 = mal cliente`:

1. **Recall se calcula sobre la clase de interes**: `Recall = VP / (VP + FN)`, donde VP son malos clientes correctamente detectados. Optimizar recall maximiza la deteccion de morosos.
2. **Alineacion con convenciones de sklearn**: Funciones como `classification_report` y `recall_score` calculan metricas para la clase positiva (`pos_label=1`) por defecto.
3. **Interpretacion intuitiva**: Un recall alto significa "estamos detectando la mayoria de los malos clientes", que es exactamente lo que la institucion financiera necesita.

---

## Ejercicio 3: Particion entrenamiento/prueba (70/30)

### Que se hizo

Se separaron las variables de entrada (`X`) y la variable de salida (`y`), y se realizo una particion estratificada 70%-30%.

```python
X = df.drop(columns=['credit_risk'])
y = df['credit_risk']
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

### Toma de decisiones

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| `test_size` | 0.3 | El articulo IEEE utiliza 70/30, y las instrucciones lo solicitan explicitamente |
| `random_state` | 42 | Semilla fija para reproducibilidad |
| `stratify` | `y` | Mantiene la proporcion de clases en ambos conjuntos (~70% buenos, ~30% malos) |

### Por que se hizo asi

- **Particion unica (sin validacion)**: Las instrucciones indican explicitamente que, siguiendo la metodologia del articulo, solo se realiza una particion train/test (no train/val/test). Esto es comun en publicaciones cientificas donde se propone una metodologia nueva.
- **Estratificacion**: Es critica en datos desbalanceados. Sin `stratify=y`, podriamos obtener un conjunto de prueba con una proporcion de clases muy diferente al original, lo que sesga la evaluacion.
- **Distribucion resultante**: ~70% clase 0 (buenos) y ~30% clase 1 (malos) en ambos conjuntos, preservando el desbalance original del dataset.

---

## Ejercicio 4: Clasificacion de variables segun Tabla 3 del articulo

### Que se hizo

Se clasificaron las 20 variables de entrada en tres categorias segun la Tabla 3 del articulo IEEE y la documentacion UCI:

**Variables numericas (quantitative) - 3 variables:**
- `duration`: Duracion del credito en meses (continua)
- `amount`: Monto del credito en marcos alemanes (continua)
- `age`: Edad del deudor en anos (continua)

**Variables ordinales (ordinal, discretized quantitative) - 6 variables:**
- `employment_duration`: Duracion del empleo actual (ordinal discretizado)
- `installment_rate`: Tasa de cuota como % del ingreso disponible (ordinal discretizado)
- `present_residence`: Tiempo en la residencia actual (ordinal discretizado)
- `property`: Propiedad mas valiosa del deudor (ordinal codificado)
- `number_credits`: Numero de creditos en este banco (ordinal discretizado)
- `job`: Calidad del empleo del deudor (ordinal)

**Variables nominales/categoricas + binarias - 11 variables:**
- `status`: Estado de cuenta corriente (categorica)
- `credit_history`: Historial de cumplimiento (categorica)
- `purpose`: Proposito del credito (categorica)
- `savings`: Ahorros del deudor (categorica)
- `personal_status_sex`: Estado civil y sexo combinados (categorica)
- `other_debtors`: Otros deudores o garantes (categorica)
- `other_installment_plans`: Otros planes de pago (categorica)
- `housing`: Tipo de vivienda (categorica)
- `people_liable`: Personas dependientes economicamente (binaria)
- `telephone`: Tiene telefono fijo registrado (binaria)
- `foreign_worker`: Es trabajador extranjero (binaria)

**Total: 3 + 6 + 11 = 20 variables de entrada**

### Toma de decisiones

La clasificacion se baso en:

1. **Documentacion UCI**: Cada variable incluye una descripcion explicitando su tipo (quantitative, ordinal, categorical, binary).
2. **Tabla 3 del articulo IEEE**: Presenta la descripcion de cada atributo del South German Credit dataset con su tipo.
3. **Naturaleza de los datos**: Variables como `duration`, `amount` y `age` son mediciones continuas reales. Variables como `employment_duration` y `installment_rate`, aunque parecen numericas, son realmente discretizaciones de rangos y tienen un orden intrinseco. Variables como `purpose` y `status` no tienen orden natural.

### Por que se hizo asi

La correcta clasificacion de variables es **fundamental** porque determina que transformacion se aplicara a cada una:
- Las variables **numericas** necesitan escalamiento (MinMax) para que modelos sensibles a magnitudes (KNN, SVM, MLP) funcionen correctamente.
- Las variables **ordinales** deben preservar su orden pero no necesitan one-hot encoding (que destruiria la relacion de orden).
- Las variables **nominales** no tienen orden, por lo que requieren one-hot encoding para evitar que el modelo asuma una relacion ordinal inexistente.

---

## Ejercicio 5: Analisis descriptivo y exploratorio

### Que se hizo

Se realizaron seis tipos de analisis sobre el conjunto de entrenamiento:

1. **Balance de clases**: Grafico de barras mostrando conteo y proporcion de cada clase.
2. **Estadisticas descriptivas**: `describe()` sobre las tres variables numericas.
3. **Histogramas por clase**: Distribucion de `duration`, `amount` y `age` separada por clase de riesgo.
4. **Boxplots por clase**: Comparacion de quartiles de variables numericas entre buenos y malos clientes.
5. **Matriz de correlacion**: Heatmap de correlaciones de Pearson entre todas las variables.
6. **Variables categoricas clave**: Graficos de barras apiladas mostrando proporcion de riesgo para `status`, `credit_history`, `purpose` y `savings`.

### Toma de decisiones

- Se uso **solo el conjunto de entrenamiento** (`Xtrain` + `ytrain`) para el analisis descriptivo, evitando incorporar informacion del conjunto de prueba en las decisiones de modelado (prevencion de data leakage).
- Se seleccionaron `status`, `credit_history`, `purpose` y `savings` para los graficos categoricos porque son las variables que, segun la literatura del dominio, tienen mayor influencia en el riesgo crediticio.
- Se incluyo la matriz de correlacion completa (no solo variables numericas) porque en este dataset todas las variables estan codificadas numericamente, lo que permite calcular correlaciones aunque algunas sean ordinales o nominales.

### Por que se hizo asi

El analisis exploratorio cumple multiples propositos:
- **Confirmar el desbalance**: Verifica que la proporcion 70/30 (buenos/malos) se mantiene en el conjunto de entrenamiento, justificando el uso de SMOTE.
- **Identificar patrones**: Los histogramas y boxplots revelan que creditos de mayor duracion y monto, asi como deudores jovenes, presentan mayor riesgo.
- **Detectar multicolinealidad**: La matriz de correlacion confirma que no hay correlaciones extremas entre features, lo cual es favorable para modelos lineales.
- **Orientar la interpretacion**: Los graficos categoricos muestran que `status` (estado de cuenta) es un discriminador muy fuerte, lo que se confirma posteriormente en el analisis de importancia de features.

---

## Ejercicio 6: ColumnTransformer con transformaciones diferenciadas

### Que se hizo

Se construyo un `ColumnTransformer` que aplica tres transformaciones distintas segun el tipo de variable:

```python
# 6a) Numericas: MinMaxScaler (normaliza al rango [0, 1])
num_pipe = Pipeline(steps=[('minmax', MinMaxScaler())])

# 6b) Nominales: OneHotEncoder (crea columnas binarias por categoria)
nom_pipe = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore',
                                                     sparse_output=False))])

# 6c) Ordinales: OrdinalEncoder (preserva el orden numerico)
ord_pipe = Pipeline(steps=[('ordinal', OrdinalEncoder())])
```

### Toma de decisiones

| Transformacion | Aplicada a | Justificacion |
|----------------|-----------|---------------|
| `MinMaxScaler` | `duration`, `amount`, `age` | El articulo IEEE usa explicitamente Min-Max normalization (Ecuacion 1: `X_norm = (X - X_min) / (X_max - X_min)`). Ademas, modelos como KNN, SVM y MLP son sensibles a la escala de las variables. |
| `OneHotEncoder` | 11 variables nominales | Variables categoricas sin orden intrinseco. `handle_unknown='ignore'` evita errores si aparece una categoria nueva en test. `sparse_output=False` retorna un array denso compatible con todos los modelos. |
| `OrdinalEncoder` | 6 variables ordinales | Preserva el orden natural de las categorias (ej: empleo de 0 a 4 anos < 4 a 7 anos). No se usa one-hot porque destruiria la relacion de orden. |

**Parametro `remainder='passthrough'`**: Se incluyo para que cualquier variable no mencionada explicitamente pase sin transformar, como medida de seguridad.

### Por que se hizo asi

La instruccion 6 del PDF dice explicitamente: *"Cuidando no llevar a cabo el filtrado de informacion, utiliza la clase ColumnTransformer"*. Esto significa:

1. **Prevencion de data leakage**: Al encapsular las transformaciones en un `ColumnTransformer` que luego se inserta en un `Pipeline`, el `fit` del scaler/encoder se hace **solo con los datos de entrenamiento** de cada fold de validacion cruzada. Si hicieramos `MinMaxScaler().fit_transform()` sobre todo el dataset antes del split, estariamos filtrando informacion del test al train.

2. **Transformaciones diferenciadas**: La Tabla 3 del articulo clasifica cada variable, y la instruccion dice que cada tipo requiere una transformacion diferente. `ColumnTransformer` es la herramienta de sklearn disenada exactamente para este proposito.

3. **Aumento de dimensionalidad**: El one-hot encoding de 11 variables nominales genera multiples columnas binarias (una por cada categoria), lo cual aumenta la dimension del dataset. Esto se verifica en Cell 22 donde se imprime el shape antes y despues de las transformaciones.

---

## Ejercicio 7: Concatenacion de conjuntos para validacion cruzada

### Que se hizo

Se concatenaron `Xtrain + Xtest` en `Xtt` y `ytrain + ytest` en `ytt`:

```python
Xtt = pd.concat([Xtrain, Xtest], axis=0).reset_index(drop=True)
ytt = pd.concat([ytrain, ytest], axis=0).reset_index(drop=True)
```

### Toma de decisiones

- Se uso `pd.concat` con `axis=0` para concatenar por filas.
- Se aplico `reset_index(drop=True)` para obtener un indice limpio y secuencial, evitando indices duplicados que podrian causar problemas en operaciones posteriores.

### Por que se hizo asi

Las instrucciones indican que se usara **validacion cruzada** para evaluar los modelos. La validacion cruzada necesita el dataset completo porque ella misma realiza las particiones internas en cada fold. Si solo usaramos `Xtrain`/`ytrain`, estariamos desperdiciando el 30% de los datos de prueba.

Al concatenar train + test en `Xtt`/`ytt`, la validacion cruzada (RepeatedStratifiedKFold) puede:
1. Crear sus propias particiones internas train/validation
2. Usar todos los 1000 registros disponibles
3. Aplicar el `ColumnTransformer` dentro del pipeline en cada fold, evitando data leakage

---

## Ejercicio 8: Justificacion de recall y calculo del baseline

### 8a) Justificacion del recall

### Que se hizo

Se justifico la eleccion de **recall (exhaustividad)** como metrica principal para el problema de riesgo crediticio.

### Razonamiento

En el contexto crediticio existen dos tipos de errores:

| Error | Significado | Costo |
|-------|-------------|-------|
| **Falso Negativo** (FN) | No detectar a un mal cliente y darle el credito | **Alto**: Se pierde el monto total del credito |
| **Falso Positivo** (FP) | Rechazar a un buen cliente | **Bajo**: Se pierde la ganancia potencial por intereses |

Dado que el costo de un FN >>> costo de un FP, queremos **minimizar los falsos negativos**. El recall mide exactamente esto:

```
Recall = VP / (VP + FN)
```

Un recall de 0.80 significa que de cada 100 malos clientes reales, el modelo detecta 80 y deja pasar 20. Maximizar recall minimiza los 20 que se escapan.

**Alternativas consideradas y descartadas:**
- **Accuracy**: No es adecuada con datos desbalanceados. Un modelo que prediga siempre "buen cliente" tendria ~70% de accuracy pero 0% de recall.
- **Precision**: Minimiza falsos positivos, pero no es prioritario aqui porque rechazar a un buen cliente es menos costoso.
- **F1-Score**: Es el promedio armonico de precision y recall, pero no prioriza suficientemente la deteccion de malos clientes.

### 8b) Calculo del baseline

### Que se hizo

Se calculo el umbral baseline para recall como la proporcion de la clase positiva en el dataset.

```python
baseline_recall = n_positivos / n_total  # 300 / 1000 = 0.30
```

### Razonamiento

Un **clasificador aleatorio estratificado** (baseline/dummy) predice la clase 1 con probabilidad igual a la proporcion de la clase 1 en los datos:

- P(prediccion = 1) = 300/1000 = 0.30
- Para cualquier instancia real de clase 1: P(correctamente detectada) = P(prediccion = 1) = 0.30
- Por lo tanto: Recall_baseline = 0.30

**Interpretacion**: Cualquier modelo con recall <= 0.30 no es mejor que adivinar aleatoriamente, lo que indica subentrenamiento. Este umbral se grafica como linea roja punteada en el boxplot del Ejercicio 9 para facilitar la comparacion visual.

---

## Ejercicio 9: Entrenamiento de 7 modelos con fine-tuning

### Que se hizo

Se definieron 7 modelos de clasificacion con hiperparametros ajustados, se selecciono SMOTE como tecnica de sobremuestreo, y se evaluaron todos mediante validacion cruzada repetida (5 folds x 3 repeticiones).

### Modelos y sus hiperparametros

#### 1. Regresion Logistica (LR)

```python
LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=1)
```

| Hiperparametro | Valor | Razon |
|----------------|-------|-------|
| `max_iter` | 1000 | Suficientes iteraciones para garantizar convergencia con datos transformados |
| `solver` | `lbfgs` | Solver eficiente para datasets pequenos, soporta `class_weight` |
| `class_weight` | `balanced` | Ajusta automaticamente los pesos inversamente proporcional a la frecuencia de clase, dando mas peso a la clase minoritaria (malos clientes) |

#### 2. K-Vecinos mas Cercanos (kNN)

```python
KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski')
```

| Hiperparametro | Valor | Razon |
|----------------|-------|-------|
| `n_neighbors` | 7 | Valor impar para evitar empates; ni muy pequeno (sobreajuste) ni muy grande (subajuste) |
| `weights` | `distance` | Los vecinos mas cercanos tienen mayor influencia, mejorando la decision en fronteras |
| `metric` | `minkowski` | Metrica euclidiana generalizada, apropiada para datos normalizados con MinMax |

#### 3. Arbol de Decisiones (DTree)

```python
DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=5,
                       class_weight='balanced', random_state=1)
```

| Hiperparametro | Valor | Razon |
|----------------|-------|-------|
| `max_depth` | 8 | Limita la profundidad para evitar sobreajuste en un dataset de solo 1000 registros |
| `min_samples_split` | 10 | Evita divisiones en nodos con pocas muestras |
| `min_samples_leaf` | 5 | Cada hoja necesita al menos 5 muestras, previniendo hojas ruidosas |
| `class_weight` | `balanced` | Compensa el desbalance de clases |

#### 4. Bosque Aleatorio (RF)

```python
RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                       min_samples_leaf=3, class_weight='balanced', random_state=1)
```

| Hiperparametro | Valor | Razon |
|----------------|-------|-------|
| `n_estimators` | 200 | Numero suficiente de arboles para estabilidad; mas de 200 tiene rendimientos decrecientes |
| `max_depth` | 10 | Mayor que DTree individual porque el ensamble reduce la varianza |
| `min_samples_split` | 5 | Menos restrictivo que DTree porque la agregacion reduce el sobreajuste |
| `class_weight` | `balanced` | Combinado con SMOTE proporciona doble manejo del desbalance |

#### 5. XGBoost

```python
XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
              scale_pos_weight=2.33, eval_metric='logloss',
              use_label_encoder=False, random_state=1)
```

| Hiperparametro | Valor | Razon |
|----------------|-------|-------|
| `n_estimators` | 200 | Numero de arboles en el ensemble de boosting |
| `max_depth` | 5 | Arboles mas superficiales que RF porque boosting construye secuencialmente |
| `learning_rate` | 0.1 | Tasa de aprendizaje moderada; equilibrio entre velocidad y estabilidad |
| `scale_pos_weight` | 2.33 | Aproximadamente 700/300 = ratio de desbalance; da mas peso a la clase positiva |
| `eval_metric` | `logloss` | Metrica interna de evaluacion; log-loss es estandar para clasificacion binaria |

#### 6. Perceptron Multicapa (MLP)

```python
MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
              solver='adam', max_iter=500, early_stopping=True, random_state=1)
```

| Hiperparametro | Valor | Razon |
|----------------|-------|-------|
| `hidden_layer_sizes` | (64, 32) | Dos capas ocultas con reduccion gradual; suficiente capacidad sin exceso para 1000 muestras |
| `activation` | `relu` | Funcion de activacion estandar, computacionalmente eficiente |
| `solver` | `adam` | Optimizador adaptativo, funciona bien en la mayoria de casos |
| `max_iter` | 500 | Iteraciones suficientes para convergencia |
| `early_stopping` | `True` | Detiene el entrenamiento si la perdida en validacion no mejora, previniendo sobreajuste |

#### 7. Maquina de Vectores de Soporte (SVM)

```python
SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=1)
```

| Hiperparametro | Valor | Razon |
|----------------|-------|-------|
| `kernel` | `rbf` | Kernel gaussiano, capaz de capturar fronteras no lineales |
| `C` | 1.0 | Valor por defecto; equilibrio entre margen amplio y clasificacion correcta |
| `gamma` | `scale` | `1 / (n_features * X.var())`, se adapta automaticamente a la dimension de los datos |
| `class_weight` | `balanced` | Compensa el desbalance de clases |

### Tecnica de sobremuestreo: SMOTE

```python
mi_uoSampling = SMOTE(random_state=1)
```

**Por que SMOTE y no otra tecnica:**
- El articulo IEEE concluye que **oversampling supera a undersampling** en datasets de credito.
- SMOTE genera **muestras sinteticas** de la clase minoritaria interpolando entre vecinos cercanos, en lugar de simplemente duplicar registros (como Random Oversampling).
- Al generar datos sinteticos en lugar de duplicar, SMOTE reduce el riesgo de sobreajuste a instancias especificas de la clase minoritaria.
- Se coloca dentro del `ImbPipeline` para que el sobremuestreo se aplique **solo a los datos de entrenamiento** de cada fold, evitando data leakage.

### Estructura del Pipeline

```python
pipeline = ImbPipeline(steps=[
    ('ct', columnasTransformer),    # Paso 1: Transformar variables
    ('uos', mi_uoSampling),        # Paso 2: Sobremuestreo SMOTE
    ('m', modelos[i])               # Paso 3: Modelo de clasificacion
])
```

Se usa `ImbPipeline` (de `imblearn`) en lugar del `Pipeline` estandar de sklearn porque el pipeline estandar **no soporta pasos de resampling** como SMOTE. `ImbPipeline` ejecuta el resampling solo durante `fit()`, no durante `predict()`.

### Validacion cruzada

```python
micv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=5)
```

- **5 folds**: Cada fold usa 80% para entrenamiento y 20% para validacion.
- **3 repeticiones**: Se repite el proceso 3 veces con diferentes particiones aleatorias.
- **Total**: 15 evaluaciones por modelo, proporcionando una estimacion robusta del desempenio.
- **Estratificado**: Mantiene la proporcion de clases en cada fold.

### Boxplot de comparacion

Se genero un grafico de cajas mostrando la distribucion del recall en validacion para cada modelo, con una linea roja punteada indicando el baseline (0.30). Esto permite verificar visualmente:
- Que todos los modelos superan el baseline
- Que modelo tiene mejor recall mediano
- Que modelo es mas estable (caja mas compacta)
- Si algun modelo tiene valores atipicos preocupantes

---

## Ejercicio 10: Reporte de metricas e importancia de factores

### 10a) Reporte de metricas

### Que se hizo

Se construyo una tabla resumen con las metricas promedio de validacion cruzada para cada modelo:

| Metrica | Descripcion |
|---------|-------------|
| Train Accuracy | Accuracy promedio en entrenamiento |
| Val Accuracy | Accuracy promedio en validacion |
| Train Recall | Recall promedio en entrenamiento |
| Val Recall | Recall promedio en validacion |
| Diff Recall | Diferencia Train - Val (indicador de sobreajuste) |

Ademas, se entreno cada modelo con `Xtrain`/`ytrain` y se evaluo con `Xtest`/`ytest` usando `classification_report`, que reporta precision, recall, f1-score y support para cada clase.

### Toma de decisiones

- **Tabla ordenada por Val Recall descendente**: Facilita identificar rapidamente el mejor modelo segun la metrica principal.
- **Columna Diff Recall**: Una diferencia grande (> 0.10) entre train y val recall sugiere sobreajuste. Si Train Recall es muy superior a Val Recall, el modelo memoriza el entrenamiento pero no generaliza.
- **Evaluacion en test con classification_report**: Proporciona una evaluacion final independiente, complementando los resultados de la validacion cruzada.

### 10b) Importancia de factores

### Que se hizo

Se utilizo el Random Forest entrenado para extraer la importancia de cada feature mediante el atributo `feature_importances_`. Se genero un grafico de barras horizontales con las 20 features mas importantes.

### Toma de decisiones

- **Se uso Random Forest** para importancia de features porque: (a) es un modelo basado en arboles con atributo `feature_importances_` nativo, (b) no requiere suposiciones de linealidad, (c) mide la reduccion de impureza (Gini) atribuida a cada feature.
- **Se reconstruyeron los nombres de features** despues de las transformaciones, ya que el `ColumnTransformer` + `OneHotEncoder` genera nuevos nombres para las columnas one-hot encoded (ej: `status_1`, `status_2`, etc.).
- **Top 20** features para el grafico: suficiente para mostrar las mas relevantes sin saturar visualmente.

### Por que se hizo asi

El analisis de importancia de factores permite:
1. **Interpretar** que variables son mas decisivas para la prediccion de riesgo.
2. **Validar** que los resultados son coherentes con el dominio del problema (ej: si `status` es la mas importante, tiene sentido financiero).
3. **Guiar** posibles mejoras futuras: se podrian eliminar features de baja importancia para simplificar el modelo sin perder desempenio.

---

## Ejercicio 11: Conclusiones finales

### Que se hizo

Se redactaron conclusiones que integran todos los hallazgos de la actividad, cubriendo:

1. **Contexto**: Problema de clasificacion binaria con desbalance 70/30.
2. **Preprocesamiento**: Transformaciones diferenciadas (MinMax, OHE, Ordinal) encapsuladas en Pipeline para evitar data leakage.
3. **Metrica**: Recall como metrica apropiada dado el costo asimetrico de los errores.
4. **Desbalance**: SMOTE como tecnica de sobremuestreo, consistente con los hallazgos del articulo.
5. **Modelos**: Comparacion de 7 algoritmos mediante validacion cruzada repetida.
6. **Factores**: Variables mas importantes alineadas con la literatura financiera.
7. **Aprendizaje**: Importancia de un pipeline completo e integrado.

---

## Resumen de decisiones tecnicas clave

| Decision | Alternativa descartada | Razon de la eleccion |
|----------|----------------------|---------------------|
| `1 - df['credit_risk']` para invertir labels | `.map({0:1, 1:0})` | Mas simple y directo |
| `stratify=y` en train_test_split | Sin estratificacion | Preserva proporciones en datos desbalanceados |
| `MinMaxScaler` para numericas | `StandardScaler` | El articulo IEEE usa explicitamente Min-Max |
| `OneHotEncoder` para nominales | `LabelEncoder` | OHE no asume orden; LabelEncoder introduciria orden artificial |
| `OrdinalEncoder` para ordinales | `OneHotEncoder` | Preserva el orden natural de las categorias |
| `SMOTE` para sobremuestreo | `RandomUnderSampler` | El articulo muestra que oversampling > undersampling |
| `ImbPipeline` | `Pipeline` de sklearn | Necesario para integrar SMOTE en el pipeline |
| `class_weight='balanced'` en modelos | Sin pesos de clase | Doble proteccion contra desbalance junto con SMOTE |
| `RepeatedStratifiedKFold(5, 3)` | Simple KFold | Estratificado preserva proporciones; repetido da estimaciones robustas |
| `early_stopping=True` en MLP | Sin early stopping | Previene sobreajuste en redes neuronales |

---

## Flujo completo del pipeline

```
Datos originales (1000 x 20)
    |
    v
Renombramiento de columnas (aleman -> ingles)
    |
    v
Inversion de etiquetas (1=mal, 0=buen)
    |
    v
Train/Test Split (70/30, estratificado)
    |
    v
Concatenacion Xtt/ytt (para validacion cruzada)
    |
    v
[Dentro de cada fold de CV:]
    |
    +-> ColumnTransformer:
    |       - MinMaxScaler (3 vars numericas)
    |       - OneHotEncoder (11 vars nominales)
    |       - OrdinalEncoder (6 vars ordinales)
    |
    +-> SMOTE (sobremuestreo de clase minoritaria)
    |
    +-> Modelo (LR, KNN, DTree, RF, XGBoost, MLP, SVM)
    |
    v
Evaluacion: accuracy y recall (train y validacion)
    |
    v
Comparacion: boxplot de recall + tabla de metricas
    |
    v
Importancia de features (Random Forest)
    |
    v
Conclusiones
```
