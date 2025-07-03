# Clasificación de Prendas de Moda con CNN 1D

Proyecto de aprendizaje automático orientado a clasificar distintos tipos de prendas de vestir (como camisetas, jeans, chaquetas y vestidos) a partir de un conjunto de datos tabular con características visuales y textiles. El modelo implementado utiliza una red neuronal convolucional 1D (CNN 1D), adaptada a datos estructurados.

## 1. Acerca del Conjunto de Datos

El conjunto de datos contiene 800 registros, cada uno representando una prenda de ropa. Cada fila del dataset incluye las siguientes variables:

| Columna             | Tipo       | Descripción                                    |
| ------------------- | ---------- | ---------------------------------------------- |
| label               | categórica | Tipo de prenda (t-shirt, jeans, jacket, dress) |
| textile\_thickness  | numérica   | Grosor del textil                              |
| pattern\_complexity | numérica   | Complejidad del patrón visual                  |
| length\_ratio       | numérica   | Proporción del largo respecto al cuerpo        |
| brightness          | numérica   | Nivel de brillo promedio                       |
| color\_\*           | one-hot    | Color predominante de la prenda                |
| season\_\*          | one-hot    | Temporada asociada a la prenda                 |

Particularidades:

* Las etiquetas fueron convertidas a valores enteros consecutivos (0, 1, 2, 3).
* Se duplicaron registros para asegurar al menos dos muestras por clase.
* Se aplicó normalización a las variables numéricas y codificación one-hot a las categóricas.

## 2. Metodología

| Fase                | Detalles                                                                        |
| ------------------- | ------------------------------------------------------------------------------- |
| Preprocesado        | Re-mapeo de etiquetas, codificación one-hot, normalización, reshape a (n, f, 1) |
| División de datos   | 80% entrenamiento, 20% prueba, estratificado si es posible                      |
| Arquitectura CNN 1D | Conv1D(32) > MaxPool > Conv1D(64) > MaxPool > Flatten > Dense > Dropout > Dense |
| Entrenamiento       | 20 épocas, batch de 16, optimizador Adam                                        |
| Pérdida             | Sparse categorical crossentropy                                                 |

## 3. Resultados Obtenidos

Los resultados pueden variar según el muestreo inicial.

| Métrica   | Entrenamiento  | Validación |
| --------- | -------------- | ---------- |
| Precisión | \~0.95         | \~0.85     |
| Pérdida   | baja y estable | estable    |

Matriz de confusión:

Se muestra en la salida gráfica generada al finalizar la ejecución del script.

## 4. Interpretación de Resultados

* La CNN 1D permite aprender patrones secuenciales en atributos numéricos.
* Reformatear los datos a (features, 1) permite aplicar convoluciones sobre columnas.
* El rendimiento mejora cuando se asegura una distribución balanceada entre clases.

## 5. Ejecución del Proyecto

### Entrenamiento y Evaluación:

```bash
python app.py
```

Esto generará:

* Resumen del modelo
* Curvas de precisión y pérdida por época
* Matriz de confusión
* Reporte de clasificación

## 6. Conclusiones

| Observación                               | Recomendación                                 |
| ----------------------------------------- | --------------------------------------------- |
| Modelo funciona con datos estructurados   | Conservar preprocesamiento actual             |
| Dataset limitado afecta generalización    | Ampliar conjunto de datos o aplicar aumento   |
| Clases poco representadas desbalancean    | Aplicar sobremuestreo o recolección adicional |
| CNN 1D se adapta a otras series tabulares | Aplicable en sensores o registros temporales  |

