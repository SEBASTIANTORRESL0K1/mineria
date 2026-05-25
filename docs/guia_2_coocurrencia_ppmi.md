# Guía de Estudio 2: El Algoritmo Neuronal Skip-Gram (Matemáticas del Modelo)

En esta guía abordaremos el núcleo matemático y el modelado neuronal de **Skip-Gram (Word2Vec)** que se ejecuta en **`embeddings_model.py`**. Veremos cómo pasamos de parejas de palabras a una representación de coordenadas densas (embeddings) optimizada mediante descenso de gradiente.

---

## 1. La Ventana de Contexto Dinámica y sus Parejas

Para aprender qué palabras se relacionan, el sistema recorre los tokens y define una "ventana de contexto" (p. ej. `VENTANA = 2`).

### Cómo funciona matemáticamente:
Para cada palabra objetivo en la posición $i$, se evalúan sus vecinos desde $i - \text{ventana}$ hasta $i + \text{ventana}$ (excluyendo la posición $i$ misma). Con esto se forman las parejas de entrenamiento:
$$\text{Pareja} = (\text{Palabra Objetivo}, \text{Palabra Contexto})$$

### Tratamiento de Límites (Fronteras):
Si la palabra objetivo está al principio del texto ($i=0$), no tiene palabras previas. Si está al final, no tiene posteriores. Usamos límites lógicos dinámicos para evitar desbordar los índices del texto:
```python
inicio = max(0, i - ventana)
fin    = min(len(tokens), i + ventana + 1)
```
* **`max(0, i - ventana)`**: Garantiza que si `i - ventana` da un número negativo, el inicio se acote a `0` (el inicio del texto).
* **`min(len(tokens), i + ventana + 1)`**: Garantiza que la ventana no intente leer más allá de la última palabra del texto, previniendo errores de desbordamiento.

---

## 2. La Arquitectura de la Red Neuronal

A diferencia de PPMI y SVD, Skip-Gram utiliza un enfoque de **aprendizaje de red neuronal superficial**. La red consta de dos conjuntos de pesos sinápticos principales:

1. **Matriz de Entrada ($W_{\text{in}}$) de tamaño $V \times d$**:
   * Cada fila de esta matriz representa el **embedding de la palabra objetivo** (su representación semántica densa de $d=50$ dimensiones).
2. **Matriz de Salida ($W_{\text{out}}$) de tamaño $d \times V$**:
   * Cada columna de esta matriz representa el **vector de contexto** de una palabra.

---

## 3. Propagación Hacia Adelante (Forward Pass) y Softmax

Dada una palabra objetivo con índice $i$ en nuestro vocabulario:

1. **Capa Oculta ($h$):** Se selecciona directamente el vector correspondiente en la matriz de pesos de entrada:
   $$h = W_{\text{in}}[i, :] \quad (\text{vector de tamaño } d)$$
2. **Logits de Salida ($z$):** Se calcula el producto punto entre la representación oculta y todas las columnas de la matriz de salida:
   $$z = h \cdot W_{\text{out}} \quad (\text{vector de tamaño } V)$$
3. **Distribución de Probabilidad ($\hat{y}$):** Se aplica la función de activación **Softmax** para convertir los logits en probabilidades continuas normalizadas que suman 1. Para evitar que números muy grandes causen inestabilidad numérica, restamos el valor máximo de $z$ antes de calcular los exponenciales:
   $$\hat{y}_k = \text{softmax}(z)_k = \frac{e^{z_k - \max(z)}}{\sum_{m=1}^V e^{z_m - \max(z)}}$$

---

## 4. Función de Pérdida (Entropía Cruzada)

El objetivo de la red es maximizar la probabilidad de predecir la palabra de contexto real (con índice $j$) dada la palabra objetivo. Esto equivale a minimizar la pérdida de **Entropía Cruzada**:
$$L = -\log P(w_{\text{contexto}} | w_{\text{objetivo}}) = -\log \hat{y}_j$$

---

## 5. Retropropagación del Error (Backpropagation) y SGD

El modelo ajusta sus pesos mediante **Gradiente Descendiente Estocástico (SGD)**. Para cada pareja de entrenamiento:

1. **Gradiente del Error de Salida ($e$):**
   $$e = \hat{y} - y$$
   Donde $y$ es un vector *one-hot* de tamaño $V$ con un 1 en la posición de la palabra de contexto real $j$. Por lo tanto:
   * Para la palabra de contexto real: $e_j = \hat{y}_j - 1$
   * Para el resto de palabras: $e_k = \hat{y}_k$
2. **Gradiente de Pesos de Salida ($dW_{\text{out}}$):**
   $$dW_{\text{out}} = h^T \cdot e \quad (\text{producto exterior, de tamaño } d \times V)$$
3. **Gradiente de la Representación Oculta ($dh$):**
   $$dh = W_{\text{out}} \cdot e \quad (\text{de tamaño } d)$$
4. **Regla de Actualización de Pesos (con Learning Rate $\eta = 0.05$):**
   * Ajustamos los pesos de contexto de todas las palabras:
     $$W_{\text{out}} \leftarrow W_{\text{out}} - \eta \cdot dW_{\text{out}}$$
   * Ajustamos únicamente el embedding de la palabra objetivo activa:
     $$W_{\text{in}}[i, :] \leftarrow W_{\text{in}}[i, :] - \eta \cdot dh$$

Al repetir este ciclo de aprendizaje iterativo a lo largo de las épocas, la red neuronal "empuja" los vectores de palabras que aparecen en contextos similares a direcciones cercanas del espacio multidimensional.
