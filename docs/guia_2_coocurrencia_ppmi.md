# Guía de Estudio 2: Coocurrencia, PPMI y SVD (Matemáticas del Modelo)

En esta guía abordaremos el núcleo matemático y de procesamiento conceptual del modelo que se ejecuta en **`embeddings_model.py`**. Veremos cómo pasamos de una secuencia lineal de palabras limpias a una representación de coordenadas densas (embeddings).

---

## 1. La Ventana de Contexto Dinámica y sus Límites

Para aprender qué palabras se relacionan, el sistema recorre los tokens y define una "ventana de contexto" (p. ej. `VENTANA = 2`).

### Cómo funciona matemáticamente:
Para cada palabra objetivo en la posición $i$, se evalúan sus vecinos desde $i - \text{ventana}$ hasta $i + \text{ventana}$.

### Tratamiento de Límites (Fronteras):
Si la palabra objetivo es la primera palabra del texto ($i=0$), no tiene palabras previas. Si es la última, no tiene posteriores. Para que el programa no falle intentando leer índices que no existen en la lista, usamos límites lógicos dinámicos:
```python
inicio = max(0, i - ventana)
fin    = min(len(tokens), i + ventana + 1)
```
* **`max(0, i - ventana)`**: Garantiza que si `i - ventana` da un número negativo (fuera del rango izquierdo), el inicio se acote a `0` (el inicio del texto).
* **`min(len(tokens), i + ventana + 1)`**: Garantiza que la ventana no intente leer más allá de la última palabra del texto, previniendo errores de desbordamiento.

---

## 2. La Matriz de Coocurrencia

La matriz de coocurrencia $M$ es una cuadrícula de tamaño $V \times V$ (donde $V$ es el tamaño de nuestro vocabulario único).
* Cada fila y columna representa una palabra única.
* La celda $M_{i,j}$ almacena el conteo acumulativo de veces que la palabra de columna $j$ aparece dentro de la ventana de contexto de la palabra de fila $i$.

Si dos palabras aparecen juntas muy seguido en el texto, el valor de su celda será muy alto.

---

## 3. PPMI (Positive Pointwise Mutual Information)

Aunque ya filtramos stopwords, palabras que son comunes por naturaleza en el idioma (como *"hacer"*, *"decir"*) van a tener conteos de coocurrencia inflados. Para solucionar esto, aplicamos **PMI** (Información Mutua Puntual).

### La Fórmula de PMI:
$$\text{PMI}(w, c) = \log_2 \left( \frac{P(w, c)}{P(w)P(c)} \right)$$

* **¿Qué significa?** Divide la probabilidad de que la palabra $w$ y su contexto $c$ aparezcan juntos ($P(w,c)$) entre la probabilidad individual de que ocurran por separado ($P(w) \cdot P(c)$). 
* Si aparecen juntos mucho más de lo que ocurriría por puro azar, el valor de PMI es **positivo**. Si aparecen menos de lo esperado, es **negativo**.

### Por qué usamos PPMI (Positive PMI):
Cuando dos palabras nunca han aparecido juntas en el texto, la probabilidad conjunta es $0$, lo que haría que $\log_2(0)$ tienda a $-\infty$ (un error matemático). Además, las relaciones menores al azar no nos interesan. Por eso, aplicamos una función "puerta" que reemplaza cualquier valor negativo o infinito por cero:

$$\text{PPMI}(w, c) = \max(0, \text{PMI}(w, c))$$

Esto limpia la matriz y resalta únicamente las asociaciones semánticas fuertes.

---

## 4. SVD (Singular Value Decomposition)

La matriz PPMI es gigante ($V \times V$) y extremadamente "dispersa" (llena de ceros). Procesar búsquedas en ella es ineficiente y contiene mucho ruido estadístico.

Para resolverlo, aplicamos **SVD**, un método matemático de álgebra lineal que descompone nuestra matriz original en tres matrices:
$$\text{PPMI} \approx U \cdot \Sigma \cdot V^T$$

* **Reducción de Dimensionalidad:** Nos quedamos únicamente con las primeras **50 dimensiones** (`DIMENSIONES = 50`) de estas matrices descompuestas, las cuales concentran la mayor cantidad de información y patrones conceptuales del texto.
* **El Producto Final (Embeddings):** Multiplicamos la matriz izquierda singular truncada por los valores singulares:
  $$\text{Embeddings} = U_{:, :50} \cdot \Sigma_{:50, :50}$$
  Esto nos da una matriz densa donde cada fila es una coordenada de 50 números reales que representa la esencia semántica comprimida de cada palabra.
