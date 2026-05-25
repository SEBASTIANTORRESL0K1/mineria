# Guía de Estudio 3: Embeddings Semánticos y Similitud Coseno

En esta guía veremos el concepto de **Embeddings**, cómo nuestro sistema almacena estos modelos entrenados de forma persistente y el funcionamiento matemático del motor de **Búsqueda Semántica**.

---

## 1. ¿Qué es un Embedding?

Un **Embedding** es una representación vectorial de una palabra en un espacio continuo de baja dimensión. 

En lugar de representar las palabras de manera discreta o aislada (donde cada palabra es una entidad separada sin relación matemática con otras, como en One-Hot), un embedding asigna a cada palabra un vector de números reales (en nuestro caso, 50 números). 

Si dos palabras tienen significados parecidos o se usan en contextos similares en el texto, sus vectores apuntarán en direcciones casi idénticas en ese mapa de 50 dimensiones.

---

## 2. Guardado y Carga Persistente (Carga Diferida)

Para la presentación del proyecto, la rúbrica exige que la fase de entrenamiento ya esté terminada y que **no sea necesario reentrenar** el modelo frente al profesor.

Nuestro sistema resuelve esto en **`embeddings_model.py`** exportando e importando las estructuras:

### Guardado (Al finalizar la Opción 1):
* **`embeddings.npy`**: Formato binario nativo de NumPy. Es extremadamente rápido de leer/escribir y guarda la matriz de 50 dimensiones con precisión matemática exacta.
* **`vocabulario.csv`**: Un archivo de texto plano donde se guarda la lista de palabras únicas, su índice en la matriz y su frecuencia de aparición.
* **`tokens_finales.csv` y `parejas_ventana.csv`**: Archivos de soporte que evidencian físicamente el entrenamiento y permiten verificar en Excel qué tokens y relaciones se usaron.

### Carga Inmediata (Al iniciar la Opción 2):
La función `cargar_modelo()` lee únicamente los archivos `embeddings.npy` y `vocabulario.csv`. Gracias a esto, el sistema se inicializa de forma instantánea para consultas semánticas sin consumir tiempo ni recursos procesando el texto original de nuevo.

---

## 3. La Similitud Coseno: Matemáticas de la Búsqueda

Para evaluar qué tan parecidas son dos palabras, el sistema calcula la **Similitud Coseno** de sus vectores de embeddings correspondientes.

### ¿Por qué no usamos la distancia clásica (Euclidiana)?
La distancia clásica (medir en línea recta de la punta de un vector a otro) tiene un gran defecto en NLP: si una palabra aparece muchísimo más que otra en el texto, sus vectores tendrán magnitudes (longitudes) muy distintas, haciendo que la distancia geométrica sea enorme aunque su significado conceptual sea idéntico.

La **Similitud Coseno** soluciona esto midiendo únicamente el **ángulo** entre los dos vectores, ignorando qué tan largos son.

### La Fórmula de Similitud Coseno:
$$\text{Similitud Coseno}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{d} A_i B_i}{\sqrt{\sum_{i=1}^{d} A_i^2} \sqrt{\sum_{i=1}^{d} B_i^2}}$$

* **Explicación sencilla de la fórmula:**
  * El numerador es el **Producto Punto** de los dos vectores (la multiplicación elemento a elemento acumulada).
  * El denominador es la multiplicación de las **Normas** (magnitudes o longitudes) de los vectores $A$ y $B$, lo cual sirve para normalizar los vectores a una longitud estándar de 1.
* **Interpretación del resultado:**
  * **$1.0$ (o cercano):** El ángulo es $0^\circ$. Los vectores apuntan en la misma dirección. Las palabras son semánticamente idénticas en el texto.
  * **$0.0$:** Los vectores son perpendiculares ($90^\circ$). Las palabras no tienen ninguna relación contextual en el texto.
  * **$-1.0$:** Los vectores apuntan en direcciones opuestas ($180^\circ$).
