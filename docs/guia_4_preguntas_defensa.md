# Guía de Estudio 4: Balotario de Preguntas Frecuentes de la Defensa (Skip-Gram)

Esta guía recopila las preguntas más probables y difíciles que los profesores suelen hacer durante la presentación y defensa de este proyecto de minería de texto con el modelo Skip-Gram. Estúdialas a fondo para asegurar la máxima calificación.

---

### Pregunta 1: ¿Por qué el sistema de búsqueda interactiva funciona de forma instantánea y no requiere que carguemos el archivo de texto largo el día de la presentación?
* **Respuesta de 10:** *"Porque implementamos una arquitectura de **Carga Diferida** para separar la fase de entrenamiento de la de inferencia. Al entrenar el modelo (Opción 1), la red neuronal optimiza sus pesos y los guarda en disco en archivos estructurados (`vocabulario.csv` y los vectores matemáticos binarios de la capa oculta en `embeddings.npy`). El día de la presentación, la Opción 2 lee directamente estos archivos en milisegundos sin repetir todo el bucle de entrenamiento, garantizando una demostración rápida y fluida."*

---

### Pregunta 2: ¿Cómo maneja tu código las palabras en los extremos del texto (palabras iniciales y finales) para que la ventana de contexto no desborde la lista?
* **Respuesta de 10:** *"El código implementa **condiciones de frontera lógicas dinámicas** mediante las funciones `max` y `min` de Python en el barrido de la ventana: `inicio = max(0, i - ventana)` y `fin = min(len(tokens), i + ventana + 1)`. Esto garantiza que si una palabra está al principio (índice 0) o al final del texto, la ventana se detenga exactamente en los límites válidos del documento, evitando accesos a índices inexistentes (que romperían el script) y controlando la variación del contexto correctamente."*

---

### Pregunta 3: ¿Qué es el algoritmo Skip-Gram (Word2Vec) y en qué consiste su arquitectura?
* **Respuesta de 10:** *"Skip-Gram es una arquitectura de **red neuronal artificial de una capa oculta lineal** diseñada para aprender embeddings de palabras. En lugar de predecir una palabra objetivo a partir de sus vecinos, Skip-Gram hace lo contrario: **recibe una palabra objetivo en la entrada e intenta predecir las palabras que ocurren en su contexto cercano**. La red consta de dos matrices de pesos principales: $W_{\text{in}}$ (pesos de entrada, que contienen los embeddings de las palabras objetivo) y $W_{\text{out}}$ (pesos de salida, que representan los vectores de contexto)."*

---

### Pregunta 4: ¿Por qué utilizas la función Softmax en la salida de tu red neuronal y qué es la pérdida de Entropía Cruzada?
* **Respuesta de 10:** *"La red neuronal produce puntuaciones sin normalizar llamadas 'logits' para cada palabra del vocabulario. La función **Softmax** es crucial porque convierte estas puntuaciones arbitrarias en una **distribución de probabilidad continua** (valores entre 0 y 1 que suman exactamente 1.0). Por otro lado, la **Entropía Cruzada** es nuestra función de pérdida, la cual evalúa qué tan buena fue la predicción de la red calculando el logaritmo negativo de la probabilidad predicha para la palabra de contexto correcta ($L = -\log \hat{y}_j$). El objetivo del entrenamiento es minimizar esta pérdida."*

---

### Pregunta 5: ¿Cómo funciona el Backpropagation (Retropropagación) y qué parámetros se actualizan en tu código?
* **Respuesta de 10:** *"El backpropagation consiste en **calcular la derivada de la pérdida respecto a cada peso de la red** aplicando la regla de la cadena para propagar los errores hacia atrás. En nuestro código de NumPy desde cero, calculamos el error en la capa de salida como la diferencia entre la distribución predicha y el vector real ($e = \hat{y} - y$). Con esto, obtenemos los gradientes para actualizar dos conjuntos de parámetros:
  1. La matriz de pesos de salida **$W_{\text{out}}$** (usando el producto exterior $h^T \cdot e$).
  2. La fila de la matriz de pesos de entrada **$W_{\text{in}}[i, :]$** que representa el vector de la palabra objetivo activa (usando $W_{\text{out}} \cdot e$).
  Ambos se actualizan mediante Gradiente Descendiente Estocástico (SGD) restando el gradiente multiplicado por el learning rate."*

---

### Pregunta 6: ¿Por qué elegiste usar "Similitud Coseno" para la búsqueda semántica en lugar de la distancia clásica (Euclidiana)?
* **Respuesta de 10:** *"En lingüística computacional, la distancia euclidiana (medir en línea recta) es engañosa porque es sensible a la longitud de los vectores. Si una palabra aparece muchísimas más veces en el texto que otra, su vector será enorme en longitud, dando una distancia euclidiana gigante aunque tengan significados idénticos. La **Similitud Coseno** soluciona esto porque **mide únicamente el ángulo entre los vectores**, ignorando su longitud. Así, se evalúa únicamente la dirección conceptual de las palabras."*

---

### Pregunta 7: ¿Por qué fue una buena práctica de ingeniería de software separar el código en módulos (`text_processor.py`, `embeddings_model.py`, etc.) en lugar de tener todo en un solo archivo?
* **Respuesta de 10:** *"Porque seguimos el principio de **Responsabilidad Única**. Al modularizar el sistema, separamos la lógica de preprocesamiento de texto de los algoritmos matemáticos avanzados (coocurrencia, PPMI, SVD) y de la interfaz de consola del usuario. Esto hace que el código sea altamente legible, fácil de mantener, escalable y estructurado bajo buenas prácticas de ingeniería de software."*

---

### Pregunta 8: ¿Qué son las Stopwords y cómo maneja tu sistema su eliminación?
* **Respuesta de 10:** *"Las stopwords son palabras vacías o funcionales como artículos, preposiciones y conectores (p. ej. 'el', 'un', 'desde') que no contienen significado semántico propio. Nuestro sistema las carga desde el archivo externo `stopwords-es.txt`, las normaliza para asegurar que se emparejen correctamente sin importar acentos, y las elimina del flujo de tokens. Esto limpia el ruido gramatical y permite enfocar los recursos semánticos en palabras clave y sustantivos con contenido real."*
