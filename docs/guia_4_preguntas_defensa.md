# Guía de Estudio 4: Balotario de Preguntas Frecuentes de la Defensa

Esta guía recopila las preguntas más probables y difíciles que los profesores suelen hacer durante la presentación y defensa de este proyecto de minería de texto. Estúdialas a fondo para asegurar la máxima calificación.

---

### Pregunta 1: ¿Por qué el sistema de búsqueda interactiva funciona de forma instantánea y no requiere que carguemos el archivo de texto largo el día de la presentación?
* **Respuesta de 10:** *"Porque implementamos una arquitectura de **Carga Diferida** para separar la fase de entrenamiento de la de inferencia. Al entrenar el modelo (Opción 1), comprimimos toda la semántica del corpus y la guardamos en disco en archivos estructurados (`vocabulario.csv` y los vectores matemáticos binarios en `embeddings.npy`). El día de la presentación, la Opción 2 lee directamente estos archivos en milisegundos sin repetir todo el pipeline de limpieza y SVD, garantizando una demostración rápida y fluida."*

---

### Pregunta 2: ¿Cómo maneja tu código las palabras en los extremos del texto (palabras iniciales y finales) para que la ventana de contexto no desborde la lista?
* **Respuesta de 10:** *"El código implementa **condiciones de frontera lógicas dinámicas** mediante las funciones `max` y `min` de Python en el barrido de la ventana: `inicio = max(0, i - ventana)` y `fin = min(len(tokens), i + ventana + 1)`. Esto garantiza que si una palabra está al principio (índice 0) o al final del texto, la ventana se detenga exactamente en los límites válidos del documento, evitando accesos a índices inexistentes (que romperían el script) y controlando la variación del contexto correctamente."*

---

### Pregunta 3: ¿Para qué sirve aplicar PPMI (Positive Pointwise Mutual Information) sobre la matriz de coocurrencia? ¿Por qué no usar solo los conteos directos?
* **Respuesta de 10:** *"Los conteos directos de coocurrencia tienen un grave sesgo: palabras muy comunes por naturaleza en el idioma tienden a aparecer juntas simplemente por casualidad, inflando falsamente su relación. La fórmula de **PMI** resuelve esto comparando la probabilidad conjunta de que dos palabras aparezcan juntas frente a sus probabilidades de aparecer por separado. Además, usamos **PPMI** (reemplazando valores negativos e infinitos por cero) porque las relaciones menores al azar no nos interesan y el logaritmo de cero daría un error matemático indeterminado en el software."*

---

### Pregunta 4: ¿Qué es el SVD y por qué reduces las dimensiones a 50 en lugar de usar todo el vocabulario?
* **Respuesta de 10:** *"La Descomposición en Valores Singulares (SVD) es una técnica matemática de álgebra lineal para **reducción de dimensionalidad**. La matriz PPMI original es gigante ($V \times V$) y extremadamente dispersa (llena de ceros), lo que hace que procesar búsquedas sea ineficiente y ruidoso. SVD comprime esa información y extrae las **50 componentes más significativas** (que concentran la mayor cantidad de varianza y patrones semánticos latentes), transformando la matriz gigante en vectores pequeños y densos de 50 dimensiones por palabra."*

---

### Pregunta 5: ¿Por qué elegiste usar "Similitud Coseno" para la búsqueda semántica en lugar de la distancia clásica (Euclidiana)?
* **Respuesta de 10:** *"En lingüística computacional, la distancia euclidiana (medir en línea recta) es engañosa porque es sensible a la longitud de los vectores. Si una palabra aparece muchísimas más veces en el texto que otra, su vector será enorme en longitud, dando una distancia euclidiana gigante aunque tengan significados idénticos. La **Similitud Coseno** soluciona esto porque **mide únicamente el ángulo entre los vectores**, ignorando su longitud. Así, se evalúa únicamente la dirección conceptual de las palabras."*

---

### Pregunta 6: ¿Por qué fue una buena práctica de ingeniería de software separar el código en módulos (`text_processor.py`, `embeddings_model.py`, etc.) en lugar de tener todo en un solo archivo?
* **Respuesta de 10:** *"Porque seguimos el principio de **Responsabilidad Única**. Al modularizar el sistema, separamos la lógica de preprocesamiento de texto de los algoritmos matemáticos avanzados (coocurrencia, PPMI, SVD) y de la interfaz de consola del usuario. Esto hace que el código sea altamente legible, fácil de mantener, escalable y estructurado bajo buenas prácticas de ingeniería de software."*

---

### Pregunta 7: ¿Qué son las Stopwords y cómo maneja tu sistema su eliminación?
* **Respuesta de 10:** *"Las stopwords son palabras vacías o funcionales como artículos, preposiciones y conectores (p. ej. 'el', 'un', 'desde') que no contienen significado semántico propio. Nuestro sistema las carga desde el archivo externo `stopwords-es.txt`, las normaliza para asegurar que se emparejen correctamente sin importar acentos, y las elimina del flujo de tokens. Esto limpia el ruido gramatical y permite enfocar los recursos semánticos en palabras clave y sustantivos con contenido real."*
