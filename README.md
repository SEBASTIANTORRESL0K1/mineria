# Sistema de Minería de Texto y Embeddings Semánticos (Modular)

Este repositorio contiene un sistema interactivo en Python para la limpieza de texto no estructurado, la modelación de relaciones semánticas de palabras mediante matrices de coocurrencia locales, el cálculo estadístico de **PPMI** (Positive Pointwise Mutual Information), la reducción de dimensiones con **SVD** (Singular Value Decomposition) y la búsqueda semántica en tiempo real usando **Similitud Coseno**.

---

## 🛠️ Requisitos e Instalación

1. **Python 3.x** instalado.
2. Instalar la dependencia externa necesaria:
   ```bash
   pip install numpy
   ```

---

## 🚀 Cómo Ejecutar el Sistema

En tu terminal (PowerShell o CMD), navega a la carpeta del proyecto y ejecuta el archivo principal:
```bash
python mineria_tex_final.py
```

El programa te presentará un menú interactivo en consola con **4 opciones**:
1. **Procesar archivo .txt y entrenar modelo:** Limpia un texto plano, crea el vocabulario, genera parejas de contexto, calcula la matriz de coocurrencia, aplica PPMI + SVD y guarda los resultados.
2. **Cargar modelo y hacer búsqueda semántica:** Carga de manera instantánea el modelo guardado previamente (embeddings y vocabulario) y permite buscar sinónimos o palabras relacionadas conceptualmente sin reentrenar.
3. **Ver resumen del modelo:** Muestra las estadísticas generales del último entrenamiento guardado.
4. **Salir:** Cierra el sistema.

---

## 📂 Arquitectura Modular

El proyecto está diseñado bajo una arquitectura limpia y modularizada para facilitar su legibilidad:
* **`config.py`**: Parámetros globales y rutas del sistema.
* **`utils.py`**: Funciones auxiliares estéticas para impresiones limpias en consola.
* **`text_processor.py`**: Limpieza lingüística, normalización Unicode y filtrado de stopwords.
* **`embeddings_model.py`**: Lógica matemática central (Coocurrencia, PPMI, SVD, Similitud Coseno y persistencia en disco).
* **`mineria_tex_final.py`**: Interfaz de menú CLI y orquestador del sistema.

---

## 🎓 Índice de la Guía de Estudio para la Defensa del Proyecto

Para ayudarte a comprender al 100% el funcionamiento matemático, el código y prepararte para las preguntas del docente durante la presentación, hemos creado una guía de estudio dividida en módulos accesibles:

### 📖 Guías Temáticas de Estudio
1. [**Guía 1: Preprocesamiento y Limpieza Lingüística**](docs/guia_1_preprocesamiento.md)
   * *Aprende cómo el código limpia el texto, elimina acentos y filtra stopwords sin dañar los datos.*
2. [**Guía 2: Coocurrencia, PPMI y SVD (Matemáticas del Modelo)**](docs/guia_2_coocurrencia_ppmi.md)
   * *Entiende paso a paso la ventana de contexto, cómo se construye la matriz y por qué se usan PPMI y SVD.*
3. [**Guía 3: Embeddings Semánticos y Similitud Coseno**](docs/guia_3_embeddings_similitud.md)
   * *Descubre qué es un embedding, cómo se almacena en disco de forma eficiente y cómo calcula la similitud entre palabras.*
4. [**Guía 4: Balotario de Preguntas Frecuentes de la Defensa**](docs/guia_4_preguntas_defensa.md)
   * *Una recopilación de las preguntas más difíciles y comunes que hacen los profesores durante la evaluación, con sus respuestas exactas.*
