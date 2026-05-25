from pathlib import Path

# Hiperparámetros por defecto del sistema
VENTANA = 2
DIMENSIONES = 50
MIN_FRECUENCIA = 1
MUESTRA_OH = 5
LEARNING_RATE = 0.05
EPOCHS = 500

# Rutas globales de archivos y directorios
ARCHIVO_STOPWORDS = Path(__file__).parent / "stopwords-es.txt"
CARPETA_MODELO = Path(__file__).parent / "resultados_modelo"
