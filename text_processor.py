import re
import unicodedata
from pathlib import Path
from collections import Counter
import config
import utils

# Conjunto de respaldo de palabras vacías si no se encuentra el archivo externo
STOPWORDS_RESPALDO = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "aunque",
    "cada", "como", "con", "cuando", "de", "del", "desde", "donde",
    "durante", "el", "ella", "ellas", "ellos", "en", "entre", "era", "es",
    "esa", "ese", "eso", "esta", "este", "estos", "fue", "han", "hasta",
    "hay", "la", "las", "le", "les", "lo", "los", "mas", "me", "mi",
    "mis", "mucho", "muy", "no", "nos", "o", "otra", "otro", "para",
    "pero", "por", "porque", "que", "quien", "se", "segun", "ser", "si",
    "sin", "sobre", "son", "su", "sus", "tambien", "te", "tiene", "todo",
    "un", "una", "unos", "y", "ya", "yo"
}

def preparar_stopword(palabra):
    """Normaliza una stopword individual (limpieza, minúsculas, remover acentos)."""
    palabra = palabra.lower().strip()
    palabra = unicodedata.normalize("NFD", palabra)
    palabra = "".join(c for c in palabra if unicodedata.category(c) != "Mn")
    palabra = re.sub(r"[^a-z\s]", " ", palabra)
    palabra = re.sub(r"\s+", " ", palabra).strip()
    return palabra

def cargar_stopwords():
    """
    Carga las stopwords en español desde el archivo de configuración.
    Normaliza los caracteres para garantizar concordancia con el texto limpio.
    """
    rutas_posibles = [
        config.ARCHIVO_STOPWORDS,
        Path("stopwords-es.txt"),
        Path("/mnt/data/stopwords-es.txt")
    ]

    ruta_encontrada = None
    for ruta in rutas_posibles:
        if ruta.exists():
            ruta_encontrada = ruta
            break

    if ruta_encontrada is None:
        utils.err("No se encontró el archivo 'stopwords-es.txt'. Se usará lista de respaldo.")
        return STOPWORDS_RESPALDO

    stopwords = set()
    with open(ruta_encontrada, "r", encoding="utf-8") as f:
        for linea_sw in f:
            linea_sw = linea_sw.strip()
            if not linea_sw or linea_sw.startswith("#"):
                continue

            limpia = preparar_stopword(linea_sw)
            if not limpia:
                continue

            stopwords.add(limpia)
            # Descomponer palabras de frases compuestas (ej. "por qué" -> "por", "que")
            for parte in limpia.split():
                if parte:
                    stopwords.add(parte)

    return stopwords

# Cargar el set global de stopwords
STOPWORDS = cargar_stopwords()

def quitar_acentos(texto):
    """Elimina acentos y caracteres diacríticos convirtiéndolos a ASCII equivalente."""
    nfkd = unicodedata.normalize("NFD", texto)
    return nfkd.encode("ascii", "ignore").decode("utf-8")

def normalizar_texto(texto):
    """
    Realiza la normalización y limpieza completa del texto.
    Elimina stopwords, puntuación, acentos, y filtra por frecuencia mínima.
    """
    texto = texto.lower()
    texto = quitar_acentos(texto)
    texto = re.sub(r"[^a-z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    tokens_crudos = texto.split()
    # Descartar stopwords y tokens de longitud igual o inferior a 1 carácter
    tokens_limpios = [t for t in tokens_crudos if t not in STOPWORDS and len(t) > 1]
    
    # Filtrar tokens por frecuencia mínima en base al vocabulario
    frecuencias = Counter(tokens_limpios)
    tokens_finales = [t for t in tokens_limpios if frecuencias[t] >= config.MIN_FRECUENCIA]
    return tokens_finales

def leer_archivo(ruta):
    """Lee y carga el contenido de un archivo de texto en formato UTF-8."""
    ruta = Path(ruta.strip().strip('"').strip("'"))
    if not ruta.exists():
        utils.err(f"No se encontró el archivo en la ruta especificada: {ruta}")
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        contenido = f.read()
    
    caracteres = len(contenido)
    palabras = len(contenido.split())
    utils.ok(f"Archivo leído exitosamente: {ruta.name}")
    utils.info(f"Caracteres totales: {caracteres}  |  Palabras aprox: {palabras}")
    
    if palabras < 400:
        print("")
        utils.err("El texto ingresado es demasiado corto para formar un modelo semántico fiable.")
        utils.err("Se recomienda utilizar un texto más extenso para mejores resultados.")
    return contenido
