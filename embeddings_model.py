import csv
import json
import numpy as np
from collections import Counter
import config
import utils
import text_processor

def crear_vocabulario(tokens):
    """Crea y ordena el vocabulario a partir de los tokens de forma descendente por frecuencia."""
    frecuencias = Counter(tokens)
    palabras_ord = sorted(frecuencias.items(), key=lambda x: -x[1])

    vocabulario = {}
    for indice, (palabra, freq) in enumerate(palabras_ord):
        vocabulario[palabra] = {"indice": indice, "frecuencia": freq}

    print("")
    print(f"  Palabras únicas en el vocabulario: {len(vocabulario)}")
    print("")
    print("  Las 15 palabras más frecuentes son:")
    print("  {:<8} {:<22} {:<10}".format("Índice", "Palabra", "Frecuencia"))
    print("  " + "-" * 42)
    for palabra, datos in list(vocabulario.items())[:15]:
        print("  {:<8} {:<22} {:<10}".format(
            datos["indice"], palabra, datos["frecuencia"]))

    return vocabulario

def generar_one_hot_muestra(vocabulario):
    """Genera y muestra una representación One-Hot de ejemplo para las palabras principales."""
    tam = len(vocabulario)
    palabras_muestra = list(vocabulario.keys())[:config.MUESTRA_OH]
    muestra = []

    print("")
    print(f"  One Hot Encoding - muestra de {config.MUESTRA_OH} palabras y sus vectores")
    print(f"  Tamaño de cada vector: {tam} dimensiones (tamaño del vocabulario)")
    print("  {:<22} {:<8} {}".format("Palabra", "Índice", "Primeros 8 valores"))
    print("  " + "-" * 60)

    for palabra in palabras_muestra:
        idx = vocabulario[palabra]["indice"]
        vector = [0] * tam
        vector[idx] = 1
        preview = str(vector[:8])
        print("  {:<22} {:<8} {}".format(palabra, idx, preview))
        muestra.append({"palabra": palabra, "indice": idx, "vector": vector})

    return muestra

def crear_parejas_ventana(tokens, ventana):
    """Crea parejas (palabra objetivo, palabra contexto) considerando la ventana de vecindad."""
    parejas = []
    for i, objetivo in enumerate(tokens):
        # max/min controlan dinámicamente las fronteras de palabras iniciales y finales
        inicio = max(0, i - ventana)
        fin    = min(len(tokens), i + ventana + 1)
        for j in range(inicio, fin):
            if i != j:
                parejas.append((objetivo, tokens[j]))

    print("")
    print(f"  Ventana de contexto usada : {ventana} palabras a cada lado")
    print(f"  TOTAL DE PAREJAS FORMADAS : {len(parejas)}")
    print("")
    print("  Muestra - primeras 15 parejas (objetivo - contexto):")
    print("  {:<5} {:<22} {:<22}".format("#", "Objetivo", "Contexto"))
    print("  " + "-" * 52)
    for i, (obj, ctx) in enumerate(parejas[:15], 1):
        print("  {:<5} {:<22} {:<22}".format(i, obj, ctx))
    if len(parejas) > 15:
        utils.info(f"... y {len(parejas) - 15} parejas más.")

    return parejas

def crear_matriz_coocurrencia(tokens, vocabulario, ventana):
    """Construye la matriz de coocurrencia acumulando apariciones locales."""
    tam = len(vocabulario)
    matriz = np.zeros((tam, tam), dtype=np.float32)

    for i, objetivo in enumerate(tokens):
        idx_obj = vocabulario[objetivo]["indice"]
        # max/min controlan las fronteras dinámicas al inicio y final del documento
        inicio  = max(0, i - ventana)
        fin     = min(len(tokens), i + ventana + 1)
        for j in range(inicio, fin):
            if i != j:
                idx_ctx = vocabulario[tokens[j]]["indice"]
                matriz[idx_obj][idx_ctx] += 1

    no_cero = int(np.count_nonzero(matriz))
    print("")
    print(f"  Dimensión de la matriz : {tam} x {tam}")
    print(f"  Celdas con valor > 0   : {no_cero}")
    print("  Densidad               : {:.2f}%".format(no_cero / (tam * tam) * 100))

    return matriz

def calcular_ppmi(matriz):
    """Calcula la matriz PPMI a partir de la matriz de coocurrencia."""
    total = matriz.sum()
    if total == 0:
        return matriz
    sf = matriz.sum(axis=1, keepdims=True)
    sc = matriz.sum(axis=0, keepdims=True)
    # Evitar divisiones por cero agregando un pequeño epsilon
    sf = np.where(sf == 0, 1e-10, sf)
    sc = np.where(sc == 0, 1e-10, sc)
    pmi  = np.log2((matriz * total) / (sf * sc) + 1e-10)
    ppmi = np.maximum(pmi, 0)
    return ppmi

def generar_embeddings(matriz_cooc, ventana):
    """Aplica PPMI y reducción SVD para producir embeddings densos de baja dimensión."""
    print("")
    print("  Método usado: PPMI + SVD")
    print("  Parámetros del Modelo:")
    print(f"    Ventana de contexto  : {ventana}")
    print(f"    Dimensiones SVD      : {config.DIMENSIONES}")
    print(f"    Frecuencia mínima    : {config.MIN_FRECUENCIA}")

    ppmi = calcular_ppmi(matriz_cooc)

    # Limitar la dimensión real si el vocabulario es más pequeño que las dimensiones estimadas
    dim_real = min(config.DIMENSIONES, ppmi.shape[0] - 1)
    print("")
    print("  Calculando Descomposición SVD...")
    U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
    embeddings = U[:, :dim_real] * S[:dim_real]

    print(f"  Embeddings generados   : {embeddings.shape[0]} palabras x {embeddings.shape[1]} dimensiones")
    return embeddings

def guardar_resultados(tokens, vocabulario, parejas, muestra_oh, embeddings, ventana):
    """Guarda todos los resultados y metadatos en archivos CSV y binarios en el disco."""
    config.CARPETA_MODELO.mkdir(exist_ok=True)

    # tokens_finales.csv
    with open(config.CARPETA_MODELO / "tokens_finales.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["posicion", "token"])
        for i, t in enumerate(tokens):
            w.writerow([i, t])

    # vocabulario.csv
    with open(config.CARPETA_MODELO / "vocabulario.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["indice", "palabra", "frecuencia"])
        for pal, d in vocabulario.items():
            w.writerow([d["indice"], pal, d["frecuencia"]])

    # parejas_ventana.csv
    with open(config.CARPETA_MODELO / "parejas_ventana.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["objetivo", "contexto"])
        for par in parejas:
            w.writerow(par)

    # one_hot_muestra.csv
    with open(config.CARPETA_MODELO / "one_hot_muestra.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        n_dims = len(muestra_oh[0]["vector"])
        w.writerow(["palabra", "indice"] + ["d" + str(i) for i in range(n_dims)])
        for e in muestra_oh:
            w.writerow([e["palabra"], e["indice"]] + e["vector"])

    # embeddings.csv
    palabras_ord = sorted(vocabulario.items(), key=lambda x: x[1]["indice"])
    with open(config.CARPETA_MODELO / "embeddings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["palabra"] + ["dim_" + str(i) for i in range(embeddings.shape[1])])
        for pal, d in palabras_ord:
            w.writerow([pal] + list(embeddings[d["indice"]]))

    # embeddings.npy (formato binario numpy para velocidad instantánea)
    np.save(config.CARPETA_MODELO / "embeddings.npy", embeddings)

    # configuracion_modelo.json (Metadatos de la corrida)
    config_dict = {
        "hiperparametros": {
            "metodo": "PPMI + SVD",
            "ventana_contexto": ventana,
            "dimensiones_svd": config.DIMENSIONES,
            "frecuencia_minima": config.MIN_FRECUENCIA,
            "archivo_stopwords": "stopwords-es.txt",
            "total_stopwords_cargadas": len(text_processor.STOPWORDS)
        },
        "estadisticas": {
            "total_tokens": len(tokens),
            "tamano_vocabulario": len(vocabulario),
            "total_parejas": len(parejas),
            "forma_embeddings": list(embeddings.shape)
        },
        "archivos_generados": [
            "tokens_finales.csv",
            "vocabulario.csv",
            "parejas_ventana.csv",
            "one_hot_muestra.csv",
            "embeddings.csv",
            "embeddings.npy",
            "configuracion_modelo.json"
        ]
    }
    with open(config.CARPETA_MODELO / "configuracion_modelo.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def cargar_modelo():
    """Carga los embeddings binarios y el vocabulario desde el disco."""
    ruta_emb   = config.CARPETA_MODELO / "embeddings.npy"
    ruta_vocab = config.CARPETA_MODELO / "vocabulario.csv"

    if not ruta_emb.exists() or not ruta_vocab.exists():
        return None, None

    embeddings = np.load(ruta_emb)
    vocabulario = {}
    with open(ruta_vocab, "r", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            vocabulario[fila["palabra"]] = {
                "indice":     int(fila["indice"]),
                "frecuencia": int(fila["frecuencia"])
            }

    utils.ok(f"Modelo cargado: {len(vocabulario)} palabras  |  embeddings {embeddings.shape}")
    return embeddings, vocabulario

def similitud_coseno(vec_a, vec_b):
    """Mide la distancia/similitud angular entre dos vectores continuos."""
    na = np.linalg.norm(vec_a)
    nb = np.linalg.norm(vec_b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (na * nb))

def buscar_similares(consulta, embeddings, vocabulario, top_n=8):
    """Busca las palabras más cercanas a la consulta empleando similitud coseno."""
    consulta_norm = text_processor.quitar_acentos(consulta.lower().strip())

    if consulta_norm not in vocabulario:
        utils.err(f"La palabra '{consulta_norm}' no está en el vocabulario entrenado.")
        # Sugerencias parciales básicas
        sugerencias = [p for p in vocabulario if consulta_norm[:4] in p][:5]
        if sugerencias:
            utils.info(f"Sugerencias similares en vocabulario: {sugerencias}")
        return

    idx = vocabulario[consulta_norm]["indice"]
    vec = embeddings[idx]

    resultados = []
    for palabra, datos in vocabulario.items():
        if palabra == consulta_norm:
            continue
        sim = similitud_coseno(vec, embeddings[datos["indice"]])
        resultados.append((palabra, sim))

    resultados.sort(key=lambda x: -x[1])

    print("")
    print(f"  Resultados de Búsqueda Semántica para: '{consulta_norm}'")
    print("  {:<5} {:<24} {:<10}".format("#", "Palabra", "Similitud Coseno"))
    print("  " + "-" * 48)
    for i, (pal, sim) in enumerate(resultados[:top_n], 1):
        print("  {:<5} {:<24} {:.4f}".format(i, pal, sim))
