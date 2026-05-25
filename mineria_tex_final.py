import json
import config
import utils
import text_processor
import embeddings_model

# Mantener registro de la ventana de contexto seleccionada
ventana_actual = config.VENTANA

def pedir_ventana():
    """Solicita dinámicamente el tamaño de la ventana de contexto al usuario."""
    valor = input("  Ventana de contexto [2]: ").strip()

    if valor == "":
        return 2

    try:
        ventana = int(valor)
        if ventana < 1:
            utils.err("La ventana debe ser mayor que cero. Se usará 2.")
            return 2
        return ventana
    except ValueError:
        utils.err("Valor no válido. Se usará 2.")
        return 2

def mostrar_tokens(tokens):
    """Muestra una vista previa estructurada de los tokens resultantes de la limpieza."""
    print("")
    print(f"  TOTAL DE TOKENS EN EL TEXTO: {len(tokens)}")
    print("")
    print("  Primeros 25 tokens con su posición:")
    print("  {:<6} {:<22}".format("Pos", "Token"))
    print("  " + "-" * 60)
    for i, token in enumerate(tokens[:25]):
        print("  {:<6} {:<22}".format(i, token))
    if len(tokens) > 25:
        utils.info(f"... y {len(tokens) - 25} tokens más.")

def mostrar_resumen():
    """Muestra un resumen descriptivo del modelo entrenado y sus hiperparámetros."""
    ruta = config.CARPETA_MODELO / "configuracion_modelo.json"
    if not ruta.exists():
        utils.err("No se encontró ningún modelo guardado en disco. Ejecute la opción 1 primero.")
        return

    with open(ruta, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    h = config_data["hiperparametros"]
    e = config_data["estadisticas"]

    utils.titulo("RESUMEN DEL MODELO ENTRENADO")
    print("  HIPERPARÁMETROS:")
    print(f"    Método              : {h['metodo']}")
    print(f"    Ventana de contexto : {h['ventana_contexto']}  palabras a cada lado")
    print(f"    Dimensiones Vector  : {h['dimensiones_svd']}")
    if "learning_rate" in h:
        print(f"    Tasa de aprendizaje : {h['learning_rate']}")
    if "epochs" in h:
        print(f"    Épocas de entrena.  : {h['epochs']}")
    print(f"    Frecuencia mínima   : {h['frecuencia_minima']}")
    if "archivo_stopwords" in h:
        print(f"    Stopwords usadas    : {h['archivo_stopwords']}")
    if "total_stopwords_cargadas" in h:
        print(f"    Total stopwords     : {h['total_stopwords_cargadas']}")
    utils.linea("-", 60)
    print("  ESTADÍSTICAS:")
    print(f"    Total de tokens     : {e['total_tokens']}")
    print(f"    Vocabulario         : {e['tamano_vocabulario']} palabras únicas")
    print(f"    Parejas generadas   : {e['total_parejas']}")
    print(f"    Forma embeddings    : {e['forma_embeddings'][0]} palabras x {e['forma_embeddings'][1]} dims")

def procesar_y_entrenar():
    """Coordina el flujo de preprocesamiento, tokenización, coocurrencia y generación de embeddings."""
    global ventana_actual
    utils.titulo("PROCESAMIENTO")

    print("  Ingresa la ruta del archivo .txt.")
    ruta = input("\n  Ruta del archivo: ")
    ventana_actual = pedir_ventana()
    utils.info(f"Ventana seleccionada: {ventana_actual}")

    utils.titulo("LECTURA DEL ARCHIVO")
    texto = text_processor.leer_archivo(ruta)
    if texto is None:
        return

    utils.titulo("NORMALIZACIÓN Y LIMPIEZA")
    tokens = text_processor.normalizar_texto(texto)
    if len(tokens) < 10:
        utils.err("Muy pocos tokens útiles en el texto. Verifique su contenido.")
        return
    utils.ok(f"Normalización completada. Tokens útiles obtenidos: {len(tokens)}")

    utils.titulo("TOKENS FINALES")
    mostrar_tokens(tokens)

    utils.titulo("VOCABULARIO")
    vocabulario = embeddings_model.crear_vocabulario(tokens)

    utils.titulo("ONE HOT ENCODING")
    muestra_oh = embeddings_model.generar_one_hot_muestra(vocabulario)

    utils.titulo("PAREJAS DE CONTEXTO")
    parejas = embeddings_model.crear_parejas_ventana(tokens, ventana_actual)

    utils.titulo("MATRIZ DE COOCURRENCIA")
    matriz = embeddings_model.crear_matriz_coocurrencia(tokens, vocabulario, ventana_actual)

    utils.titulo("EMBEDDINGS")
    embeddings = embeddings_model.generar_embeddings(vocabulario, parejas, ventana_actual)

    utils.titulo("GUARDAR RESULTADOS EN DISCO")
    embeddings_model.guardar_resultados(tokens, vocabulario, parejas, muestra_oh, embeddings, ventana_actual)

    print("")
    print("  *** ENTRENAMIENTO COMPLETADO CON ÉXITO ***")

def busqueda_interactiva():
    """Carga de forma instantánea el modelo pre-entrenado y permite buscar términos similares."""
    utils.titulo("BÚSQUEDA SEMÁNTICA (INFERENCIA)")

    embeddings, vocabulario = embeddings_model.cargar_modelo()
    if embeddings is None:
        print("")
        utils.err(f"No se encontró modelo guardado en la ruta: {config.CARPETA_MODELO}")
        utils.err("Ejecute la opción 1 primero para procesar un archivo de texto.")
        return

    print("")
    print("  Escriba una palabra para encontrar términos semánticamente similares en el corpus.")
    print("  Escriba 'salir' para volver al menú de opciones.")

    while True:
        print("")
        consulta = input("  Palabra a buscar: ").strip()
        if consulta.lower() == "salir":
            break
        if not consulta:
            continue
        embeddings_model.buscar_similares(consulta, embeddings, vocabulario)

def menu():
    """Lazo de interacción principal del sistema mediante consola CLI."""
    while True:
        utils.linea("-", 60)
        print("  SISTEMA DE MINERÍA DE TEXTO Y EMBEDDINGS (VERSIÓN MODULAR)")
        print("  1. Procesar archivo .txt y entrenar modelo")
        print("  2. Cargar modelo y hacer búsqueda semántica")
        print("  3. Ver resumen del modelo")
        print("  4. Salir")
        utils.linea("-", 60)

        opcion = input("  Elige una opción (1-4): ").strip()

        if   opcion == "1":  procesar_y_entrenar()
        elif opcion == "2":  busqueda_interactiva()
        elif opcion == "3":  mostrar_resumen()
        elif opcion == "4":
            print("")
            print("  ¡Éxito en su presentación escolar! Hasta luego.")
            print("")
            break
        else:
            utils.err("Opción no válida. Escriba un número del 1 al 4.")

if __name__ == "__main__":
    menu()
