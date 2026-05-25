def linea(caracter="=", largo=60):
    """Imprime una línea divisoria decorativa en la consola."""
    print("")
    print(caracter * largo)

def titulo(texto):
    """Imprime un encabezado con formato estético para separar secciones."""
    linea()
    print(f"  {texto}")
    linea("-", 60)

def ok(msg):
    """Imprime un mensaje de éxito o confirmación."""
    print(f"  [✓] {msg}")

def err(msg):
    """Imprime un aviso o mensaje de error."""
    print(f"  [!] Aviso: {msg}")

def info(msg):
    """Imprime un mensaje informativo estándar."""
    print(f"  {msg}")
