from typing import Optional
import pyscreeze
from PIL import Image
from mss import mss
import screeninfo # Para obtener información detallada de los monitores
import os # Para verificar si los archivos existen
import traceback # Para depuración de errores
import sys # Para un manejo más limpio de la salida en la función main

def take_screenshot(monitor_number: Optional[int] = None) -> Optional[Image.Image]:
    """
    Toma una captura de pantalla.
    Args:
        monitor_number (int, optional): Índice del monitor a capturar (0 para el primer monitor físico, 1 para el segundo, etc.).
                                        Si es None, captura todos los monitores como una sola imagen (combinada por MSS).
    Returns:
        PIL.Image.Image: La imagen de la captura de pantalla, o None si falla.
    Raises:
        ValueError: Si el número de monitor es inválido.
    """
    try:
        with mss() as sct:
            # monitors[0] es la pantalla combinada de todos los monitores.
            # monitors[1], monitors[2], etc., son los monitores individuales físicos.
            # Por lo tanto, el monitor_number proporcionado por el usuario (0-indexed para monitores físicos)
            # debe mapearse a mss_monitor_index = monitor_number + 1.
            
            # Si solo hay un monitor físico (monitors[0] y monitors[1] disponibles), len(monitors) será 2.
            # Los IDs de monitor físico utilizables serán del 0 al len(monitors) - 2.
            num_physical_monitors = len(sct.monitors) - 1 # Excluye el monitor[0] que es el área combinada

            if monitor_number is not None:
                if not (0 <= monitor_number < num_physical_monitors):
                    print(f"ERROR: Número de monitor inválido: {monitor_number}. Monitores físicos disponibles: 0 a {num_physical_monitors - 1}.")
                    print(f"DEBUG: 'monitors' de mss contiene {len(sct.monitors)} elementos (monitors[0] es el área combinada de todos los monitores físicos).")
                    raise ValueError(
                        f"Número de monitor inválido: {monitor_number}. Monitores físicos disponibles: 0 a {num_physical_monitors - 1}."
                        " (0 es el primer monitor físico, 1 el segundo, etc.)"
                    )
                # Mapear el índice proporcionado (0-indexed físico) al índice de mss (1-indexed físico)
                mss_monitor_index = monitor_number + 1
                monitor_region = sct.monitors[mss_monitor_index]
                print(f"DEBUG: Capturando el monitor {monitor_number} (región: {monitor_region})...")
            else:
                # Capturar todos los monitores (monitors[0] es la región de todos los monitores combinados)
                monitor_region = sct.monitors[0]
                print("DEBUG: Capturando todos los monitores (área combinada)...")

            sct_img = sct.grab(monitor_region)
            # Convertir la imagen de mss a un objeto PIL.Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            print("DEBUG: Captura de pantalla realizada con éxito.")
            return img
    except ValueError as ve:
        print(f"ERROR al tomar captura de pantalla: {ve}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR inesperado al tomar captura de pantalla: {e}", file=sys.stderr)
        traceback.print_exc() # Imprime el stack trace para depuración
        return None

def find_image_on_screen(template_image_path: str, screenshot_image: Image.Image, confidence: float = 0.9):
    """
    Busca una imagen (template) dentro de una imagen de captura de pantalla.
    Args:
        template_image_path (str): Ruta al archivo de imagen del recorte (template a buscar).
        screenshot_image (PIL.Image.Image): El objeto PIL.Image de la captura de pantalla completa o del monitor.
        confidence (float): Nivel de confianza para la detección de la imagen (0.0 a 1.0).
    Returns:
        tuple: Coordenadas (left, top, width, height) de la imagen encontrada, o None si no se encuentra.
               Es el mismo formato de 'Box' que usa PyAutoGUI.
    """
    try:
        if not os.path.exists(template_image_path):
            print(f"ERROR: El archivo de imagen template '{template_image_path}' no existe.", file=sys.stderr)
            return None

        print(f"DEBUG: Buscando '{os.path.basename(template_image_path)}' en la captura con confianza {confidence}...")
        
        # pyscreeze.locate puede tomar un objeto PIL.Image como 'haystackImage'
        # El primer argumento es siempre la ruta del template.
        location = pyscreeze.locate(template_image_path, screenshot_image, confidence=confidence)
        
        if location:
            print(f"DEBUG: Recorte '{os.path.basename(template_image_path)}' encontrado en la pantalla en: {location}")
            return location # Retorna un objeto Box de PyAutoGUI
        else:
            print(f"DEBUG: Recorte '{os.path.basename(template_image_path)}' NO encontrado en la pantalla con confianza {confidence}.")
            return None
    except pyscreeze.PyScreezeException as e:
        print(f"ERROR al buscar imagen '{template_image_path}' con pyscreeze: {e}.", file=sys.stderr)
        print("Asegúrate de que los archivos de imagen existan y sean válidos.", file=sys.stderr)
        traceback.print_exc() # Imprime el stack trace para depuración
        return None
    except Exception as e:
        print(f"ERROR inesperado en find_image_on_screen: {e}", file=sys.stderr)
        traceback.print_exc() # Imprime el stack trace para depuración
        return None

def get_monitor_info() -> list:
    """
    Obtiene información sobre los monitores conectados.
    Returns:
        list: Una lista de diccionarios, cada uno con información de un monitor.
              El índice 'id' corresponde al primer monitor físico (0, 1, etc.).
    """
    try:
        monitors_info = []
        monitors = screeninfo.get_monitors()
        for i, m in enumerate(monitors):
            # Asignamos un ID incremental a partir de 0 para los monitores físicos
            monitors_info.append({
                "id": i,
                "width": m.width,
                "height": m.height,
                "x": m.x,
                "y": m.y,
                "is_primary": m.is_primary,
                "name": m.name if hasattr(m, 'name') else f"Monitor {i}"
            })
        print("DEBUG: Información de monitores obtenida.")
        return monitors_info
    except Exception as e:
        print(f"ERROR al obtener información de los monitores: {e}", file=sys.stderr)
        traceback.print_exc() # Imprime el stack trace para depuración
        return []

# Ejemplo de uso (para pruebas directas de este módulo)
if __name__ == "__main__":
    print("--- Probando utilidades de pantalla (utils/screen_utils.py) ---")

    # Mostrar información de monitores
    mon_info = get_monitor_info()
    if mon_info:
        print("\nMonitores detectados:")
        for mon in mon_info:
            print(f"  ID: {mon['id']}, Nombre: {mon['name']}, Resolución: {mon['width']}x{mon['height']}, Posición: ({mon['x']},{mon['y']}), Primario: {mon['is_primary']}")
    else:
        print("No se pudo detectar información de monitores.")

    # Prueba de captura del primer monitor físico (si existe)
    if mon_info:
        try:
            print("\nIntentando capturar el primer monitor físico (ID 0)...")
            img_monitor_0 = take_screenshot(monitor_number=0)
            if img_monitor_0:
                img_monitor_0.save("test_screenshot_monitor_0.png")
                print("Captura del monitor 0 guardada como 'test_screenshot_monitor_0.png'")
            else:
                print("No se pudo obtener la captura del monitor 0.")
        except ValueError as e:
            print(f"Error específico de valor al capturar monitor 0: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error inesperado al capturar monitor 0: {e}", file=sys.stderr)

    # Prueba de captura de pantalla completa
    try:
        print("\nIntentando capturar todos los monitores (combinados)...")
        full_screenshot = take_screenshot()
        if full_screenshot:
            full_screenshot.save("test_full_screenshot.png")
            print("Captura completa guardada como 'test_full_screenshot.png'")
        else:
            print("No se pudo obtener la captura de pantalla completa.")
    except Exception as e:
        print(f"Error al capturar pantalla completa: {e}", file=sys.stderr)

    # Para probar find_image_on_screen, necesitarías un recorte de prueba
    # y que la captura de pantalla completa exista.
    # Usaremos los archivos de ejemplo que me has proporcionado previamente:
    # "image_01a69f.png" y "image_0204d7.png"
    
    # Intenta buscar "image_01a69f.png" dentro de "image_0204d7.png"
    # Esto es solo un ejemplo de cómo se usaría la función, no necesariamente
    # que una imagen contenga a la otra en un escenario real de pantalla.
    
    # Se simula la existencia de los archivos temporales para la prueba.
    # En un caso real, 'test_full_screenshot.png' sería el 'screenshot_image'
    # y 'image_01a69f.png' sería el 'template_image_path'.

    print("\nIntentando simular búsqueda de 'image_01a69f.png' dentro de 'image_0204d7.png'...")

    # Creamos un archivo temporal para simular 'image_01a69f.png'
    temp_template_path = "temp_image_01a69f.png"
    # Guardamos el contenido de "image_01a69f.png" en el archivo temporal
    # Esto asume que el contenido de la imagen ha sido previamente cargado o que el archivo existe
    # Para este ejemplo, lo generaremos si no existe.
    if not os.path.exists(temp_template_path):
        # En un escenario real, este archivo provendría de tu carpeta 'clippings'
        # Aquí, simplemente lo creamos para que la prueba pueda ejecutar.
        # Puedes reemplazar esto con una copia de tu archivo real si es necesario.
        try:
            # Si tienes el ID de contenido de la imagen que me pasaste antes
            # puedes intentar obtenerla, pero para una prueba local esto no es directo.
            # Simplemente crearemos un archivo dummy o te pediría que lo crearas.
            # Suponiendo que tienes un archivo real llamado 'image_01a69f.png' en la misma carpeta para probar
            # Si no lo tienes, esta parte puede fallar.
            # Para la prueba: puedes copiar una imagen pequeña existente en tu sistema a este nombre.
            # Por ejemplo: shutil.copy('path_to_some_image.png', temp_template_path)
            with open(temp_template_path, 'w') as f:
                f.write("dummy content for image_01a69f.png") # Esto no es una imagen real, solo para que os.path.exists funcione
            print(f"NOTA: Creado un archivo dummy '{temp_template_path}'. Para una prueba real, reemplázalo con una imagen PNG válida.")
        except Exception as ex:
            print(f"No se pudo crear el archivo dummy {temp_template_path}: {ex}", file=sys.stderr)

    # Creamos un archivo temporal para simular 'image_0204d7.png' (la "captura de pantalla")
    temp_screenshot_path = "temp_image_0204d7.png"
    if not os.path.exists(temp_screenshot_path):
        try:
            with open(temp_screenshot_path, 'w') as f:
                f.write("dummy content for image_0204d7.png")
            print(f"NOTA: Creado un archivo dummy '{temp_screenshot_path}'. Para una prueba real, reemplázalo con una imagen PNG válida.")
        except Exception as ex:
            print(f"No se pudo crear el archivo dummy {temp_screenshot_path}: {ex}", file=sys.stderr)


    if os.path.exists(temp_template_path) and os.path.exists(temp_screenshot_path):
        try:
            # Es crucial que estas sean imágenes PIL reales para que pyscreeze funcione.
            # Si estás ejecutando esto localmente, asegúrate de que 'temp_image_01a69f.png'
            # y 'temp_image_0204d7.png' sean archivos PNG válidos.
            # Para una prueba real, idealmente:
            # screenshot_to_search = Image.open("test_full_screenshot.png")
            # template_to_find = "path/to/your/clippings/my_button.png"

            # Para que el ejemplo sea ejecutable sin requerir las imágenes de tu sistema de archivos,
            # voy a usar una imagen de ejemplo muy simple.
            # En un entorno real, `full_screenshot` de la prueba anterior sería el `screenshot_image`
            # y `clipping_path` de tu carpeta de recortes sería `template_image_path`.
            
            # Recreamos 'full_screenshot' si no existe o si se borró.
            if 'full_screenshot' not in locals() or full_screenshot is None:
                 full_screenshot = take_screenshot()
                 if full_screenshot:
                     full_screenshot.save("test_full_screenshot_for_search.png")
                 else:
                     print("No se pudo obtener una captura de pantalla para la prueba de búsqueda de imagen.", file=sys.stderr)
                     full_screenshot = None # Asegurar que es None si falla

            if full_screenshot:
                # Ahora usamos un clipping real que se podría haber generado o que el usuario tenga.
                # Si el usuario tiene 'image_01a69f.png' en su directorio, lo usaremos.
                actual_clipping_path = "image_01a69f.png"
                if os.path.exists(actual_clipping_path):
                    print(f"\nBuscando '{actual_clipping_path}' en 'test_full_screenshot_for_search.png'...")
                    found_loc = find_image_on_screen(actual_clipping_path, full_screenshot)
                    if found_loc:
                        print(f"Encontrado en: {found_loc}")
                    else:
                        print("No encontrado.")
                else:
                    print(f"\nSkipping search test: '{actual_clipping_path}' not found. Please ensure your clipping files are in the working directory.")
            else:
                print("Skipping image search test because no full screenshot could be taken.", file=sys.stderr)


        except Exception as e:
            print(f"Error al cargar imágenes para la prueba de búsqueda: {e}", file=sys.stderr)
            traceback.print_exc()
    else:
        print("\nPara probar find_image_on_screen con tus archivos, asegúrate de que 'image_01a69f.png' y 'image_0204d7.png' existan en el directorio de trabajo.")

    # Limpiar archivos temporales de dummy si se crearon
    for dummy_file in [temp_template_path, temp_screenshot_path, "test_full_screenshot_for_search.png"]:
        if os.path.exists(dummy_file) and (dummy_file.startswith("temp_") or "for_search" in dummy_file):
            try:
                os.remove(dummy_file)
                print(f"Limpiado archivo dummy/temporal: {dummy_file}")
            except Exception as e:
                print(f"Error al limpiar archivo dummy/temporal {dummy_file}: {e}", file=sys.stderr)