import os
import uuid
from PIL import Image
import pyautogui # Para hacer clic en las coordenadas encontradas
import time # Para pausas entre acciones
import traceback # Para depuración de errores
from typing import Optional # Importar Optional para las anotaciones de tipo

# Importar las clases y funciones necesarias de nuestros módulos
from qdrant_handler import QdrantHandler
from image_processor import ImageProcessor
# take_screenshot ahora devuelve un objeto PIL.Image
# find_image_on_screen debe poder tomar un objeto PIL.Image para la 'screenshot_image'
from utils.screen_utils import take_screenshot, find_image_on_screen, get_monitor_info 

CLIPPINGS_DIR = "clippings"
os.makedirs(CLIPPINGS_DIR, exist_ok=True) # Asegurarse de que la carpeta exista

# Inicializar los handlers globalmente (o pasarlos como argumentos si se prefiere)
qdrant_handler = QdrantHandler()
image_processor = ImageProcessor()

def process_and_store_clipping(clipping_image_path: str):
    """
    Procesa un recorte de imagen: obtiene descripción, palabras clave, tipo,
    texto OCR y texto visible por IA, genera embedding,
    y lo almacena en Qdrant, moviendo el archivo a la carpeta 'clippings'.
    Args:
        clipping_image_path (str): Ruta al archivo de imagen del recorte original.
    Returns:
        str: El ID único del recorte en Qdrant (o None si falla).
    """
    print(f"\n--- Procesando y almacenando recorte: {clipping_image_path} ---")
    
    if not os.path.exists(clipping_image_path):
        print(f"Error: El archivo de recorte '{clipping_image_path}' no existe.")
        return None

    try:
        # 1. Obtener descripción, palabras clave, tipo de elemento, texto OCR y texto visible por IA
        print(f"DEBUG: Generando descripción para '{clipping_image_path}' con IA y OCR...")
        # CAMBIO CLAVE 1: describe_image_with_ai ahora devuelve ocr_text y ai_extracted_text
        description, keywords, element_type, ocr_text, ai_extracted_text = image_processor.describe_image_with_ai(clipping_image_path)
        
        if not description:
            print("Error: No se pudo generar una descripción con IA. Abortando.")
            return None
        print(f"DEBUG: Descripción IA: '{description}' | Texto IA: '{ai_extracted_text}' | Texto OCR: '{ocr_text}' | Palabras clave: {keywords} | Tipo: {element_type}")

        # 2. Generar embedding de la descripción (priorizando el texto OCR o IA si son más descriptivos)
        print("DEBUG: Generando embedding de la descripción (combinando IA y OCR si es necesario)...")
        
        # Estrategia para combinar textos para el embedding:
        # Priorizamos el texto visible de la IA si es significativo.
        # Si el OCR detecta texto y el texto de la IA es "Ninguno" o menos completo, lo incorporamos.
        text_for_embedding = description # La descripción base de la IA
        
        # Si la IA no extrajo texto visible pero el OCR sí, o si el OCR es más extenso
        if (ai_extracted_text.lower() == "ninguno" or len(ocr_text) > len(ai_extracted_text)) and \
           ocr_text and "ERROR" not in ocr_text: # Asegurarse de que OCR no devolvió un error
            text_for_embedding = f"{description}. Texto detectado: {ocr_text}"
        elif ai_extracted_text.lower() != "ninguno" and ai_extracted_text not in description:
            # Si la IA extrajo texto visible y no está ya en la descripción, lo añadimos
            text_for_embedding = f"{description}. Texto visible: {ai_extracted_text}"
        
        # Si ambos tienen texto y no están ya en la descripción, combinarlos
        if ai_extracted_text.lower() != "ninguno" and ocr_text and "ERROR" not in ocr_text:
            if ai_extracted_text not in text_for_embedding and ocr_text not in text_for_embedding:
                text_for_embedding = f"{description}. Texto visible: {ai_extracted_text}. OCR: {ocr_text}"
            elif ocr_text not in text_for_embedding:
                 text_for_embedding = f"{text_for_embedding}. OCR: {ocr_text}"


        embedding = image_processor.generate_embedding_from_text(text_for_embedding)
        
        if len(embedding) != qdrant_handler.VECTOR_DIMENSION:
            print(f"Error: El embedding generado tiene dimensión {len(embedding)}, se esperaba {qdrant_handler.VECTOR_DIMENSION}. Abortando.")
            return None
        print(f"DEBUG: Embedding generado (dimensión: {len(embedding)}) usando texto: '{text_for_embedding[:100]}...'")

        # 3. Generar un ID único para Qdrant y para el nombre del archivo
        point_id = str(uuid.uuid4())
        
        # 4. Definir la ruta final del recorte en la carpeta CLIPPINGS_DIR
        final_clipping_path = os.path.join(CLIPPINGS_DIR, f"{point_id}.png")

        # 5. Preparar payload para Qdrant
        payload = {
            "image_id": point_id,
            "image_path": final_clipping_path, # Ruta donde se guardará el recorte
            "description": description,
            "keywords": keywords,
            "type": element_type,
            "ocr_text": ocr_text,           # CAMBIO CLAVE 2: Añadir el texto OCR al payload
            "ai_extracted_text": ai_extracted_text, # CAMBIO CLAVE 3: Añadir el texto visible extraído por IA al payload
            "original_file_name": os.path.basename(clipping_image_path)
        }

        # 6. Almacenar en Qdrant
        print(f"DEBUG: Almacenando punto en Qdrant con ID: {point_id}...")
        success = qdrant_handler.upsert_point(point_id, embedding, payload)
        if not success:
            print(f"Falló el almacenamiento en Qdrant para '{clipping_image_path}'. Abortando.")
            return None
        print(f"DEBUG: Punto {point_id} almacenado en Qdrant.")

        # 7. Mover/Copiar el archivo de recorte a la carpeta 'clippings' con el ID como nombre
        print(f"DEBUG: Guardando recorte en: {final_clipping_path}...")
        Image.open(clipping_image_path).save(final_clipping_path)
        print(f"Recorte guardado en: {final_clipping_path}")

        print(f"Proceso de almacenamiento de recorte completado con éxito. ID: {point_id}")
        return point_id

    except Exception as e:
        print(f"Error general al procesar y almacenar el recorte '{clipping_image_path}': {e}")
        traceback.print_exc() # Imprime el stack trace para depuración
        return None

# MODIFICACIÓN CLAVE AQUÍ: Usar Optional[int] para indicar que monitor_to_capture puede ser int o None
def execute_action_from_text(instruction: str, monitor_to_capture: Optional[int] = None, confidence: float = 0.9):
    """
    Toma una instrucción de texto, busca el recorte más relevante en Qdrant,
    lo localiza en pantalla y ejecuta un clic.
    Args:
        instruction (str): La instrucción de texto del usuario (ej. "abrir papelera").
        monitor_to_capture (int, optional): El ID del monitor a capturar (None para todos).
        confidence (float): Nivel de confianza para la detección de imagen.
    Returns:
        bool: True si la acción se ejecutó, False en caso contrario.
    """
    print(f"\n--- Ejecutando acción para la instrucción: '{instruction}' ---")
    try:
        # 1. Generar embedding de la instrucción
        print(f"DEBUG: Generando embedding de la instrucción: '{instruction}'...")
        query_embedding = image_processor.generate_embedding_from_text(instruction)
        if len(query_embedding) != qdrant_handler.VECTOR_DIMENSION:
            print("Error: El embedding de la instrucción no tiene la dimensión esperada. Abortando búsqueda.")
            return False
        print("DEBUG: Embedding de la instrucción generado.")

        # 2. Buscar recortes relevantes en Qdrant
        print("DEBUG: Buscando el recorte más relevante en Qdrant...")
        search_results = qdrant_handler.search_points(query_embedding, limit=1) # Buscar el más relevante
        
        if not search_results:
            print(f"No se encontraron recortes relevantes en Qdrant para la instrucción: '{instruction}'.")
            return False

        best_match = search_results[0]
        match_id = best_match.id
        match_payload = best_match.payload
        match_score = best_match.score

        print("Mejor coincidencia encontrada en Qdrant:")
        print(f"   ID: {match_id}, Score: {match_score:.4f}")
        print(f"   Descripción: {match_payload.get('description', 'N/A')}")
        print(f"   Texto IA: {match_payload.get('ai_extracted_text', 'N/A')}") # Mostrar texto IA
        print(f"   Texto OCR: {match_payload.get('ocr_text', 'N/A')}")       # Mostrar texto OCR
        print(f"   Ruta de imagen: {match_payload.get('image_path', 'N/A')}")
        
        clipping_file_path = match_payload.get("image_path")
        if not clipping_file_path or not os.path.exists(clipping_file_path):
            print(f"Error: La ruta de imagen del recorte en Qdrant no es válida o no existe: {clipping_file_path}")
            return False

        # 3. Tomar captura de pantalla actual (del monitor/es deseado/s)
        print(f"DEBUG: Tomando captura de pantalla del monitor {monitor_to_capture if monitor_to_capture is not None else 'principal'}...")
        current_screenshot_image = take_screenshot(monitor_to_capture) 
        
        if current_screenshot_image is None:
            print(f"Error: No se pudo tomar una captura de pantalla del monitor {monitor_to_capture if monitor_to_capture is not None else 'principal'}.")
            return False

        # 4. Localizar el recorte en la captura de pantalla
        print(f"DEBUG: Buscando el recorte '{clipping_file_path}' en la captura de pantalla con confianza {confidence}...")
        location = find_image_on_screen(clipping_file_path, current_screenshot_image, confidence=confidence)

        if location:
            # Calcular el centro del recorte encontrado
            center_x = location.left + location.width / 2
            center_y = location.top + location.height / 2
            
            print(f"Recorte localizado en pantalla en: {location}. Clicando en ({center_x}, {center_y})...")
            pyautogui.doubleClick(center_x, center_y) # Realiza un doble clic
            print("Clic ejecutado.")
            return True
        else:
            print(f"No se pudo localizar el recorte '{clipping_file_path}' en la pantalla actual con la confianza {confidence}.")
            return False

    except Exception as e:
        print(f"Error general al ejecutar la acción para '{instruction}': {e}")
        traceback.print_exc() # Imprime el stack trace completo
        return False

if __name__ == "__main__":
    print("--- Iniciando el sistema de automatización de UI ---")

    # --- Parte 1: Ingesta Manual de Recortes (simulada) ---
    print("\n### Fase de Ingesta de Recortes ###")
    print("Para probar la ingesta, asegúrate de tener una imagen de recorte en la raíz del proyecto.")
    print("Por ejemplo, recorta el icono de la Papelera de Reciclaje y guárdalo como 'papelera_reciclaje.png'")
    
    # Puedes usar las imágenes que me has subido antes, por ejemplo:
    # 'image_01a69f.png' (icono de Excel)
    # 'image_0204d7.png' (un botón con texto)
    test_clipping_path = "icono_papelera_de_reciclaje.png" # <--- CAMBIA ESTO A TU ARCHIVO DE RECORTE DE PRUEBA
    
    if os.path.exists(test_clipping_path):
        qdrant_id_ingested = process_and_store_clipping(test_clipping_path)
        if qdrant_id_ingested:
            print(f"Recorte '{test_clipping_path}' procesado y almacenado con ID: {qdrant_id_ingested}")
        else:
            print(f"Falló el procesamiento de '{test_clipping_path}'.")
    else:
        print(f"El archivo de recorte de prueba '{test_clipping_path}' no se encontró.")
        print("Por favor, crea uno (ej. recortando el icono de la Papelera de Reciclaje) para probar la funcionalidad de ingesta.")

    # Dar un pequeño respiro al sistema antes de la siguiente fase
    time.sleep(2)

    # --- Parte 2: Ejecución de Acciones basada en Texto ---
    print("\n### Fase de Ejecución de Acciones ###")
    print("Ahora intentaremos ejecutar una acción basada en una instrucción de texto.")
    print("Asegúrate de que el elemento que intentas clicar esté visible en la pantalla.")

    # Ejemplo 1: Intentar buscar por un texto que podría estar en el OCR o la descripción
    instruction_1 = "hacer click en el botón de aceptar" # Si tu icono_papelera_de_reciclaje.png es un botón de aceptar
    print(f"\nIntento 1: Ejecutar la instrucción: '{instruction_1}'")
    success_1 = execute_action_from_text(instruction_1, monitor_to_capture=None, confidence=0.8) 
    if success_1:
        print(f"Acción para '{instruction_1}' ejecutada con éxito.")
    else:
        print(f"Falló la acción para '{instruction_1}'. Revisa logs, la descripción y la visibilidad del elemento.")
    
    time.sleep(3) # Pausa para ver el resultado

    # Ejemplo 2: Una instrucción que NO debería encontrar nada si solo tenemos el icono de la papelera
    instruction_2 = "Cerrar ventana de Chrome"
    print(f"\nIntento 2: Ejecutar la instrucción: '{instruction_2}' (no debería encontrar coincidencia si solo tienes un recorte de papelera)")
    success_2 = execute_action_from_text(instruction_2, monitor_to_capture=None, confidence=0.8)
    if success_2:
        print(f"Acción para '{instruction_2}' ejecutada con éxito (inesperado si solo tienes un recorte de papelera).")
    else:
        print(f"Falló la acción para '{instruction_2}' (esperado si no hay un recorte de 'cerrar ventana de Chrome').")

    print("\n--- Sistema de automatización de UI finalizado para esta ejecución ---")
    print("Para instrucciones más complejas que requieran múltiples pasos, se utiliza el 'automation_agent.py' con LangChain y OpenAI.")