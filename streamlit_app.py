import streamlit as st
import os
import uuid
from PIL import Image
import time
import pyautogui
import io # Para manejar la imagen cargada en memoria
import pytesseract # Importar pytesseract
import cv2 # Importar OpenCV

# Importar los módulos que hemos creado
from qdrant_handler import QdrantHandler
from image_processor import ImageProcessor
from utils.screen_utils import take_screenshot, find_image_on_screen, get_monitor_info
from qdrant_client import models # Necesario para models.VectorParams en la función de limpieza de Qdrant

# ### CAMBIO NUEVO ###
# Importar el AutomationAgent
from automation_agent import AutomationAgent
from utils.audio_utils import record_audio, transcribe_audio

# --- Inicializar handlers usando st.session_state para evitar re-inicializaciones ---
# Esto asegura que los objetos se inicialicen solo una vez por sesión de usuario de Streamlit.
if 'qdrant_handler' not in st.session_state:
    st.session_state.qdrant_handler = QdrantHandler()
    # st.write("DEBUG: QdrantHandler inicializado.") # Mensaje de depuración

# La inicialización de ImageProcessor ahora depende de qdrant_handler para obtener VECTOR_DIMENSION
# Aseguramos que qdrant_handler esté inicializado primero.
if 'image_processor' not in st.session_state:
    # Asegúrate de que ImageProcessor no intente cargar el modelo si no puede obtener la dimensión
    # Aunque tu ImageProcessor ya maneja la carga del modelo internamente.
    st.session_state.image_processor = ImageProcessor()
    # st.write("DEBUG: ImageProcessor inicializado.") # Mensaje de depuración

if 'automation_agent' not in st.session_state:
    st.session_state.automation_agent = AutomationAgent()
    # st.write("DEBUG: AutomationAgent inicializado.") # Mensaje de depuración

# Acceder a los objetos inicializados a través de st.session_state
qdrant_handler = st.session_state.qdrant_handler
image_processor = st.session_state.image_processor
automation_agent = st.session_state.automation_agent
# ### FIN CAMBIO NUEVO ###

CLIPPINGS_DIR = "clippings"
os.makedirs(CLIPPINGS_DIR, exist_ok=True) # Asegurarse de que la carpeta exista

# Definición centralizada de la carpeta temporal para evitar confusiones
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True) # Asegurarse de que la carpeta 'temp' exista

AUDIO_TEMP_DIR = "audio_temp"
os.makedirs(AUDIO_TEMP_DIR, exist_ok=True)

st.set_page_config(layout="wide", page_title="Automatización de UI con IA y RAG")

st.title("🤖 Automatización de UI y Control PLC con IA Generativa")
st.markdown("Esta aplicación te permite automatizar interacciones con la interfaz de Windows (clics) y controlar un PLC S7-1200, todo impulsado por IA generativa (Open AI), RAG y Qdrant.")

# --- Pestañas para organizar la interfaz ---
tab_ingesta, tab_acciones, tab_ocr, tab_complex_automation, tab_plc, tab_config = st.tabs([ # AÑADIDO tab_ocr
    "📝 Ingesta de Recortes",
    "🚀 Ejecutar Acciones UI",
    "👁️ OCR de Pantalla", # Nueva pestaña para OCR
    "🧠 Tareas Complejas (Agente IA)",
    "🏭 Control PLC",
    "⚙️ Configuración"
])

# --- Tab 1: Ingesta de Recortes ---
with tab_ingesta:
    st.header("Gestionar Recortes de UI")
    st.write("Sube imágenes de recortes de elementos de la UI de Windows (iconos, pestañas, botones, campos de texto, etc.) para que la IA los describa y almacene en Qdrant.")
    st.warning("⚠️ **¡Importante!** Cada imagen que subas aquí será tratada como un *nuevo* recorte y se le asignará un nuevo ID. Sube cada elemento UI solo una vez para evitar duplicados en Qdrant.")

    uploaded_file = st.file_uploader("Sube un archivo de imagen (PNG, JPG) de tu recorte", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        temp_clipping_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.{file_extension}") # Usamos TEMP_DIR

        # Asegurarse de que el archivo temporal se guarda antes del try
        with open(temp_clipping_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Recorte subido", use_container_width=True)
        st.info("Archivo subido con éxito. Procesando y generando descripción...")

        try:
            # === CÓDIGO CORREGIDO AQUÍ ===
            # La función describe_image_with_ai ahora devuelve 5 valores:
            # description, keywords, element_type, ocr_text, ai_extracted_text
            description, keywords, element_type, ocr_text, ai_extracted_text = image_processor.describe_image_with_ai(temp_clipping_path)
            # =============================

            if description:
                st.subheader("Descripción generada por Open AI:")
                st.write(f"**Descripción:** {description}")
                st.write(f"**Texto visible (IA):** {ai_extracted_text}") # Mostrar el texto extraído por IA
                st.write(f"**Texto visible (OCR):** {ocr_text}") # Mostrar el texto extraído por OCR
                st.write(f"**Palabras clave:** {', '.join(keywords)}")
                st.write(f"**Tipo de Elemento:** {element_type}")

                # Decidir qué texto usar para el embedding: descripción combinada con el mejor texto visible
                text_for_embedding = description
                if ai_extracted_text and ai_extracted_text.lower() != "ninguno":
                    text_for_embedding += f". Texto visible: {ai_extracted_text}"
                elif ocr_text and ocr_text.strip() != "":
                    text_for_embedding += f". Texto OCR: {ocr_text}"

                embedding = image_processor.generate_embedding_from_text(text_for_embedding)
                if len(embedding) == qdrant_handler.VECTOR_DIMENSION:
                    point_id = str(uuid.uuid4())
                    final_clipping_path = os.path.join(CLIPPINGS_DIR, f"{point_id}.png")

                    # Copiar el archivo subido a la carpeta de clippings con el nuevo ID
                    Image.open(temp_clipping_path).save(final_clipping_path)

                    payload = {
                        "image_id": point_id,
                        "image_path": final_clipping_path,
                        "description": description,
                        "keywords": keywords,
                        "type": element_type,
                        "ocr_text": ocr_text, # Guardar el texto OCR en el payload
                        "ai_extracted_text": ai_extracted_text, # Guardar el texto AI en el payload
                        "original_file_name": uploaded_file.name
                    }

                    if qdrant_handler.upsert_point(point_id, embedding, payload):
                        st.success(f"Recorte procesado y almacenado en Qdrant con ID: `{point_id}`")
                        st.write(f"Guardado en: `{final_clipping_path}`")
                    else:
                        st.error("Error al guardar el recorte en Qdrant.")
                else:
                    st.error(f"Error: El embedding generado tiene una dimensión incorrecta ({len(embedding)}). Esperado: {qdrant_handler.VECTOR_DIMENSION}.")
            else:
                st.error("Open AI no pudo generar una descripción para la imagen. El archivo temporal se eliminará.") # Mensaje más claro
        except Exception as e:
            st.error(f"Ocurrió un error durante el procesamiento del recorte: {e}")
        finally:
            # Asegurarse de que el archivo temporal se elimine siempre, incluso si hay un error
            if os.path.exists(temp_clipping_path):
                try:
                    os.remove(temp_clipping_path)
                    st.info(f"Archivo temporal '{temp_clipping_path}' eliminado.")
                except Exception as e_del:
                    st.error(f"ERROR: No se pudo eliminar el archivo temporal '{temp_clipping_path}': {e_del}. Por favor, verifica permisos o si el archivo está en uso.")


# --- Tab 2: Ejecutar Acciones UI ---
with tab_acciones:
    st.header("Ejecutar Acciones en la UI")
    st.write("Introduce una instrucción de texto o voz para que la aplicación localice y clique un elemento en tu pantalla.")

    st.subheader("Configuración de Captura de Pantalla")
    monitors_info = get_monitor_info()
    monitor_options = ["Todos los monitores"] + [f"Monitor {m['id']} ({m['width']}x{m['height']})" for m in monitors_info if m['id'] is not None]
    selected_monitor_option_actions = st.selectbox("Selecciona un monitor para la captura:", monitor_options, key="monitor_select_actions") # Añadido key

    monitor_to_capture_id_actions = None
    if selected_monitor_option_actions != "Todos los monitores":
        monitor_to_capture_id_actions = int(selected_monitor_option_actions.split('(')[0].replace('Monitor ', '').strip())

    confidence_level = st.slider("Nivel de confianza para la detección de imagen (PyAutoGUI):", 0.0, 1.0, 0.8, 0.05)

    instruction_text = st.text_input("Instrucción de texto (ej. 'Abrir papelera de reciclaje', 'Clicar en el botón Aceptar'):")

    if st.button("Ejecutar Acción"):
        if instruction_text:
            st.info(f"Buscando elementos relacionados con: '{instruction_text}'...")
            
            # Inicializar screenshot_temp_path aquí para que esté disponible en el finally
            screenshot_temp_path = None 

            try:
                query_embedding = image_processor.generate_embedding_from_text(instruction_text)

                if len(query_embedding) != qdrant_handler.VECTOR_DIMENSION:
                    st.error("Error al generar embedding para la instrucción. Dimensión incorrecta.")
                else:
                    search_results = qdrant_handler.search_points(query_embedding, limit=3)

                    if search_results:
                        st.subheader("Resultados de la búsqueda RAG:")
                        best_match = search_results[0]

                        st.write(f"**Mejor coincidencia (Score: {best_match.score:.4f}):**")
                        st.write(f"    **Descripción:** {best_match.payload.get('description', 'N/A')}")
                        st.write(f"    **Texto visible (IA):** {best_match.payload.get('ai_extracted_text', 'N/A')}") # Mostrar en la búsqueda
                        st.write(f"    **Texto visible (OCR):** {best_match.payload.get('ocr_text', 'N/A')}") # Mostrar en la búsqueda
                        st.write(f"    **Tipo:** {best_match.payload.get('type', 'N/A')}")
                        st.write(f"    **Ruta de Imagen:** `{best_match.payload.get('image_path', 'N/A')}`")

                        clipping_file_path = best_match.payload.get("image_path")
                        if clipping_file_path and os.path.exists(clipping_file_path):
                            st.image(clipping_file_path, caption="Recorte Sugerido", width=100)

                            st.write("Tomando captura de pantalla y buscando el elemento...")
                            current_screenshot_image = None
                            
                            # Genera un nombre de archivo temporal para la captura de pantalla
                            screenshot_temp_path = os.path.join(TEMP_DIR, f"screenshot_{uuid.uuid4()}.png")

                            try:
                                current_screenshot_image = take_screenshot(monitor_number=monitor_to_capture_id_actions) # Usar la variable de esta pestaña
                                current_screenshot_image.save(screenshot_temp_path) # Guarda la captura temporalmente
                            except ValueError as ve:
                                st.error(f"Error de monitor al tomar captura: {ve}")
                                current_screenshot_image = None
                            except Exception as e:
                                st.error(f"Error general al tomar captura de pantalla: {e}")
                                current_screenshot_image = None
                            
                            # La limpieza de la captura temporal se hará en el finally principal
                            if current_screenshot_image:
                                # Aquí pasamos la ruta de la captura temporal a find_image_on_screen
                                # que es lo que espera PyAutoGUI para el segundo argumento.
                                location = find_image_on_screen(clipping_file_path, screenshot_temp_path, confidence=confidence_level)

                                if location:
                                    center_x = location.left + location.width / 2
                                    center_y = location.top + location.height / 2
                                    st.success(f"Elemento localizado en pantalla en: ({location.left}, {location.top}, {location.width}, {location.height}). Clicando...")
                                    pyautogui.doubleClick(center_x, center_y) # Doble clic para asegurarnos de que se activa el elemento
                                    st.success("¡Clic ejecutado con éxito!")
                                else:
                                    st.warning("No se pudo localizar el elemento en la pantalla actual con la confianza dada.")
                            else:
                                st.error("No se pudo obtener una captura de pantalla válida para buscar el elemento.")

                        else:
                            st.error(f"Error: La imagen del recorte ({clipping_file_path}) no se encontró en el sistema de archivos.")
                    else:
                        st.info("No se encontraron coincidencias suficientes en Qdrant para tu instrucción.")
            except Exception as e:
                st.error(f"Ocurrió un error al procesar la instrucción: {e}")
            finally: # Asegura que la captura de pantalla temporal siempre se elimine
                if screenshot_temp_path and os.path.exists(screenshot_temp_path):
                    try:
                        os.remove(screenshot_temp_path)
                        st.info(f"Archivo temporal de captura '{screenshot_temp_path}' eliminado.")
                    except Exception as e_del:
                        st.error(f"ERROR: No se pudo eliminar la captura temporal '{screenshot_temp_path}': {e_del}. Por favor, verifica permisos o si el archivo está en uso.")
        else:
            st.warning("Por favor, introduce una instrucción de texto.")

# --- Tab 3: OCR de Pantalla (NUEVA PESTAÑA) ---
with tab_ocr:
    st.header("Reconocimiento de Texto (OCR) de Pantalla")
    st.write("Captura una porción de tu pantalla y extrae el texto usando OCR.")

    st.subheader("Configuración de Captura de Pantalla para OCR")
    monitors_info_ocr = get_monitor_info()
    monitor_options_ocr = ["Todos los monitores"] + [f"Monitor {m['id']} ({m['width']}x{m['height']})" for m in monitors_info_ocr if m['id'] is not None]
    selected_monitor_option_ocr = st.selectbox("Selecciona un monitor para la captura de OCR:", monitor_options_ocr, key="monitor_select_ocr")

    monitor_to_capture_id_ocr = None
    if selected_monitor_option_ocr != "Todos los monitores":
        monitor_to_capture_id_ocr = int(selected_monitor_option_ocr.split('(')[0].replace('Monitor ', '').strip())

    st.info("Para capturar una región específica, puedes usar la herramienta de recorte de tu sistema operativo (ej. Recortes y anotación en Windows) y luego subir la imagen en la pestaña de 'Ingesta de Recortes'. Aquí se capturará toda la pantalla seleccionada.")

    if st.button("Realizar OCR de Pantalla"):
        ocr_screenshot_path = None
        try:
            st.info("Tomando captura de pantalla para OCR...")
            ocr_screenshot_image = take_screenshot(monitor_number=monitor_to_capture_id_ocr)
            
            ocr_screenshot_path = os.path.join(TEMP_DIR, f"ocr_screenshot_{uuid.uuid4()}.png")
            ocr_screenshot_image.save(ocr_screenshot_path)

            st.image(ocr_screenshot_image, caption="Captura de Pantalla para OCR", use_container_width=True)

            with st.spinner("Realizando OCR..."):
                # Aquí se invoca la funcionalidad de OCR.
                # Asegúrate de que ImageProcessor tenga el método perform_ocr_on_image.
                # O si no, puedes realizar el OCR directamente aquí si es una función simple.
                # Por ahora, asumiremos que ImageProcessor puede hacerlo.
                # Si no existe, tendrías que añadirlo en image_processor.py.
                # Ejemplo de cómo se llamaría:
                recognized_text = image_processor.perform_ocr_on_image(ocr_screenshot_path) 
                
                if recognized_text:
                    st.subheader("Texto Reconocido (OCR):")
                    st.code(recognized_text)
                else:
                    st.warning("No se detectó texto en la captura de pantalla.")
        except pytesseract.TesseractNotFoundError:
            st.error("Error: Tesseract OCR no está instalado o no se encuentra en el PATH. Por favor, instálalo desde https://tesseract-ocr.github.io/tessdoc/Installation.html y asegúrate de que esté en tu PATH o configura pytesseract.pytesseract.tesseract_cmd.")
        except Exception as e:
            st.error(f"Ocurrió un error al realizar OCR: {e}")
        finally:
            if ocr_screenshot_path and os.path.exists(ocr_screenshot_path):
                try:
                    os.remove(ocr_screenshot_path)
                    st.info(f"Archivo temporal de OCR '{ocr_screenshot_path}' eliminado.")
                except Exception as e_del:
                    st.error(f"ERROR: No se pudo eliminar el archivo temporal de OCR '{ocr_screenshot_path}': {e_del}. Por favor, verifica permisos o si el archivo está en uso.")

# --- Tab 4: Automatización de Tareas Complejas (Agente IA) --- (Ahora es la 4ª pestaña)
with tab_complex_automation:
    st.header("Automatización de Tareas Complejas (Agente IA)")
    st.write("Introduce una instrucción compleja para el agente de IA. El agente intentará desglosarla y usar las herramientas disponibles (UI y PLC) para completarla.")
    st.info("""
    **Ejemplos de instrucciones:**
    - 'Abre la Papelera de Reciclaje.'
    - 'Arranca la secuencia de mezclado en el PLC y luego dime el estado del Batch ID (DB1.DBW10).' (Requiere que DB1.DBX0.0 sea el bit de arranque y DB1.DBW10 el Batch ID).
    - 'Escribe 'Hola ChatGPT' en la barra de búsqueda de Windows (puedes describirla como 'icono de búsqueda' o 'barra de búsqueda').'
    - 'Toma una captura de pantalla del monitor principal y dime la hora actual.'
    """)

    # --- CAMBIOS CLAVE AQUÍ: Usar st.session_state para la instrucción compleja ---
    if 'complex_instruction_text' not in st.session_state:
        st.session_state.complex_instruction_text = ""

    complex_instruction = st.text_area(
        "Instrucción para el Agente:",
        value=st.session_state.complex_instruction_text, # Lee el valor de session_state
        height=100,
        placeholder="Ej: Abrir TIA Portal, navegar a mi proyecto, y arrancar el motor principal del PLC.",
        key="main_complex_instruction_input" # Key para el widget
    )

    # Actualizar st.session_state si el usuario escribe directamente en el text_area
    st.session_state.complex_instruction_text = complex_instruction

    # --- Sección para Instrucción por Voz ---
    st.subheader("O bien, introduce la instrucción por voz:")
    audio_duration = st.slider("Duración de la grabación (segundos):", 1, 10, 5) # Slider para la duración

    if st.button("Grabar Instrucción por Voz"):
        # Inicializar audio_filename y recorded_path fuera del try para que estén disponibles en el finally
        audio_filename = None
        recorded_path = None
        with st.spinner(f"Grabando audio durante {audio_duration} segundos..."):
            try:
                # Crear un nombre de archivo único para la grabación
                audio_filename = os.path.join(AUDIO_TEMP_DIR, f"instruction_{uuid.uuid4()}.wav")

                # Grabar el audio
                recorded_path = record_audio(audio_filename, duration=audio_duration)

                if recorded_path:
                    st.audio(recorded_path, format="audio/wav")
                    st.success(f"Audio grabado en: {recorded_path}")

                    with st.spinner("Transcribiendo audio..."):
                        transcribed_text = transcribe_audio(recorded_path)
                        if "ERROR" in transcribed_text:
                            st.error(f"Error de transcripción: {transcribed_text}")
                        else:
                            st.success("Transcripción completada.")
                            # Guardar la transcripción directamente en session_state
                            st.session_state.complex_instruction_text = transcribed_text
                            st.rerun() # Fuerza una nueva ejecución para actualizar el text_area con el valor transcrito
                else:
                    st.error("La grabación de audio falló o no se generó un archivo.")

            except Exception as e:
                st.error(f"Error en la grabación/transcripción de voz: {e}")
            finally: # Asegura que el archivo de audio temporal siempre se elimine
                if recorded_path and os.path.exists(recorded_path): # Usar recorded_path para la limpieza
                    try:
                        os.remove(recorded_path)
                        st.info(f"Archivo de audio temporal '{recorded_path}' eliminado.")
                    except Exception as e_del:
                        st.error(f"ERROR: No se pudo eliminar el archivo de audio temporal '{recorded_path}': {e_del}. Por favor, verifica permisos o si el archivo está en uso.")

    # El botón de ejecutar ahora lee la instrucción de st.session_state
    if st.button("Ejecutar Tarea Compleja por Texto", key="run_complex_task_button"):
        if st.session_state.complex_instruction_text: # Usar el valor persistente de session_state
            with st.spinner("El Agente está procesando la tarea... Esto puede tomar un tiempo y mostrará los pasos en la consola."):
                try:
                    # Pasar la instrucción desde st.session_state al agente
                    st_response = automation_agent.run_task(st.session_state.complex_instruction_text)
                    st.subheader("Resultado del Agente:")
                    st.success(st_response)
                except Exception as e:
                    st.error(f"Error al ejecutar la tarea compleja: {e}")
        else:
            st.warning("Por favor, introduce una instrucción para el agente (texto o voz).")

# --- Tab 5: Control PLC (Ahora es la 5ª pestaña) ---
with tab_plc: 
    st.header("Control y Simulación de PLC (Siemens S7-1200)")
    st.write("Esta sección permitirá la interacción directa con el PLC (lectura/escritura de tags) o la visualización de su estado.")
    st.markdown("""
    Funcionalidades planeadas para esta pestaña:
    - **Conexión al PLC S7-1200** (gestionada internamente por el `AutomationAgent` y `PLCHandler`).
    - **Lectura/Escritura manual** de tags (DBs, M) para depuración o control directo.
    - **Visualización en tiempo real** de valores clave del PLC.
    """)
    st.info("Actualmente, la interacción con el PLC se realiza a través de las instrucciones complejas del Agente IA en la pestaña 'Tareas Complejas'.")

# --- Tab 6: Configuración (Ahora es la 6ª pestaña) ---
with tab_config: 
    st.header("Configuración de la Aplicación")
    st.write("Aquí puedes ver y gestionar las configuraciones de la aplicación.")

    st.subheader("Variables de Entorno Cargadas")
    st.code(f"QDRANT_HOST: {os.getenv('QDRANT_HOST')}")
    st.code(f"QDRANT_API_KEY: {'*' * len(os.getenv('QDRANT_API_KEY')) if os.getenv('QDRANT_API_KEY') else 'No configurado'}")
    st.code(f"OPENAI_API_KEY: {'*' * len(os.getenv('OPENAI_API_KEY')) if os.getenv('OPENAI_API_KEY') else 'No configurado'}")
    st.code(f"PLC_IP_ADDRESS: {os.getenv('PLC_IP_ADDRESS')}")
    st.code(f"PLC_RACK: {os.getenv('PLC_RACK')}")
    st.code(f"PLC_SLOT: {os.getenv('PLC_SLOT')}")

    st.subheader("Información de Monitores Detectados")
    monitors_data = get_monitor_info()
    if monitors_data:
        for mon in monitors_data:
            st.json(mon)
    else:
        st.info("No se pudo obtener la información de los monitores.")

    st.subheader("Directorio de Recortes")
    st.write(f"Los recortes se guardan en: `{CLIPPINGS_DIR}`")
    st.write(f"Número de recortes guardados: {len(os.listdir(CLIPPINGS_DIR)) if os.path.exists(CLIPPINGS_DIR) else 0}")
    
    st.subheader("Herramientas de Limpieza")
    if st.button(f"Vaciar carpeta '{TEMP_DIR}'", key="clear_temp_button"):
        try:
            for f in os.listdir(TEMP_DIR):
                os.remove(os.path.join(TEMP_DIR, f))
            st.success(f"Carpeta '{TEMP_DIR}' vaciada.")
        except Exception as e:
            st.error(f"Error al vaciar carpeta '{TEMP_DIR}': {e}")

    # Asegúrate de importar `models` de `qdrant_client` al inicio del archivo si no lo está.
    # from qdrant_client import models
    if st.button("Vaciar carpeta 'clippings' y colección Qdrant (¡CUIDADO!)", key="clear_clippings_qdrant_button"):
        st.warning("Esta acción borrará PERMANENTEMENTE todos los recortes de UI guardados y la colección en Qdrant.")
        confirm = st.checkbox("Estoy seguro de que quiero borrar TODOS los recortes de UI y vaciar Qdrant.", key="confirm_delete_clippings_qdrant")
        if confirm and st.button("Confirmar Borrado TOTAL de Recortes y Qdrant"):
            try:
                # Borrar archivos de clippings
                if os.path.exists(CLIPPINGS_DIR):
                    for f in os.listdir(CLIPPINGS_DIR):
                        os.remove(os.path.join(CLIPPINGS_DIR, f))
                    st.success(f"Carpeta '{CLIPPINGS_DIR}' vaciada.")
                else:
                    st.info(f"La carpeta '{CLIPPINGS_DIR}' no existe o ya está vacía.")
                
                # Recrear la colección Qdrant (esto la vacía)
                # Asegúrate de que qdrant_handler.COLLECTION_NAME y qdrant_handler.VECTOR_DIMENSION están bien inicializados
                qdrant_handler.client.recreate_collection(
                    collection_name=qdrant_handler.COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=qdrant_handler.VECTOR_DIMENSION, distance=models.Distance.COSINE),
                )
                st.success(f"Colección Qdrant '{qdrant_handler.COLLECTION_NAME}' recreada (vaciada).")
                st.rerun() # Recargar la página para reflejar los cambios
            except Exception as e:
                st.error(f"Error al vaciar carpeta 'clippings' o colección Qdrant: {e}")