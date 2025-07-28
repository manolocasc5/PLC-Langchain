import streamlit as st
import os
import uuid
from PIL import Image
import time
import pyautogui
import io # Para manejar la imagen cargada en memoria

# Importar los módulos que hemos creado
from qdrant_handler import QdrantHandler
from image_processor import ImageProcessor
from utils.screen_utils import take_screenshot, find_image_on_screen, get_monitor_info

# ### CAMBIO NUEVO ###
# Importar el AutomationAgent
from automation_agent import AutomationAgent
from utils.audio_utils import record_audio, transcribe_audio

# --- Inicializar handlers usando st.session_state para evitar re-inicializaciones ---
# Esto asegura que los objetos se inicialicen solo una vez por sesión de usuario de Streamlit.
if 'qdrant_handler' not in st.session_state:
    st.session_state.qdrant_handler = QdrantHandler()
    # st.write("DEBUG: QdrantHandler inicializado.") # Mensaje de depuración

if 'image_processor' not in st.session_state:
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

AUDIO_TEMP_DIR = "audio_temp"
os.makedirs(AUDIO_TEMP_DIR, exist_ok=True)

st.set_page_config(layout="wide", page_title="Automatización de UI con IA y RAG")

st.title("🤖 Automatización de UI y Control PLC con IA Generativa")
st.markdown("Esta aplicación te permite automatizar interacciones con la interfaz de Windows (clics) y controlar un PLC S7-1200, todo impulsado por IA generativa (Open AI), RAG y Qdrant.")

# --- Pestañas para organizar la interfaz ---
tab_ingesta, tab_acciones, tab_complex_automation, tab_plc, tab_config = st.tabs([
    "📝 Ingesta de Recortes",
    "🚀 Ejecutar Acciones UI",
    "🧠 Tareas Complejas (Agente IA)",
    "🏭 Control PLC",
    "⚙️ Configuración"
])

# --- Tab 1: Ingesta de Recortes ---
with tab_ingesta:
    st.header("Gestionar Recortes de UI")
    st.write("Sube imágenes de recortes de elementos de la UI de Windows (iconos, pestañas, botones, campos de texto, etc.) para que la IA los describa y almacene en Qdrant.")

    uploaded_file = st.file_uploader("Sube un archivo de imagen (PNG, JPG) de tu recorte", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Guardar temporalmente el archivo subido
        file_extension = uploaded_file.name.split('.')[-1]
        temp_clipping_path = os.path.join("temp", f"{uuid.uuid4()}.{file_extension}")
        os.makedirs("temp", exist_ok=True) # Asegurarse de que la carpeta 'temp' exista

        with open(temp_clipping_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Recorte subido", use_container_width=True) # CAMBIO: use_column_width -> use_container_width
        st.success("Archivo subido con éxito. Procesando...")

        # Llama a la función de procesamiento
        try:
            description, keywords, element_type = image_processor.describe_image_with_ai(temp_clipping_path)

            if description:
                st.subheader("Descripción generada por Open AI:")
                st.write(f"**Descripción:** {description}")
                st.write(f"**Palabras clave:** {', '.join(keywords)}")
                st.write(f"**Tipo de Elemento:** {element_type}")

                embedding = image_processor.generate_embedding_from_text(description)
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
                st.error("Open AI no pudo generar una descripción para la imagen.")

            # Limpiar el archivo temporal
            if os.path.exists(temp_clipping_path):
                os.remove(temp_clipping_path)
                st.info(f"Archivo temporal '{temp_clipping_path}' eliminado.")

        except Exception as e:
            st.error(f"Ocurrió un error durante el procesamiento: {e}")
            if os.path.exists(temp_clipping_path):
                os.remove(temp_clipping_path)

# --- Tab 2: Ejecutar Acciones UI ---
with tab_acciones:
    st.header("Ejecutar Acciones en la UI")
    st.write("Introduce una instrucción de texto o voz para que la aplicación localice y clique un elemento en tu pantalla.")

    st.subheader("Configuración de Captura de Pantalla")
    monitors_info = get_monitor_info()
    monitor_options = ["Todos los monitores"] + [f"Monitor {m['id']} ({m['width']}x{m['height']})" for m in monitors_info if m['id'] is not None]
    selected_monitor_option = st.selectbox("Selecciona un monitor para la captura:", monitor_options)

    monitor_to_capture_id = None
    if selected_monitor_option != "Todos los monitores":
        monitor_to_capture_id = int(selected_monitor_option.split('(')[0].replace('Monitor ', '').strip())

    confidence_level = st.slider("Nivel de confianza para la detección de imagen (PyAutoGUI):", 0.0, 1.0, 0.8, 0.05)

    instruction_text = st.text_input("Instrucción de texto (ej. 'Abrir papelera de reciclaje', 'Clicar en el botón Aceptar'):")

    if st.button("Ejecutar Acción"):
        if instruction_text:
            st.info(f"Buscando elementos relacionados con: '{instruction_text}'...")

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
                        st.write(f"   **Descripción:** {best_match.payload.get('description', 'N/A')}")
                        st.write(f"   **Tipo:** {best_match.payload.get('type', 'N/A')}")
                        st.write(f"   **Ruta de Imagen:** `{best_match.payload.get('image_path', 'N/A')}`")

                        clipping_file_path = best_match.payload.get("image_path")
                        if clipping_file_path and os.path.exists(clipping_file_path):
                            st.image(clipping_file_path, caption="Recorte Sugerido", width=100)

                            st.write("Tomando captura de pantalla y buscando el elemento...")
                            current_screenshot_image = None
                            try:
                                current_screenshot_image = take_screenshot(monitor_number=monitor_to_capture_id)
                                current_screenshot_image_path = "temp_current_screenshot.png"
                                current_screenshot_image.save(current_screenshot_image_path)
                                # st.image(current_screenshot_image, caption="Captura de Pantalla Actual", use_column_width=True)
                            except ValueError as ve:
                                st.error(f"Error de monitor: {ve}")
                                current_screenshot_image = None
                            except Exception as e:
                                st.error(f"Error al tomar captura de pantalla: {e}")
                                current_screenshot_image = None

                            if current_screenshot_image:
                                location = find_image_on_screen(clipping_file_path, current_screenshot_image, confidence=confidence_level)

                                if location:
                                    center_x = location.left + location.width / 2
                                    center_y = location.top + location.height / 2
                                    st.success(f"Elemento localizado en pantalla en: ({location.left}, {location.top}, {location.width}, {location.height}). Clicando...")
                                    # pyautogui.click(center_x, center_y)
                                    pyautogui.doubleClick(center_x, center_y) # Doble clic para asegurarnos de que se activa el elemento
                                    st.success("¡Clic ejecutado con éxito!")
                                else:
                                    st.warning("No se pudo localizar el elemento en la pantalla actual con la confianza dada.")
                            else:
                                st.error("No se pudo obtener una captura de pantalla para buscar el elemento.")

                            if os.path.exists(current_screenshot_image_path):
                                os.remove(current_screenshot_image_path) # Limpiar archivo temporal

                        else:
                            st.error(f"Error: La imagen del recorte ({clipping_file_path}) no se encontró en el sistema de archivos.")
                    else:
                        st.info("No se encontraron coincidencias suficientes en Qdrant para tu instrucción.")
            except Exception as e:
                st.error(f"Ocurrió un error al procesar la instrucción: {e}")
        else:
            st.warning("Por favor, introduce una instrucción de texto.")

# --- Tab 3: Automatización de Tareas Complejas (Agente IA) ---
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

                        # Limpiar el archivo de audio temporal
                        os.remove(recorded_path)
                        st.info(f"Archivo de audio temporal '{recorded_path}' eliminado.")

            except Exception as e:
                st.error(f"Error en la grabación/transcripción de voz: {e}")
                # Limpiar si hubo error y el archivo existe
                if 'audio_filename' in locals() and os.path.exists(audio_filename):
                    os.remove(audio_filename)

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

# --- Tab 4: Control PLC (Contenido original) ---
with tab_plc: # Ahora tab_plc es la cuarta pestaña
    st.header("Control y Simulación de PLC (Siemens S7-1200)")
    st.write("Esta sección permitirá la interacción directa con el PLC (lectura/escritura de tags) o la visualización de su estado.")
    st.markdown("""
    Funcionalidades planeadas para esta pestaña:
    - **Conexión al PLC S7-1200** (gestionada internamente por el `AutomationAgent` y `PLCHandler`).
    - **Lectura/Escritura manual** de tags (DBs, M) para depuración o control directo.
    - **Visualización en tiempo real** de valores clave del PLC.
    """)
    st.info("Actualmente, la interacción con el PLC se realiza a través de las instrucciones complejas del Agente IA en la pestaña 'Tareas Complejas'.")

# --- Tab 5: Configuración (Contenido original, solo cambio de índice de pestaña) ---
with tab_config: # Ahora tab_config es la quinta pestaña
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