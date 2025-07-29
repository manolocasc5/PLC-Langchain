import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool # Importa el decorador @tool
from langchain_core.messages import HumanMessage, AIMessage # Importar para el historial de chat
import traceback
from typing import List, Dict, Any, Tuple, Optional, Union
import time
import pyautogui
import uuid # Para generar nombres de archivos únicos
import tempfile # Para manejar directorios temporales de forma segura
from PIL import Image # NECESARIO: Importar PIL para trabajar con imágenes para OCR

# Importar las clases y funciones necesarias de nuestros módulos
from qdrant_handler import QdrantHandler
from image_processor import ImageProcessor
# Asumo que take_screenshot devuelve PIL.Image y find_image_on_screen puede tomar PIL.Image directamente
from utils.screen_utils import take_screenshot, find_image_on_screen 
from plc_handler import PLCHandler, PLCConnectionError, PLCReadWriteError

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY debe estar configurado en el archivo .env")

# Inicializar los handlers necesarios
# Estas instancias se inicializan una única vez al cargar el módulo
qdrant_handler = QdrantHandler()
image_processor = ImageProcessor()
plc_handler = PLCHandler()

# Definir la carpeta temporal para el agente si es necesaria, aunque take_screenshot ya la usa internamente
# para las imágenes que necesita PyAutoGUI
AGENT_TEMP_DIR = "agent_temp_files"
os.makedirs(AGENT_TEMP_DIR, exist_ok=True)


class AutomationAgent:
    def __init__(self):
        # Utilizar el modelo "gpt-4o" para un mejor rendimiento
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0) 
        
        # Definir las herramientas que el agente de LangChain puede usar
        self.tools = self._define_tools() # Llama al método para obtener la lista de herramientas
        
        # Definir el prompt del agente
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                "Eres un asistente experto en automatización de interfaces de usuario y control de PLCs Siemens S7-1200. "
                "Tu objetivo es ayudar al usuario a realizar tareas complejas en Windows y en el PLC. "
                "Utiliza las herramientas disponibles para lograr los objetivos. "
                "Prioriza siempre el uso de las herramientas para interacciones con la UI y el PLC. "
                "Sé **extremadamente preciso y descriptivo** con las 'description_or_instruction' que uses para la herramienta `search_and_click_ui_element`. "
                "Incluye el tipo de elemento (ej. 'icono', 'botón'), el texto visible exacto y cualquier característica visual distintiva para asegurar que se encuentre el elemento correcto y no uno parecido. "
                "Cuando el usuario te dé una instrucción compleja que involucre una secuencia de pasos en la UI (como abrir programas, navegar por menús, o escribir texto en campos), desglosa la tarea en pasos individuales y llama a las herramientas `search_and_click_ui_element` y `write_text_ui` para cada paso. "
                "Por ejemplo, si el usuario dice 'abre el TIA Portal y crea un nuevo proyecto', tu primer paso debería ser `search_and_click_ui_element(description_or_instruction='icono de TIA Portal V15')` y luego continuar con los pasos para crear el proyecto. "
                "Cuando el usuario te pida escribir código en un IDE, primero deberás usar las herramientas de UI para navegar hasta ese IDE y localizar el área donde se escribe el código. Una vez localizado, utiliza la herramienta `write_text_ui` para insertar el código proporcionado por ti o por el usuario. "
                "Para la interacción con el PLC, puedes leer y escribir en Data Blocks (DBs) o en memoria Merker (M), y con tipos específicos (BOOL, INT, REAL)."
                "Cuando interactúes con el PLC, el usuario puede darte comandos estructurados (ej. 'WRITE DB1.DBW10 INT 123') o lenguaje natural (ej. 'Arranca la secuencia de mezclado'). "
                "Si es lenguaje natural para el PLC, tradúcelo a operaciones de lectura/escritura de bits/bytes/palabras en el PLC y usa la herramienta adecuada."
                "Muestra el estado actual del PLC cuando se te pida o después de una operación de escritura relevante."
                "Utiliza la herramienta `perform_ocr_on_screen` cuando necesites leer texto directamente de la pantalla, como valores numéricos, etiquetas, o cualquier información textual que no pueda ser obtenida a través de la búsqueda de elementos UI o interacciones directas con el PLC. "
                "Esta herramienta es útil para extraer datos de interfaces que no son fácilmente accesibles por otros medios, como HMI, aplicaciones legacy, o reportes visuales. "
                "Especifica claramente la región de interés (x, y, width, height) o la descripción del elemento UI si la región es la de un recorte conocido."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        # Crear el agente que usa llamadas a herramientas
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        
        # Crear el ejecutor del agente
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

        self.chat_history = [] # Para mantener el contexto de la conversación


    def _define_tools(self) -> List[tool]:
        """
        Define las herramientas que el agente puede utilizar.
        Estas funciones se convierten en herramientas de LangChain gracias al decorador @tool.
        Acceden a las instancias globales de qdrant_handler, image_processor y plc_handler.
        """

        @tool
        def search_and_click_ui_element(description_or_instruction: str, monitor_id: Optional[int] = None, confidence: float = 0.8) -> str:
            """
            Busca un elemento de UI en pantalla basado en su descripción y hace clic en él.
            Útil para navegar por ventanas, abrir aplicaciones o interactuar con botones/iconos.
            La 'description_or_instruction' debe ser una descripción clara, concisa y única del elemento a buscar,
            ej: 'icono de la papelera de reciclaje', 'botón Aceptar', 'pestaña Configuración', 'carpeta Prueba'.
            'monitor_id': ID del monitor (0 para el principal). Si es None, busca en el monitor principal.
            'confidence': Umbral de confianza para la detección de imagen (0.0 a 1.0).
            Devuelve 'SUCCESS: Clic ejecutado en [Descripción]' o 'ERROR: [Mensaje de error]'.
            """
            print("\n[TOOL] search_and_click_ui_element invocado.")
            print(f"DEBUG: description_or_instruction recibida: '{description_or_instruction}'")
            print(f"DEBUG: monitor_id: {monitor_id}, confidence: {confidence}")

            clipping_file_path = None # Inicializar para el bloque finally
            
            try:
                # 1. Generar embedding de la instrucción
                query_embedding = image_processor.generate_embedding_from_text(description_or_instruction)
                
                if len(query_embedding) != qdrant_handler.VECTOR_DIMENSION:
                    return f"ERROR: El embedding de la instrucción no tiene la dimensión esperada ({len(query_embedding)} vs {qdrant_handler.VECTOR_DIMENSION})."
                
                # 2. Buscar recortes relevantes en Qdrant
                print(f"DEBUG: Buscando en Qdrant para: '{description_or_instruction}'")
                search_results = qdrant_handler.search_points(query_embedding, limit=1)
                
                if not search_results:
                    print(f"WARNING: No se encontraron recortes relevantes en Qdrant para: '{description_or_instruction}'.")
                    return f"ERROR: No se encontraron recortes relevantes en Qdrant para: '{description_or_instruction}'. Asegúrate de que el recorte esté ingresado y la descripción sea precisa."
                
                best_match = search_results[0]
                clipping_file_path = best_match.payload.get("image_path")
                
                print(f"DEBUG: Mejor coincidencia en Qdrant (score: {best_match.score:.4f}): {best_match.payload.get('description')} (Path: {clipping_file_path})")

                if not clipping_file_path or not os.path.exists(clipping_file_path):
                    print(f"ERROR: La ruta de imagen del recorte no es válida o no existe: {clipping_file_path}")
                    return f"ERROR: La ruta de imagen del recorte no es válida o no existe: {clipping_file_path}. Por favor, verifica la carpeta 'clippings'."
                
                # 3. Tomar captura de pantalla actual
                print(f"DEBUG: Tomando captura de pantalla del monitor {monitor_id if monitor_id is not None else 'principal'}.")
                # take_screenshot ya devuelve un objeto PIL.Image
                current_screenshot_image = take_screenshot(monitor_number=monitor_id) 
                
                if current_screenshot_image is None:
                    return f"ERROR: No se pudo tomar una captura de pantalla del monitor {monitor_id if monitor_id is not None else 'principal'}."

                # 4. Localizar el recorte en la captura de pantalla
                # Opción b) Es la más sencilla y compatible con pyautogui, asegurando la limpieza.
                # Generamos un nombre de archivo temporal para la captura de pantalla del agente.
                # Usamos tempfile para un manejo más robusto de archivos temporales.
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=AGENT_TEMP_DIR) as temp_screenshot_file:
                    current_screenshot_image.save(temp_screenshot_file.name)
                    temp_screenshot_path = temp_screenshot_file.name
                
                location = find_image_on_screen(clipping_file_path, temp_screenshot_path, confidence=confidence)
                
                # Asegurar la eliminación del archivo temporal de la captura
                if os.path.exists(temp_screenshot_path):
                    os.remove(temp_screenshot_path)
                    print(f"DEBUG: Archivo temporal de captura '{temp_screenshot_path}' eliminado.")

                if location:
                    center_x = location.left + location.width / 2
                    center_y = location.top + location.height / 2
                    
                    # Un solo click suele ser suficiente para botones/iconos, 
                    # pero doubleClick puede ser necesario para abrir carpetas/programas.
                    # El agente debe decidir si es necesario un doble clic basado en el contexto.
                    # Por simplicidad y para asegurar la acción, lo dejamos como doubleClick por ahora.
                    pyautogui.doubleClick(center_x, center_y) 
                    
                    print(f"DEBUG: Clic ejecutado en ({center_x}, {center_y}) para: '{description_or_instruction}'")
                    return f"SUCCESS: Clic ejecutado en '{description_or_instruction}'."
                else:
                    print(f"WARNING: No se pudo localizar '{description_or_instruction}' en la pantalla actual con confianza {confidence}.")
                    return f"ERROR: No se pudo localizar '{description_or_instruction}' en la pantalla actual con confianza {confidence}. Intenta ajustar la descripción o la confianza."
                
            except Exception as e:
                traceback.print_exc()
                return f"ERROR: Ocurrió un error en search_and_click_ui_element: {e}"

        @tool
        def write_text_ui(text_to_write: str, target_element_description: Optional[str] = None, monitor_id: Optional[int] = None, confidence: float = 0.8) -> str:
            """
            Escribe texto en un campo de UI. Si se proporciona 'target_element_description',
            intentará localizar y clicar el campo antes de escribir. Si no, escribirá donde esté el foco actual.
            'target_element_description': ej. 'campo de texto Nombre de Usuario', 'barra de búsqueda'.
            Devuelve 'SUCCESS: Texto escrito' o 'ERROR: [Mensaje de error]'.
            """
            print("\n[TOOL] write_text_ui invocado.")
            print(f"DEBUG: Texto a escribir: '{text_to_write}'")
            print(f"DEBUG: Target element description: '{target_element_description}'")
            try:
                if target_element_description:
                    invoke_args = {
                        'description_or_instruction': target_element_description,
                        'confidence': confidence
                    }
                    if monitor_id is not None:
                        invoke_args['monitor_id'] = monitor_id

                    print(f"DEBUG: Intentando clicar el campo de texto con search_and_click_ui_element con args: {invoke_args}")
                    click_result = search_and_click_ui_element.invoke(invoke_args)

                    if "ERROR" in click_result:
                        print(f"ERROR: No se pudo localizar el campo de texto '{target_element_description}': {click_result}")
                        return f"ERROR: No se pudo localizar el campo de texto '{target_element_description}': {click_result}"
                    time.sleep(0.5) # Pequeña pausa para asegurar el foco

                pyautogui.write(text_to_write)
                print(f"DEBUG: Texto '{text_to_write}' escrito en UI.")
                return f"SUCCESS: Texto '{text_to_write}' escrito en la UI."
            except Exception as e:
                traceback.print_exc()
                return f"ERROR: Ocurrió un error al escribir texto en la UI: {e}"
            
        @tool
        def read_plc_data(data_type: str, db_number: Optional[int] = None, byte_offset: int = 0, bit_offset: Optional[int] = None) -> Union[int, float, bool, str]:
            """
            Lee un valor del PLC (S7-1200).
            data_type: 'BOOL', 'INT', 'REAL'.
            db_number: Número del Data Block (solo para DBs).
            byte_offset: Offset del byte.
            bit_offset: Offset del bit (solo para BOOL).
            Devuelve el valor leído o un mensaje de error.
            """
            print("\n[TOOL] read_plc_data invocado.")
            print(f"DEBUG: Lectura PLC: {data_type}, DB:{db_number}, Byte:{byte_offset}, Bit:{bit_offset}")
            try:
                if not plc_handler.is_connected:
                    print("DEBUG: PLC no conectado. Intentando conectar...")
                    plc_handler.connect() # Intentar conectar si no lo está
                    if not plc_handler.is_connected:
                        raise PLCConnectionError("No se pudo conectar al PLC para lectura. Verifica la conexión y la configuración en .env.")

                value = None
                if data_type.upper() == "BOOL": # Usar .upper() para ser más robusto
                    if db_number is None or bit_offset is None:
                        return "ERROR: Para 'BOOL' se requiere 'db_number' y 'bit_offset'."
                    value = plc_handler.read_bool(db_number, byte_offset, bit_offset)
                elif data_type.upper() == "INT":
                    if db_number is None:
                        return "ERROR: Para 'INT' se requiere 'db_number'."
                    value = plc_handler.read_int(db_number, byte_offset)
                elif data_type.upper() == "REAL":
                    if db_number is None:
                        return "ERROR: Para 'REAL' se requiere 'db_number'."
                    value = plc_handler.read_real(db_number, byte_offset)
                else:
                    return f"ERROR: Tipo de dato no soportado para lectura: {data_type}. Use 'BOOL', 'INT', 'REAL'."
                
                print(f"SUCCESS: Valor leído: {value}")
                return f"SUCCESS: Valor leído de PLC ({data_type}, DB{db_number}, Byte{byte_offset}, Bit{bit_offset if bit_offset is not None else ''}): {value}"
            except (PLCConnectionError, PLCReadWriteError) as plc_err:
                traceback.print_exc()
                return f"ERROR PLC: {plc_err}"
            except Exception as e:
                traceback.print_exc()
                return f"ERROR: Ocurrió un error inesperado al leer del PLC: {e}"

        @tool
        def write_plc_data(data_type: str, value: Any, db_number: Optional[int] = None, byte_offset: int = 0, bit_offset: Optional[int] = None) -> str:
            """
            Escribe un valor en el PLC (S7-1200).
            data_type: 'BOOL', 'INT', 'REAL'.
            value: El valor a escribir.
            db_number: Número del Data Block (solo para DBs).
            byte_offset: Offset del byte.
            bit_offset: Offset del bit (solo para BOOL).
            Devuelve 'SUCCESS: Valor escrito' o 'ERROR: [Mensaje de error]'.
            """
            print("\n[TOOL] write_plc_data invocado.")
            print(f"DEBUG: Escritura PLC: {data_type}, Value:{value}, DB:{db_number}, Byte:{byte_offset}, Bit:{bit_offset}")
            try:
                if not plc_handler.is_connected:
                    print("DEBUG: PLC no conectado. Intentando conectar...")
                    plc_handler.connect()
                    if not plc_handler.is_connected:
                        raise PLCConnectionError("No se pudo conectar al PLC para escritura. Verifica la conexión y la configuración en .env.")
                        
                success = False
                if data_type.upper() == "BOOL":
                    if db_number is None or bit_offset is None:
                        return "ERROR: Para 'BOOL' se requiere 'db_number' y 'bit_offset'."
                    # Asegurarse de convertir el valor a booleano, ya que el LLM podría dar 0/1
                    success = plc_handler.write_bool(db_number, byte_offset, bit_offset, bool(value))
                elif data_type.upper() == "INT":
                    if db_number is None:
                        return "ERROR: Para 'INT' se requiere 'db_number'."
                    success = plc_handler.write_int(db_number, byte_offset, int(value))
                elif data_type.upper() == "REAL":
                    if db_number is None:
                        return "ERROR: Para 'REAL' se requiere 'db_number'."
                    success = plc_handler.write_real(db_number, byte_offset, float(value))
                else:
                    return f"ERROR: Tipo de dato no soportado para escritura: {data_type}. Use 'BOOL', 'INT', 'REAL'."
                
                if success:
                    print(f"SUCCESS: Valor '{value}' escrito en PLC.")
                    return f"SUCCESS: Valor '{value}' escrito en PLC ({data_type}, DB{db_number}, Byte{byte_offset}, Bit{bit_offset if bit_offset is not None else ''})."
                else:
                    print(f"ERROR: Falló la escritura en PLC para {data_type} con valor {value}.")
                    return f"ERROR: Falló la escritura en PLC para {data_type} con valor {value}."
            except (PLCConnectionError, PLCReadWriteError) as plc_err:
                traceback.print_exc()
                return f"ERROR PLC: {plc_err}"
            except ValueError as ve:
                traceback.print_exc()
                return f"ERROR: Valor '{value}' no es compatible con el tipo de dato '{data_type}': {ve}"
            except Exception as e:
                traceback.print_exc()
                return f"ERROR: Ocurrió un error inesperado al escribir en el PLC: {e}"

        @tool
        def get_current_time() -> str:
            """
            Obtiene la hora actual del sistema. Útil para tareas que necesitan información temporal.
            """
            print("\n[TOOL] get_current_time invocado.")
            current_time = time.strftime('%Y-%m-%d %H:%M:%S') # Formato más completo
            print(f"SUCCESS: La hora actual es: {current_time}")
            return f"SUCCESS: La hora actual es: {current_time}"

        @tool
        def take_system_screenshot(monitor_id: Optional[int] = None, save_path: str = os.path.join(AGENT_TEMP_DIR, "last_agent_screenshot.png")) -> str:
            """
            Toma una captura de pantalla del sistema (de un monitor específico o de todos)
            y la guarda en un archivo. Útil para verificar el estado de la UI o depurar.
            monitor_id (int, optional): ID del monitor a capturar (0 para el primero físico, etc.).
                                        Si es None, captura todos los monitores en una sola imagen.
            save_path (str): Ruta completa con nombre de archivo para guardar la imagen. 
                             Por defecto se guarda en el directorio temporal del agente.
            Devuelve 'SUCCESS: Captura guardada en [ruta]' o 'ERROR: [Mensaje de error]'.
            """
            print("\n[TOOL] take_system_screenshot invocado.")
            print(f"DEBUG: Tomando captura de pantalla para monitor_id: {monitor_id}, guardando en: {save_path}")
            try:
                # Asegurarse de que el directorio de destino exista
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                screenshot = take_screenshot(monitor_number=monitor_id)
                if screenshot:
                    screenshot.save(save_path)
                    absolute_path = os.path.abspath(save_path)
                    print(f"SUCCESS: Captura de pantalla guardada en {absolute_path}")
                    return f"SUCCESS: Captura de pantalla guardada en {absolute_path}"
                else:
                    return f"ERROR: No se pudo tomar la captura de pantalla para el monitor {monitor_id if monitor_id is not None else 'principal'}."
            except Exception as e:
                print(f"ERROR: Falló la captura de pantalla en take_system_screenshot: {e}")
                traceback.print_exc()
                return f"ERROR: Falló la captura de pantalla: {e}"

        @tool
        def perform_ocr_on_screen(description_or_instruction: Optional[str] = None, x: Optional[int] = None, y: Optional[int] = None, width: Optional[int] = None, height: Optional[int] = None, monitor_id: Optional[int] = None, confidence: float = 0.8) -> str:
            """
            Realiza OCR (Reconocimiento Óptico de Caracteres) en una región específica de la pantalla
            o en una región identificada por un elemento de UI predefinido.
            Puedes especificar la región usando:
            1. 'description_or_instruction': Una descripción clara del elemento de UI (como "campo de valor numérico", "etiqueta de estado")
               cuyo recorte ha sido previamente ingresado en Qdrant. La herramienta localizará este recorte y realizará OCR en él.
            2. Coordenadas manuales: 'x', 'y', 'width', 'height' para definir la región en píxeles.
               Si se usan coordenadas manuales, 'description_or_instruction' debe ser None.
            
            monitor_id (int, optional): ID del monitor desde el que tomar la captura (0 para el principal).
            confidence (float): Umbral de confianza para la detección de imagen si se usa 'description_or_instruction'.
            
            Devuelve 'SUCCESS: Texto extraído: [texto]' o 'ERROR: [Mensaje de error]'.
            """
            print("\n[TOOL] perform_ocr_on_screen invocado.")
            print(f"DEBUG: OCR params: desc='{description_or_instruction}', coords=({x},{y},{width},{height}), monitor={monitor_id}, conf={confidence}")

            try:
                # Tomar una captura de pantalla del monitor especificado
                screenshot_image = take_screenshot(monitor_number=monitor_id)
                if screenshot_image is None:
                    return f"ERROR: No se pudo tomar una captura de pantalla del monitor {monitor_id if monitor_id is not None else 'principal'} para OCR."

                region_to_ocr = None # Variable para almacenar la imagen de la región a procesar
                
                if description_or_instruction:
                    # Si se proporciona una descripción, usar search_and_click para encontrar la región
                    query_embedding = image_processor.generate_embedding_from_text(description_or_instruction)
                    search_results = qdrant_handler.search_points(query_embedding, limit=1)

                    if not search_results:
                        return f"ERROR: No se encontraron recortes relevantes en Qdrant para: '{description_or_instruction}'. No se puede realizar OCR."
                    
                    best_match = search_results[0]
                    clipping_file_path = best_match.payload.get("image_path")

                    if not clipping_file_path or not os.path.exists(clipping_file_path):
                        return f"ERROR: La ruta de imagen del recorte no es válida o no existe: {clipping_file_path}. No se puede realizar OCR."
                    
                    # Guardar la captura de pantalla temporalmente para find_image_on_screen
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=AGENT_TEMP_DIR) as temp_screenshot_file:
                        screenshot_image.save(temp_screenshot_file.name)
                        temp_screenshot_path = temp_screenshot_file.name
                    
                    location = find_image_on_screen(clipping_file_path, temp_screenshot_path, confidence=confidence)
                    
                    if os.path.exists(temp_screenshot_path):
                        os.remove(temp_screenshot_path)
                        print(f"DEBUG: Archivo temporal de captura '{temp_screenshot_path}' eliminado.")

                    if location:
                        # Recortar la región de interés de la captura de pantalla original
                        region_to_ocr = screenshot_image.crop((location.left, location.top, location.right, location.bottom))
                        print(f"DEBUG: Región para OCR identificada por descripción: {location}")
                    else:
                        print(f"WARNING: No se pudo localizar '{description_or_instruction}' en la pantalla con confianza {confidence}.")
                        return f"ERROR: No se pudo localizar '{description_or_instruction}' en la pantalla actual para OCR. Intenta ajustar la descripción o la confianza."
                elif x is not None and y is not None and width is not None and height is not None:
                    # Recortar la región de interés usando las coordenadas proporcionadas
                    # Asegurar que las coordenadas estén dentro de los límites de la imagen
                    img_width, img_height = screenshot_image.size
                    right = min(x + width, img_width)
                    bottom = min(y + height, img_height)
                    
                    if x < 0 or y < 0 or right <= x or bottom <= y:
                        return f"ERROR: Coordenadas de la región de OCR ({x},{y},{width},{height}) están fuera de los límites de la pantalla o son inválidas."
                    
                    region_to_ocr = screenshot_image.crop((x, y, right, bottom))
                    print(f"DEBUG: Región para OCR identificada por coordenadas: ({x},{y},{width},{height})")
                else:
                    return "ERROR: Debes proporcionar 'description_or_instruction' o las coordenadas (x, y, width, height) para realizar OCR."
                
                if region_to_ocr:
                    # Generar un nombre de archivo temporal para la región recortada para OCR
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=AGENT_TEMP_DIR) as temp_ocr_file:
                        region_to_ocr.save(temp_ocr_file.name)
                        temp_ocr_path = temp_ocr_file.name
                    
                    extracted_text = image_processor.extract_text_from_image(temp_ocr_path)
                    
                    if os.path.exists(temp_ocr_path):
                        os.remove(temp_ocr_path)
                        print(f"DEBUG: Archivo temporal de OCR '{temp_ocr_path}' eliminado.")

                    print(f"SUCCESS: Texto extraído: '{extracted_text}'")
                    return f"SUCCESS: Texto extraído: '{extracted_text}'"
                else:
                    return "ERROR: No se pudo definir la región para realizar OCR."

            except Exception as e:
                traceback.print_exc()
                return f"ERROR: Ocurrió un error al realizar OCR: {e}"

        # Devuelve la lista de todas las herramientas definidas
        return [
            search_and_click_ui_element,
            write_text_ui,
            read_plc_data,
            write_plc_data,
            get_current_time,
            take_system_screenshot,
            perform_ocr_on_screen # AÑADIDO: La nueva herramienta de OCR
        ]

    def run_task(self, instruction: str) -> str:
        """
        Ejecuta una tarea compleja dada una instrucción en lenguaje natural.
        Args:
            instruction (str): La instrucción del usuario (ej. "Navegar a TIA Portal y arrancar mezclador").
        Returns:
            str: El resultado de la ejecución de la tarea.
        """
        print(f"\n[AGENT] Recibida instrucción para run_task: '{instruction}'")
        try:
            result = self.agent_executor.invoke(
                {"input": instruction, "chat_history": self.chat_history}
            )
            
            # El output del agente puede estar en result["output"] o result["agent_outcome"].
            # Para AgentExecutor con create_tool_calling_agent, normalmente el resultado final está en "output".
            final_output = result.get("output", "No se encontró salida final del agente.")
            
            # Opcional: Actualizar el historial de chat si quieres mantener el contexto entre invocaciones
            # self.chat_history.append(HumanMessage(content=instruction))
            # self.chat_history.append(AIMessage(content=final_output)) # Asegúrate de que final_output sea una cadena
            
            print(f"\n[AGENT] Tarea completada. Output del Agente: {final_output}")
            return final_output
        except Exception as e:
            print(f"ERROR: Falló la ejecución de la tarea compleja: {e}")
            traceback.print_exc() # Imprime el stack trace completo
            return f"ERROR: Falló la ejecución de la tarea compleja: {e}"

# Ejemplo de uso (para pruebas directas de este módulo)
if __name__ == "__main__":
    print("--- Probando AutomationAgent (automation_agent.py) ---")
    agent = AutomationAgent()

    # Asegurarse de que el directorio temporal del agente exista para las pruebas
    os.makedirs(AGENT_TEMP_DIR, exist_ok=True)

    print("\n--- Prueba 1: Tarea de UI simple (abrir papelera de reciclaje) ---")
    # Asegúrate de que un recorte de "icono de papelera de reciclaje" esté ingresado en Qdrant
    ui_instruction = "Por favor, abre la papelera de reciclaje."
    print(f"\nInstrucción: {ui_instruction}")
    response = agent.run_task(ui_instruction)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(2) # Pausa para observar la acción

    print("\n--- Prueba 2: Escribir texto en una barra de búsqueda de Windows ---")
    # Asegúrate de que un recorte de "barra de búsqueda de Windows" (o similar) esté ingresado.
    # Esta prueba intentará hacer clic primero y luego escribir.
    write_instruction = "Escribe 'bloc de notas' en la barra de búsqueda de Windows y presiona Enter."
    print(f"\nInstrucción: {write_instruction}")
    response = agent.run_task(write_instruction)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(2) # Pausa para observar la acción
    
    # Después de escribir, podrías querer presionar Enter si la instrucción lo implica
    # Si el agente no lo hace automáticamente, podrías añadir una herramienta para 'press_key'
    pyautogui.press('enter')
    print("DEBUG: Se presionó Enter después de escribir texto.")
    time.sleep(1)


    print("\n--- Prueba 3: Leer la hora actual del sistema ---")
    time_instruction = "Dime la hora actual del sistema."
    print(f"\nInstrucción: {time_instruction}")
    response = agent.run_task(time_instruction)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(1)

    print("\n--- Prueba 4: Tomar una captura de pantalla ---")
    screenshot_instruction = "Toma una captura de pantalla de mi monitor principal y guárdala."
    print(f"\nInstrucción: {screenshot_instruction}")
    response = agent.run_task(screenshot_instruction)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(2) # Pausa para que el archivo se guarde

    print("\n--- Prueba 5: Interacción con PLC (lectura de BOOL - asumiendo conexión) ---")
    # Asegúrate de que PLC_IP_ADDRESS, PLC_RACK, PLC_SLOT estén configurados en .env y el PLC esté accesible.
    # Este ejemplo asume que tienes un DB1 con un BOOL en DB1.DBX0.0
    plc_read_instruction = "Lee el estado del bit DB1.DBX0.0 en el PLC."
    print(f"\nInstrucción: {plc_read_instruction}")
    response = agent.run_task(plc_read_instruction)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(1)

    print("\n--- Prueba 6: Interacción con PLC (escritura de BOOL - asumiendo conexión) ---")
    # Este ejemplo asume que puedes escribir en DB1.DBX0.0
    plc_write_instruction = "Establece el bit DB1.DBX0.0 a TRUE en el PLC."
    print(f"\nInstrucción: {plc_write_instruction}")
    response = agent.run_task(plc_write_instruction)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(1)
    
    print("\n--- Prueba 7: Realizar OCR en una región de la pantalla (usando coordenadas) ---")
    # ESTA ES UNA PRUEBA EJEMPLO. NECESITARÁS AJUSTAR LAS COORDENADAS (x, y, width, height)
    # PARA UNA REGIÓN DONDE ESPERES ENCONTRAR TEXTO EN TU PANTALLA ACTUAL.
    # Por ejemplo, si tienes una ventana con "Hola Mundo" en (100, 100) y de 200x50 píxeles.
    ocr_instruction_coords = "Realiza OCR en la región de la pantalla que abarca desde la coordenada X 100, Y 100 con un ancho de 200 píxeles y un alto de 50 píxeles."
    print(f"\nInstrucción: {ocr_instruction_coords}")
    # Nota: El agente interpretará esto y llamará a la herramienta con los parámetros adecuados.
    # La herramienta `perform_ocr_on_screen` recibirá los parámetros `x=100, y=100, width=200, height=50`.
    response = agent.run_task(ocr_instruction_coords)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(1)

    print("\n--- Prueba 8: Realizar OCR en un elemento de UI conocido (usando descripción) ---")
    # Para esta prueba, NECESITAS HABER INGRESADO PREVIAMENTE UN RECORTE
    # DE UN ELEMENTO DE UI QUE CONTENGA TEXTO EN QDRANT.
    # Por ejemplo, podrías tener un recorte de un "campo de valor de temperatura"
    # y que la descripción en Qdrant sea "campo de valor de temperatura".
    ocr_instruction_desc = "Lee el texto del 'campo de valor de temperatura' en la pantalla."
    print(f"\nInstrucción: {ocr_instruction_desc}")
    # Nota: El agente buscará el recorte "campo de valor de temperatura" y le aplicará OCR.
    response = agent.run_task(ocr_instruction_desc)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(1)

    print("\n--- Fin de las pruebas del AutomationAgent ---")
    # Asegurarse de desconectar del PLC al finalizar las pruebas
    plc_handler.disconnect() 

    # Opcional: Limpiar el directorio temporal del agente al finalizar todas las pruebas
    try:
        if os.path.exists(AGENT_TEMP_DIR):
            for f in os.listdir(AGENT_TEMP_DIR):
                os.remove(os.path.join(AGENT_TEMP_DIR, f))
            os.rmdir(AGENT_TEMP_DIR)
            print(f"Directorio temporal '{AGENT_TEMP_DIR}' vaciado y eliminado.")
    except Exception as e:
        print(f"Error al limpiar el directorio temporal del agente: {e}")