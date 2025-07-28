# tu_proyecto/automation_agent.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool # Importa el decorador @tool
import traceback
from typing import List, Dict, Any, Tuple, Optional, Union
import time
import pyautogui

# Importar las clases y funciones necesarias de nuestros módulos
from qdrant_handler import QdrantHandler
from image_processor import ImageProcessor
from utils.screen_utils import take_screenshot, find_image_on_screen # Asumo que take_screenshot devuelve PIL.Image y find_image_on_screen puede tomar PIL.Image
# from utils.screen_utils import get_monitor_info # Si no la usas, puedes comentarla
from plc_handler import PLCHandler, PLCConnectionError, PLCReadWriteError

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY debe estar configurado en el archivo .env")

# Inicializar los handlers necesarios
# Asegúrate de que estas instancias sean únicas y se compartan.
# Las hacemos globales para que las herramientas decoradas @tool puedan acceder a ellas
qdrant_handler = QdrantHandler()
image_processor = ImageProcessor()
plc_handler = PLCHandler()

class AutomationAgent:
    def __init__(self):
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
                "**Para tareas complejas que involucren una secuencia de pasos en la UI (como abrir programas, navegar por menús, o escribir texto en campos), desglosa la tarea en pasos individuales y llama a las herramientas `search_and_click_ui_element` y `write_text_ui` para cada paso.** "
                "Por ejemplo, si el usuario dice 'abre el TIA Portal y crea un nuevo proyecto', tu primer paso debería ser `search_and_click_ui_element(description_or_instruction='icono de TIA Portal V15')` y luego continuar con los pasos para crear el proyecto. "
                "Cuando el usuario te pida escribir código en un IDE, primero deberás usar las herramientas de UI para navegar hasta ese IDE y localizar el área donde se escribe el código. Una vez localizado, utiliza la herramienta `write_text_ui` para insertar el código proporcionado por ti o por el usuario. "
                "Para la interacción con el PLC, puedes leer y escribir en Data Blocks (DBs) o en memoria Merker (M), y con tipos específicos (BOOL, INT, REAL)."
                "Cuando interactúes con el PLC, el usuario puede darte comandos estructurados (ej. 'WRITE DB1.DBW10 INT 123') o lenguaje natural (ej. 'Arranca la secuencia de mezclado'). "
                "Si es lenguaje natural para el PLC, tradúcelo a operaciones de lectura/escritura de bits/bytes/palabras en el PLC y usa la herramienta adecuada."
                "Por ejemplo, 'Arranca la secuencia de mezclado' podría significar poner a TRUE el bit 0 del byte 0 del DB1 (DB1.DBX0.0)."
                "Muestra el estado actual del PLC cuando se te pida o después de una operación de escritura relevante."
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
            # --- DEBUGGING CRÍTICO ---
            print(f"\n[TOOL] search_and_click_ui_element invocado.")
            print(f"DEBUG: description_or_instruction recibida: '{description_or_instruction}'")
            print(f"DEBUG: monitor_id: {monitor_id}, confidence: {confidence}")
            # --- FIN DEBUGGING CRÍTICO ---

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
                    return f"ERROR: No se encontraron recortes relevantes en Qdrant para: '{description_or_instruction}'. Asegúrate de que el recorte esté ingresado."
                
                best_match = search_results[0]
                clipping_file_path = best_match.payload.get("image_path")
                
                print(f"DEBUG: Mejor coincidencia en Qdrant (score: {best_match.score}): {best_match.payload.get('description')} (Path: {clipping_file_path})")

                if not clipping_file_path or not os.path.exists(clipping_file_path):
                    print(f"ERROR: La ruta de imagen del recorte no es válida o no existe: {clipping_file_path}")
                    return f"ERROR: La ruta de imagen del recorte no es válida o no existe: {clipping_file_path}"
                
                # 3. Tomar captura de pantalla actual
                print(f"DEBUG: Tomando captura de pantalla del monitor {monitor_id if monitor_id is not None else 'principal'}.")
                current_screenshot_image = take_screenshot(monitor_number=monitor_id)
                
                # 4. Localizar el recorte en la captura de pantalla
                # NOTA IMPORTANTE: Asegúrate que find_image_on_screen pueda tomar una PIL.Image para 'screenshot_image'
                print(f"DEBUG: Buscando '{clipping_file_path}' en la captura de pantalla con confianza {confidence}...")
                location = find_image_on_screen(clipping_file_path, current_screenshot_image, confidence=confidence)

                if location:
                    center_x = location.left + location.width / 2
                    center_y = location.top + location.height / 2
                    pyautogui.doubleClick(center_x, center_y)
                    # pyautogui.click() # A menudo un solo clic es suficiente, el doble clic puede causar aperturas dobles
                    print(f"DEBUG: Clic ejecutado en ({center_x}, {center_y}) para: '{description_or_instruction}'")
                    return f"SUCCESS: Clic ejecutado en '{description_or_instruction}'."
                else:
                    print(f"WARNING: No se pudo localizar '{description_or_instruction}' en la pantalla actual con confianza {confidence}.")
                    return f"ERROR: No se pudo localizar '{description_or_instruction}' en la pantalla actual con confianza {confidence}. Intenta ajustar la descripción o la confianza."
            except Exception as e:
                # Se imprime el traceback para tener más detalles del error
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
            print(f"\n[TOOL] write_text_ui invocado.")
            print(f"DEBUG: Texto a escribir: '{text_to_write}'")
            print(f"DEBUG: Target element description: '{target_element_description}'")
            try:
                if target_element_description:
                    # **CORRECCIÓN CLAVE:** Pasar argumentos a .invoke() como un diccionario
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
            print(f"\n[TOOL] read_plc_data invocado.")
            print(f"DEBUG: Lectura PLC: {data_type}, DB:{db_number}, Byte:{byte_offset}, Bit:{bit_offset}")
            try:
                if not plc_handler.is_connected:
                    print("DEBUG: PLC no conectado. Intentando conectar...")
                    plc_handler.connect() # Intentar conectar si no lo está
                    if not plc_handler.is_connected:
                        raise PLCConnectionError("No se pudo conectar al PLC para lectura.")

                value = None
                if data_type.upper() == "BOOL": # Usar .upper() para ser más robusto
                    if db_number is None or bit_offset is None:
                        return "ERROR: Para BOOL se requiere db_number y bit_offset."
                    value = plc_handler.read_bool(db_number, byte_offset, bit_offset)
                elif data_type.upper() == "INT":
                    if db_number is None:
                        return "ERROR: Para INT se requiere db_number."
                    value = plc_handler.read_int(db_number, byte_offset)
                elif data_type.upper() == "REAL":
                    if db_number is None:
                        return "ERROR: Para REAL se requiere db_number."
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
            print(f"\n[TOOL] write_plc_data invocado.")
            print(f"DEBUG: Escritura PLC: {data_type}, Value:{value}, DB:{db_number}, Byte:{byte_offset}, Bit:{bit_offset}")
            try:
                if not plc_handler.is_connected:
                    print("DEBUG: PLC no conectado. Intentando conectar...")
                    plc_handler.connect()
                    if not plc_handler.is_connected:
                        raise PLCConnectionError("No se pudo conectar al PLC para escritura.")
                        
                success = False
                if data_type.upper() == "BOOL":
                    if db_number is None or bit_offset is None:
                        return "ERROR: Para BOOL se requiere db_number y bit_offset."
                    success = plc_handler.write_bool(db_number, byte_offset, bit_offset, bool(value))
                elif data_type.upper() == "INT":
                    if db_number is None:
                        return "ERROR: Para INT se requiere db_number."
                    success = plc_handler.write_int(db_number, byte_offset, int(value))
                elif data_type.upper() == "REAL":
                    if db_number is None:
                        return "ERROR: Para REAL se requiere db_number."
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
            current_time = time.strftime('%H:%M:%S')
            print(f"SUCCESS: La hora actual es: {current_time}")
            return f"SUCCESS: La hora actual es: {current_time}"

        @tool
        def take_system_screenshot(monitor_id: Optional[int] = None, save_path: str = "last_screenshot.png") -> str:
            """
            Toma una captura de pantalla del sistema (de un monitor específico o de todos)
            y la guarda en un archivo. Útil para verificar el estado de la UI o depurar.
            monitor_id (int, optional): ID del monitor a capturar (0 para el primero físico, etc.).
                                        Si es None, captura todos.
            save_path (str): Ruta para guardar la imagen. Por defecto 'last_screenshot.png'.
            Devuelve 'SUCCESS: Captura guardada en [ruta]' o 'ERROR: [Mensaje de error]'.
            """
            print("\n[TOOL] take_system_screenshot invocado.")
            print(f"DEBUG: Tomando captura de pantalla para monitor_id: {monitor_id}, guardando en: {save_path}")
            try:
                screenshot = take_screenshot(monitor_number=monitor_id)
                screenshot.save(save_path)
                absolute_path = os.path.abspath(save_path)
                print(f"SUCCESS: Captura de pantalla guardada en {absolute_path}")
                return f"SUCCESS: Captura de pantalla guardada en {absolute_path}"
            except Exception as e:
                print(f"ERROR: Falló la captura de pantalla en take_system_screenshot: {e}")
                traceback.print_exc()
                return f"ERROR: Falló la captura de pantalla: {e}"

        # Devuelve la lista de todas las herramientas definidas
        return [
            search_and_click_ui_element,
            write_text_ui,
            read_plc_data,
            write_plc_data,
            get_current_time,
            take_system_screenshot
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
            # Opcional: Actualizar el historial de chat si quieres mantener el contexto entre invocaciones
            # self.chat_history.append(HumanMessage(content=instruction))
            # self.chat_history.append(AIMessage(content=result["output"]))
            
            # El output del agente puede estar en result["output"] o result["agent_outcome"].
            # Para AgentExecutor con create_tool_calling_agent, normalmente el resultado final está en "output".
            final_output = result.get("output", "No se encontró salida final del agente.")
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

    print("\n--- Prueba 1: Tarea de UI simple (abrir carpeta prueba) ---")
    # Asegúrate de que un recorte de "carpeta prueba" esté ingresado en Qdrant
    # y que la carpeta sea visible en tu escritorio.
    ui_instruction = "Por favor, abre la carpeta llamada 'prueba' en el escritorio."
    print(f"\nInstrucción: {ui_instruction}")
    response = agent.run_task(ui_instruction)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(2)

    # ... (otras pruebas, las he dejado como estaban) ...

    print("\n--- Fin de las pruebas del AutomationAgent ---")
    plc_handler.disconnect() # Asegurarse de desconectar al finalizar