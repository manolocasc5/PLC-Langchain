# tu_proyecto/automation_agent.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI # Usaremos el cliente de OpenAI de LangChain
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import traceback
from typing import List, Dict, Any, Tuple, Optional, Union
import time
import pyautogui

# Importar las clases y funciones necesarias de nuestros módulos
from qdrant_handler import QdrantHandler
from image_processor import ImageProcessor
from utils.screen_utils import take_screenshot, find_image_on_screen, get_monitor_info
from plc_handler import PLCHandler, PLCConnectionError, PLCReadWriteError

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY debe estar configurado en el archivo .env")

# Inicializar los handlers necesarios
qdrant_handler = QdrantHandler()
image_processor = ImageProcessor()
plc_handler = PLCHandler() # Inicializamos el handler del PLC

class AutomationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0) # Usamos gpt-4o por su capacidad de razonamiento y Tool Calling
        
        # Definir las herramientas que el agente de LangChain puede usar
        self.tools = self._define_tools()
        
        # Definir el prompt del agente
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                 "Eres un asistente experto en automatización de interfaces de usuario y control de PLCs Siemens S7-1200. "
                 "Tu objetivo es ayudar al usuario a realizar tareas complejas en Windows y en el PLC. "
                 "Utiliza las herramientas disponibles para lograr los objetivos. "
                 "Prioriza siempre el uso de las herramientas para interacciones con la UI y el PLC. "
                 "Si el usuario pide una tarea compleja, desglósala en pasos y utiliza las herramientas para cada paso. "
                 "Sé muy preciso con las descripciones que usas para buscar elementos de UI."
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
        """

        @tool
        def search_and_click_ui_element(description_or_instruction: str, monitor_id: int = None, confidence: float = 0.5) -> str:
            """
            Busca un elemento de UI en pantalla basado en su descripción y hace clic en él.
            Útil para navegar por ventanas, abrir aplicaciones o interactuar con botones/iconos.
            La 'description_or_instruction' debe ser una descripción clara del elemento a buscar,
            ej: 'icono de la papelera de reciclaje', 'botón Aceptar', 'pestaña Configuración'.
            Devuelve 'SUCCESS: Clic ejecutado en [Descripción]' o 'ERROR: [Mensaje de error]'.
            """
            print(f"\n[TOOL] Recibida instrucción UI: {description_or_instruction}")
            try:
                # 1. Generar embedding de la instrucción
                query_embedding = image_processor.generate_embedding_from_text(description_or_instruction)
                
                if len(query_embedding) != qdrant_handler.VECTOR_DIMENSION:
                    return f"ERROR: El embedding de la instrucción no tiene la dimensión esperada ({len(query_embedding)} vs {qdrant_handler.VECTOR_DIMENSION})."
                
                # 2. Buscar recortes relevantes en Qdrant
                search_results = qdrant_handler.search_points(query_embedding, limit=1)
                
                if not search_results:
                    return f"ERROR: No se encontraron recortes relevantes en Qdrant para: '{description_or_instruction}'. Asegúrate de que el recorte esté ingresado."
                
                best_match = search_results[0]
                clipping_file_path = best_match.payload.get("image_path")
                
                if not clipping_file_path or not os.path.exists(clipping_file_path):
                    return f"ERROR: La ruta de imagen del recorte no es válida o no existe: {clipping_file_path}"
                
                # 3. Tomar captura de pantalla actual
                current_screenshot_image = take_screenshot(monitor_number=monitor_id)
                
                # 4. Localizar el recorte en la captura de pantalla
                location = find_image_on_screen(clipping_file_path, current_screenshot_image, confidence=confidence)

                if location:
                    center_x = location.left + location.width / 2
                    center_y = location.top + location.height / 2
                    pyautogui.click(center_x, center_y)
                    pyautogui.click()
                    print(f"DEBUG: Clic ejecutado en ({center_x}, {center_y}) para: {description_or_instruction}")
                    return f"SUCCESS: Clic ejecutado en '{description_or_instruction}'."
                else:
                    return f"ERROR: No se pudo localizar '{description_or_instruction}' en la pantalla actual con confianza {confidence}. Intenta ajustar la descripción o la confianza."
            except Exception as e:
                return f"ERROR: Ocurrió un error en search_and_click_ui_element: {e}"

        @tool
        def write_text_ui(text_to_write: str, target_element_description: str = None, monitor_id: int = None, confidence: float = 0.8) -> str:
            """
            Escribe texto en un campo de UI. Si se proporciona 'target_element_description',
            intentará localizar y clicar el campo antes de escribir. Si no, escribirá donde esté el foco actual.
            'target_element_description': ej. 'campo de texto Nombre de Usuario', 'barra de búsqueda'.
            Devuelve 'SUCCESS: Texto escrito' o 'ERROR: [Mensaje de error]'.
            """ # <-- ¡Este docstring es el que faltaba!
            print(f"\n[TOOL] Recibida instrucción de escritura UI: '{text_to_write}'")
            try:
                if target_element_description:
                    # Construir los argumentos para invoke dinámicamente
                    invoke_args = {
                        'description_or_instruction': target_element_description,
                        'confidence': confidence
                    }
                    if monitor_id is not None: # Solo añade monitor_id si no es None
                        invoke_args['monitor_id'] = monitor_id

                    # Cambio importante aquí: Usar .invoke() con el diccionario de argumentos
                    click_result = search_and_click_ui_element.invoke(invoke_args)

                    if "ERROR" in click_result:
                        return f"ERROR: No se pudo localizar el campo de texto '{target_element_description}': {click_result}"
                    time.sleep(0.5) # Pequeña pausa para asegurar el foco

                pyautogui.write(text_to_write)
                print(f"DEBUG: Texto '{text_to_write}' escrito en UI.")
                return f"SUCCESS: Texto '{text_to_write}' escrito en la UI."
            except Exception as e:
                # Aquí es útil un traceback también para depuración de este error
                import traceback
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
            print(f"\n[TOOL] Recibida instrucción de lectura PLC: {data_type}, DB:{db_number}, Byte:{byte_offset}, Bit:{bit_offset}")
            try:
                if not plc_handler.is_connected:
                    plc_handler.connect() # Intentar conectar si no lo está
                    if not plc_handler.is_connected:
                        raise PLCConnectionError("No se pudo conectar al PLC para lectura.")

                value = None
                if data_type == "BOOL":
                    if db_number is None or bit_offset is None:
                        return "ERROR: Para BOOL se requiere db_number y bit_offset."
                    value = plc_handler.read_bool(db_number, byte_offset, bit_offset)
                elif data_type == "INT":
                    if db_number is None:
                        return "ERROR: Para INT se requiere db_number."
                    value = plc_handler.read_int(db_number, byte_offset)
                elif data_type == "REAL":
                    if db_number is None:
                        return "ERROR: Para REAL se requiere db_number."
                    value = plc_handler.read_real(db_number, byte_offset)
                else:
                    return f"ERROR: Tipo de dato no soportado para lectura: {data_type}. Use 'BOOL', 'INT', 'REAL'."
                
                return f"SUCCESS: Valor leído de PLC ({data_type}, DB{db_number}, Byte{byte_offset}, Bit{bit_offset if bit_offset is not None else ''}): {value}"
            except (PLCConnectionError, PLCReadWriteError) as plc_err:
                return f"ERROR PLC: {plc_err}"
            except Exception as e:
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
            print(f"\n[TOOL] Recibida instrucción de escritura PLC: {data_type}, Value:{value}, DB:{db_number}, Byte:{byte_offset}, Bit:{bit_offset}")
            try:
                if not plc_handler.is_connected:
                    plc_handler.connect() # Intentar conectar si no lo está
                    if not plc_handler.is_connected:
                        raise PLCConnectionError("No se pudo conectar al PLC para escritura.")
                        
                success = False
                if data_type == "BOOL":
                    if db_number is None or bit_offset is None:
                        return "ERROR: Para BOOL se requiere db_number y bit_offset."
                    success = plc_handler.write_bool(db_number, byte_offset, bit_offset, bool(value))
                elif data_type == "INT":
                    if db_number is None:
                        return "ERROR: Para INT se requiere db_number."
                    success = plc_handler.write_int(db_number, byte_offset, int(value))
                elif data_type == "REAL":
                    if db_number is None:
                        return "ERROR: Para REAL se requiere db_number."
                    success = plc_handler.write_real(db_number, byte_offset, float(value))
                else:
                    return f"ERROR: Tipo de dato no soportado para escritura: {data_type}. Use 'BOOL', 'INT', 'REAL'."
                
                if success:
                    return f"SUCCESS: Valor '{value}' escrito en PLC ({data_type}, DB{db_number}, Byte{byte_offset}, Bit{bit_offset if bit_offset is not None else ''})."
                else:
                    return f"ERROR: Falló la escritura en PLC para {data_type} con valor {value}."
            except (PLCConnectionError, PLCReadWriteError) as plc_err:
                return f"ERROR PLC: {plc_err}"
            except ValueError as ve:
                return f"ERROR: Valor '{value}' no es compatible con el tipo de dato '{data_type}': {ve}"
            except Exception as e:
                return f"ERROR: Ocurrió un error inesperado al escribir en el PLC: {e}"

        @tool
        def get_current_time() -> str:
            """
            Obtiene la hora actual del sistema. Útil para tareas que necesitan información temporal.
            """
            return f"SUCCESS: La hora actual es: {time.strftime('%H:%M:%S')}"

        @tool
        def take_system_screenshot(monitor_id: int = None, save_path: str = "last_screenshot.png") -> str:
            """
            Toma una captura de pantalla del sistema (de un monitor específico o de todos)
            y la guarda en un archivo. Útil para verificar el estado de la UI o depurar.
            monitor_id (int, optional): ID del monitor a capturar (0 para el primero físico, etc.).
                                        Si es None, captura todos.
            save_path (str): Ruta para guardar la imagen. Por defecto 'last_screenshot.png'.
            Devuelve 'SUCCESS: Captura guardada en [ruta]' o 'ERROR: [Mensaje de error]'.
            """
            print("\n[TOOL] Recibida instrucción para captura de pantalla.")
            try:
                screenshot = take_screenshot(monitor_number=monitor_id)
                screenshot.save(save_path)
                return f"SUCCESS: Captura de pantalla guardada en {os.path.abspath(save_path)}"
            except Exception as e:
                print(f"ERROR: Falló la captura de pantalla en take_system_screenshot: {e}") # <--- Mensaje más descriptivo
                traceback.print_exc()
                return f"ERROR: Falló la captura de pantalla: {e}"

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
        try:
            result = self.agent_executor.invoke(
                {"input": instruction, "chat_history": self.chat_history}
            )
            # Opcional: Actualizar el historial de chat si quieres mantener el contexto entre invocaciones
            # self.chat_history.append(HumanMessage(content=instruction))
            # self.chat_history.append(AIMessage(content=result["output"]))
            return result["output"]
        except Exception as e:
            print(f"ERROR: Ocurrió un error en search_and_click_ui_element: {e}") # <--- Mensaje más descriptivo
            traceback.print_exc()
            return f"ERROR: Falló la ejecución de la tarea compleja: {e}"

# Ejemplo de uso (para pruebas directas de este módulo)
if __name__ == "__main__":
    print("--- Probando AutomationAgent (automation_agent.py) ---")
    agent = AutomationAgent()

    print("\n--- Prueba 1: Tarea de UI simple (requiere recorte de papelera) ---")
    # Asegúrate de que "papelera_reciclaje.png" esté ingresado en Qdrant
    # y que la Papelera de Reciclaje sea visible en tu escritorio.
    # Puedes ajustar el monitor_id si usas múltiples monitores.
    ui_instruction = "Por favor, abre la papelera de reciclaje en el escritorio."
    print(f"\nInstrucción: {ui_instruction}")
    response = agent.run_task(ui_instruction)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(2)

    print("\n--- Prueba 2: Tarea de PLC de escritura (simulada si no hay PLC real) ---")
    # Esto asume que el agente decidirá usar la herramienta write_plc_data.
    # El LLM interpretará 'poner a TRUE el bit 0 del byte 0 del DB1' en los parámetros de la herramienta.
    plc_instruction_write = "Pon a TRUE el bit 0 del byte 0 del Data Block 1 (DB1.DBX0.0) para arrancar la secuencia."
    print(f"\nInstrucción: {plc_instruction_write}")
    response = agent.run_task(plc_instruction_write)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(2)

    print("\n--- Prueba 3: Tarea de PLC de lectura (simulada si no hay PLC real) ---")
    plc_instruction_read = "Cuál es el valor actual del entero en el byte 10 del Data Block 1 (DB1.DBW10)?"
    print(f"\nInstrucción: {plc_instruction_read}")
    response = agent.run_task(plc_instruction_read)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(2)

    print("\n--- Prueba 4: Tarea de UI de escritura (simulada si no hay campo de texto) ---")
    # El agente intentará buscar un campo de texto y escribir en él.
    # Necesitarías un recorte de un campo de texto para que esto funcione realmente.
    # Por ejemplo, un campo de búsqueda en el explorador de archivos.
    # Asegúrate de que "campo_busqueda.png" esté ingresado en Qdrant
    # ui_write_instruction = "En la barra de búsqueda de la ventana actual, escribe 'documentos importantes'."
    # print(f"\nInstrucción: {ui_write_instruction}")
    # response = agent.run_task(ui_write_instruction)
    # print(f"\nRespuesta del Agente: {response}")
    # time.sleep(2)

    print("\n--- Prueba 5: Pregunta general al agente ---")
    general_instruction = "¿Qué hora es y luego toma una captura de pantalla de todos los monitores?"
    print(f"\nInstrucción: {general_instruction}")
    response = agent.run_task(general_instruction)
    print(f"\nRespuesta del Agente: {response}")
    time.sleep(2)

    print("\n--- Fin de las pruebas del AutomationAgent ---")
    plc_handler.disconnect() # Asegurarse de desconectar al finalizar