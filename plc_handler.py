import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Union
import time # Añadir import de time para las pausas en el ejemplo

# Importar la librería Snap7. Si aún no la tienes instalada:
# pip install python-snap7

S7Client = None
Snap7Exception = Exception # Default a Exception si no se puede importar Snap7Exception
try:
    import snap7.client as S7Client
    import snap7.util as S7Util
    from snap7.snap7exceptions import Snap7Exception as ActualSnap7Exception
    Snap7Exception = ActualSnap7Exception # Asignar la excepción real si se importa correctamente
    print("python-snap7 importado correctamente.")
except ImportError:
    print("Advertencia: python-snap7 no está instalado. Las funciones del PLC estarán simuladas.")
except Exception as e:
    print(f"Advertencia: Error al importar snap7 o Snap7Exception: {e}. Las funciones del PLC estarán simuladas.")
    # Asegurarse de que S7Client permanezca None si hay un problema al importar cualquier parte de snap7
    S7Client = None


load_dotenv()

# Variables de entorno para la conexión al PLC
PLC_IP_ADDRESS = os.getenv("PLC_IP_ADDRESS")
PLC_RACK = int(os.getenv("PLC_RACK", 0)) # Rack suele ser 0 para S7-1200/1500
PLC_SLOT = int(os.getenv("PLC_SLOT", 1)) # Slot suele ser 1 para S7-1200/1500

class PLCConnectionError(Exception):
    """Excepción personalizada para errores de conexión al PLC."""
    pass

class PLCReadWriteError(Exception):
    """Excepción personalizada para errores de lectura/escritura al PLC."""
    pass

class PLCHandler:
    def __init__(self):
        self.client = None
        self.is_connected = False
        
        if not PLC_IP_ADDRESS:
            print("Advertencia: PLC_IP_ADDRESS no está configurado en .env. La conexión al PLC será simulada.")
        
        # Intentar conectar. Si falla, el estado de is_connected se mantendrá False
        # y las operaciones futuras usarán la simulación.
        try:
            self.connect()
        except PLCConnectionError as e:
            print(f"Error inicial de conexión al PLC: {e}. Las operaciones de PLC serán simuladas.")
            self.is_connected = False # Asegurar que esté en False si la conexión inicial falla

    def connect(self):
        """Intenta conectar al PLC."""
        if self.is_connected:
            print("Ya conectado al PLC.")
            return True
        
        # Si S7Client es None, significa que python-snap7 no se importó correctamente,
        # así que simulamos la conexión.
        if S7Client is None or not PLC_IP_ADDRESS:
            print("Simulando conexión al PLC (python-snap7 no disponible o IP no configurada)...")
            self.is_connected = True # Simular conexión exitosa
            return True

        try:
            self.client = S7Client.Client()
            self.client.connect(PLC_IP_ADDRESS, PLC_RACK, PLC_SLOT)
            self.is_connected = self.client.get_connected()
            if self.is_connected:
                print(f"Conectado al PLC en {PLC_IP_ADDRESS} (Rack: {PLC_RACK}, Slot: {PLC_SLOT})")
            else:
                print(f"No se pudo conectar al PLC en {PLC_IP_ADDRESS}.")
                self.client.destroy()
                self.client = None
                raise PLCConnectionError(f"No se pudo establecer conexión con el PLC en {PLC_IP_ADDRESS}.")
            return self.is_connected
        except Snap7Exception as e: # Ahora Snap7Exception será una clase de excepción válida
            print(f"Error de conexión con Snap7: {e}")
            self.client = None
            self.is_connected = False
            raise PLCConnectionError(f"Error de Snap7 al conectar: {e}")
        except Exception as e:
            print(f"Error inesperado al conectar al PLC: {e}")
            self.client = None
            self.is_connected = False
            raise PLCConnectionError(f"Error inesperado al conectar: {e}")

    def disconnect(self):
        """Desconecta del PLC."""
        if self.is_connected and self.client:
            if S7Client is None: # Simulación
                print("Simulando desconexión del PLC...")
            else:
                self.client.disconnect()
                self.client.destroy()
                print("Desconectado del PLC.")
            self.is_connected = False
            self.client = None
        elif self.is_connected: # Caso simulado
            print("Simulando desconexión del PLC (ya estaba 'conectado').")
            self.is_connected = False
        else:
            print("No conectado al PLC.")

    def _ensure_connection(self):
        # Intentar conectar si no lo está. Si falla, lanzar una excepción.
        if not self.is_connected and not self.connect():
            raise PLCConnectionError("No hay conexión con el PLC. Falló el intento de reconexión.")

    def read_db(self, db_number: int, start_byte: int, size: int) -> bytearray:
        """
        Lee datos de un Data Block (DB) en el PLC.
        Args:
            db_number (int): Número del Data Block.
            start_byte (int): Byte inicial a leer.
            size (int): Cantidad de bytes a leer.
        Returns:
            bytearray: Los datos leídos.
        """
        self._ensure_connection() # Esto ahora maneja la conexión/reconexión
        if S7Client is None or not self.client or not self.is_connected:
            print(f"Simulando lectura de DB{db_number}, StartByte: {start_byte}, Size: {size}")
            # Retorna un bytearray simulado (ej. todos ceros)
            return bytearray([0] * size)
        
        try:
            data = self.client.db_read(db_number, start_byte, size)
            print(f"Lectura exitosa de DB{db_number} desde byte {start_byte}, {size} bytes.")
            return data
        except Snap7Exception as e:
            raise PLCReadWriteError(f"Error de Snap7 al leer DB{db_number}: {e}")
        except Exception as e:
            raise PLCReadWriteError(f"Error inesperado al leer DB{db_number}: {e}")

    def write_db(self, db_number: int, start_byte: int, data: bytearray) -> bool:
        """
        Escribe datos en un Data Block (DB) en el PLC.
        Args:
            db_number (int): Número del Data Block.
            start_byte (int): Byte inicial para escribir.
            data (bytearray): Los datos a escribir.
        Returns:
            bool: True si la escritura fue exitosa, False en caso contrario.
        """
        self._ensure_connection() # Esto ahora maneja la conexión/reconexión
        if S7Client is None or not self.client or not self.is_connected:
            print(f"Simulando escritura en DB{db_number}, StartByte: {start_byte}, Data: {data.hex()}")
            return True # Simular escritura exitosa
        
        try:
            self.client.db_write(db_number, start_byte, data)
            print(f"Escritura exitosa en DB{db_number} en byte {start_byte}, {len(data)} bytes.")
            return True
        except Snap7Exception as e:
            raise PLCReadWriteError(f"Error de Snap7 al escribir en DB{db_number}: {e}")
        except Exception as e:
            raise PLCReadWriteError(f"Error inesperado al escribir en DB{db_number}: {e}")

    def read_m(self, start_byte: int, size: int) -> bytearray:
        """Lee datos de la memoria M (Merker) del PLC."""
        self._ensure_connection()
        if S7Client is None or not self.client or not self.is_connected:
            print(f"Simulando lectura de M, StartByte: {start_byte}, Size: {size}")
            return bytearray([0] * size) # Simulado
        try:
            data = self.client.read_area(0x83, 0, start_byte, size) # Area ID for Merker (M) is 0x83
            return data
        except Snap7Exception as e:
            raise PLCReadWriteError(f"Error de Snap7 al leer M{start_byte}: {e}")
        except Exception as e:
            raise PLCReadWriteError(f"Error inesperado al leer M{start_byte}: {e}")

    def write_m(self, start_byte: int, data: bytearray) -> bool:
        """Escribe datos en la memoria M (Merker) del PLC."""
        self._ensure_connection()
        if S7Client is None or not self.client or not self.is_connected:
            print(f"Simulando escritura en M, StartByte: {start_byte}, Data: {data.hex()}")
            return True # Simulado
        try:
            self.client.write_area(0x83, 0, start_byte, data)
            return True
        except Snap7Exception as e:
            raise PLCReadWriteError(f"Error de Snap7 al escribir en M{start_byte}: {e}")
        except Exception as e:
            raise PLCReadWriteError(f"Error inesperado al escribir en M{start_byte}: {e}")

    # --- Funciones para leer/escribir tipos de datos específicos (usando snap7.util) ---
    def read_real(self, db_number: int, byte_offset: int) -> float:
        data = self.read_db(db_number, byte_offset, 4) # Real es 4 bytes
        return S7Util.get_real(data, 0) if S7Client else 0.0

    def write_real(self, db_number: int, byte_offset: int, value: float) -> bool:
        buffer = bytearray(4)
        if S7Client:
            S7Util.set_real(buffer, 0, value)
        return self.write_db(db_number, byte_offset, buffer)

    def read_int(self, db_number: int, byte_offset: int) -> int:
        data = self.read_db(db_number, byte_offset, 2) # Int es 2 bytes
        return S7Util.get_int(data, 0) if S7Client else 0

    def write_int(self, db_number: int, byte_offset: int, value: int) -> bool:
        buffer = bytearray(2)
        if S7Client:
            S7Util.set_int(buffer, 0, value)
        return self.write_db(db_number, byte_offset, buffer)

    def read_bool(self, db_number: int, byte_offset: int, bit_offset: int) -> bool:
        data = self.read_db(db_number, byte_offset, 1) # Leer el byte completo
        return S7Util.get_bool(data, 0, bit_offset) if S7Client else False

    def write_bool(self, db_number: int, byte_offset: int, bit_offset: int, value: bool) -> bool:
        # Para escribir un booleano, primero leemos el byte, modificamos el bit y luego escribimos el byte modificado
        # Asegurarse de que el bytearray tenga al menos 1 byte
        current_byte_data = self.read_db(db_number, byte_offset, 1)
        if len(current_byte_data) == 0: # Si la lectura simulada o fallida devuelve vacío
             current_byte_data = bytearray([0]) # Crear un byte nuevo inicializado a 0
        
        if S7Client:
            S7Util.set_bool(current_byte_data, 0, bit_offset, value)
        return self.write_db(db_number, byte_offset, current_byte_data)

    # Añadir más funciones para otros tipos de datos (DINT, WORD, DWORD, etc.) según sea necesario.

# Ejemplo de uso (para pruebas directas de este módulo)
if __name__ == "__main__":
    print("--- Probando PLCHandler (plc_handler.py) ---")

    # Asegúrate de tener PLC_IP_ADDRESS en tu .env si quieres probar la conexión real
    # Por ejemplo: PLC_IP_ADDRESS="192.168.0.10"
    # y si tu PLC está configurado en Rack 0, Slot 1

    try:
        plc = PLCHandler()

        if plc.is_connected:
            print("\nPLC conectado. Realizando operaciones de prueba...")

            # Ejemplo: Leer un BOOL (Bit 0 del Byte 0 del DB1)
            # Asegúrate de que DB1 exista en tu PLC y que el byte 0 esté accesible
            db_num = 1
            byte_off = 0
            bit_off = 0
            print(f"\n--- Probando lectura/escritura de BOOL (DB{db_num}.DBX{byte_off}.{bit_off}) ---")
            try:
                current_bool_value = plc.read_bool(db_num, byte_off, bit_off)
                print(f"Estado actual de DB{db_num}.DBX{byte_off}.{bit_off}: {current_bool_value}")

                # Alternar el valor
                new_bool_value = not current_bool_value
                print(f"Cambiando DB{db_num}.DBX{byte_off}.{bit_off} a: {new_bool_value}")
                if plc.write_bool(db_num, byte_off, bit_off, new_bool_value):
                    print("Escritura de BOOL exitosa.")
                    # Verificar la escritura leyendo de nuevo
                    time.sleep(0.1) # Pequeña pausa para que el PLC actualice
                    verified_bool_value = plc.read_bool(db_num, byte_off, bit_off)
                    print(f"Estado verificado de DB{db_num}.DBX{byte_off}.{bit_off}: {verified_bool_value}")
                else:
                    print("Error al escribir BOOL.")
            except PLCReadWriteError as rw_err:
                print(f"Error en operación BOOL: {rw_err}")
            except Exception as e:
                print(f"Error inesperado en prueba BOOL: {e}")

            # Ejemplo: Leer y escribir un INT (Byte 10 del DB1)
            print(f"\n--- Probando lectura/escritura de INT (DB{db_num}.DBW10) ---")
            byte_off_int = 10
            try:
                current_int_value = plc.read_int(db_num, byte_off_int)
                print(f"Estado actual de DB{db_num}.DBW{byte_off_int}: {current_int_value}")

                new_int_value = (current_int_value + 1) % 100 # Incrementar y modular para ejemplo
                print(f"Cambiando DB{db_num}.DBW{byte_off_int} a: {new_int_value}")
                if plc.write_int(db_num, byte_off_int, new_int_value):
                    print("Escritura de INT exitosa.")
                    time.sleep(0.1)
                    verified_int_value = plc.read_int(db_num, byte_off_int)
                    print(f"Estado verificado de DB{db_num}.DBW{byte_off_int}: {verified_int_value}")
                else:
                    print("Error al escribir INT.")
            except PLCReadWriteError as rw_err:
                print(f"Error en operación INT: {rw_err}")
            except Exception as e:
                print(f"Error inesperado en prueba INT: {e}")
                
            # Ejemplo de lectura de memoria M
            print("\n--- Probando lectura de memoria M (MW0) ---")
            m_byte_offset = 0
            m_size = 2 # Leer una palabra (2 bytes)
            try:
                m_data = plc.read_m(m_byte_offset, m_size)
                print(f"Datos de MW{m_byte_offset}: {m_data.hex()} (bytearray)")
                # Puedes convertirlo a INT si es una palabra
                if S7Client:
                    m_int_value = S7Util.get_int(m_data, 0)
                    print(f"Valor entero de MW{m_byte_offset}: {m_int_value}")
            except PLCReadWriteError as rw_err:
                print(f"Error en lectura de M: {rw_err}")


        else:
            print("\nNo se pudo establecer conexión real con el PLC. Las operaciones serán simuladas.")
            # Puedes añadir pruebas con simulación aquí si lo deseas
            db_num = 1
            byte_off = 0
            bit_off = 0
            print(f"\n--- Probando simulación de escritura de BOOL (DB{db_num}.DBX{byte_off}.{bit_off}) ---")
            try:
                if plc.write_bool(db_num, byte_off, bit_off, True):
                    print("Simulación de escritura de BOOL exitosa.")
                else:
                    print("Error en simulación de escritura de BOOL.")
            except PLCReadWriteError as rw_err:
                print(f"Error en simulación de operación BOOL: {rw_err}")

    except PLCConnectionError as ce:
        print(f"Error de conexión inicial al PLC: {ce}")
    except Exception as e:
        print(f"Error inesperado durante la prueba del PLCHandler: {e}")
    finally:
        if 'plc' in locals() and plc.is_connected:
            plc.disconnect()