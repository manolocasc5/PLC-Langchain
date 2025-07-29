import sounddevice as sd
from scipy.io.wavfile import write # Aunque wavio es preferido, scipy.io.wavfile puede ser útil para algunos casos
import wavio
import os
import whisper # Usaremos el modelo local de Whisper
import traceback # Para imprimir el stack trace en caso de errores inesperados
import sys # Para un manejo más limpio de la salida de errores

# Cargar el modelo de Whisper una vez al inicio del módulo
WHISPER_MODEL = None # Inicializar a None para que esté definido globalmente y evitar UnboundLocalError
try:
    # Puedes elegir entre 'tiny', 'base', 'small', 'medium', 'large'
    # 'base' es un buen equilibrio para empezar. 'tiny' es muy rápido.
    print("DEBUG: Cargando modelo Whisper 'base'...")
    WHISPER_MODEL = whisper.load_model("base")
    print("Modelo Whisper 'base' cargado.")
except Exception as e:
    WHISPER_MODEL = None # Asegurarse de que sigue siendo None si falla la carga
    print(f"ERROR: Error al cargar el modelo Whisper: {e}. La transcripción de voz no estará disponible.", file=sys.stderr)
    traceback.print_exc(file=sys.stderr) # Imprime el stack trace a stderr para depuración

def record_audio(filename="output.wav", duration=5, samplerate=44100):
    """
    Graba audio del micrófono durante una duración especificada.
    Guarda el audio en un archivo WAV.
    Args:
        filename (str): Nombre del archivo donde se guardará el audio.
        duration (int): Duración de la grabación en segundos.
        samplerate (int): Tasa de muestreo del audio.
    Returns:
        str: La ruta al archivo grabado si es exitoso, None en caso contrario.
    Raises:
        ValueError: Si no se encuentran dispositivos de entrada de audio.
        Exception: Si ocurre un error durante la grabación.
    """
    try:
        # Verificar si hay algún dispositivo de entrada de audio disponible
        # Se verifica sd.query_devices() directamente, ya que sd.default.device puede no existir o no tener 'max_input_channels'
        input_devices = [device for device in sd.query_devices() if device['max_input_channels'] > 0]
        if not input_devices:
            raise ValueError("No se encontraron dispositivos de entrada de audio (micrófonos). Asegúrate de tener uno conectado y configurado.")

        print(f"Grabando audio durante {duration} segundos en {filename}...")
        # Usa el dispositivo de entrada por defecto (device=None). Si hay problemas, se podría especificar un device=ID
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()   # Espera a que la grabación termine
        
        # Guardar usando wavio para mejor compatibilidad con diferentes encabezados WAV
        # Convierte el array de numpy a un formato adecuado para wavio.
        # wavio espera un array con el eje de canales como el último.
        # sd.rec devuelve (frames, channels), así que ya está en el formato correcto para channels=1
        wavio.write(filename, audio_data, samplerate, sampwidth=2) 
        print(f"Grabación guardada con éxito en {filename}")
        return filename
    except ValueError as ve:
        print(f"ERROR al grabar audio: {ve}", file=sys.stderr)
        return None
    except sd.PortAudioError as pae:
        print(f"ERROR de PortAudio al grabar audio: {pae}. Posiblemente el micrófono está en uso o mal configurado.", file=sys.stderr)
        print("Intenta reiniciar la aplicación o verificar la configuración de audio del sistema.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR inesperado al grabar audio: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Imprime el stack trace a stderr para depuración
        return None

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe un archivo de audio a texto usando el modelo Whisper.
    Args:
        audio_path (str): Ruta al archivo de audio a transcribir.
    Returns:
        str: El texto transcrito o un mensaje de error.
    """
    if not WHISPER_MODEL:
        return "ERROR: El modelo Whisper no está cargado. No se puede transcribir el audio."

    if not os.path.exists(audio_path):
        return f"ERROR: El archivo de audio '{audio_path}' no existe. No se puede transcribir."

    print(f"Transcribiendo audio de {audio_path}...")
    try:
        # Si tienes problemas de rendimiento o memoria con la GPU, puedes probar fp16=False:
        # result = WHISPER_MODEL.transcribe(audio_path, fp16=False) 
        result = WHISPER_MODEL.transcribe(audio_path)
        transcription = result["text"].strip() # .strip() para limpiar espacios extra
        print(f"Transcripción: \"{transcription}\"")
        return transcription
    except Exception as e:
        print(f"ERROR: Falló la transcripción del audio '{audio_path}': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Imprime el stack trace a stderr para depuración
        return f"ERROR: Falló la transcripción del audio: {e}"

if __name__ == "__main__":
    # Ejemplo de uso
    print("--- Probando audio_utils (grabación y transcripción) ---")
    try:
        output_file = "temp_recording.wav"
        # Eliminar archivo anterior si existe
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                print(f"Archivo anterior '{output_file}' eliminado para asegurar una prueba limpia.")
            except OSError as e:
                print(f"Advertencia: No se pudo eliminar el archivo anterior '{output_file}': {e}", file=sys.stderr)

        print("\nPreparado para grabar 5 segundos de audio. Por favor, habla ahora...")
        recorded_file = record_audio(output_file, duration=5) # Graba 5 segundos
        
        if recorded_file and os.path.exists(recorded_file): # Verificar que el archivo se creó y existe
            print("\nGrabación completada. Transcribiendo...")
            transcribed_text = transcribe_audio(recorded_file)
            print(f"\nResultado final de la transcripción: \"{transcribed_text}\"")
            
            # Limpiar archivo temporal
            try:
                os.remove(recorded_file) 
                print(f"Archivo temporal '{recorded_file}' eliminado.")
            except OSError as e:
                print(f"Advertencia: No se pudo eliminar el archivo temporal '{recorded_file}': {e}", file=sys.stderr)
        else:
            print("No se pudo grabar el audio. No se procederá con la transcripción.", file=sys.stderr)
            
    except Exception as e:
        print(f"ERROR en la prueba principal de audio_utils: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Imprime el stack trace a stderr para depuración