import sounddevice as sd
from scipy.io.wavfile import write
import wavio
import os
import whisper # Usaremos el modelo local de Whisper

# Cargar el modelo de Whisper una vez
try:
    # Puedes elegir entre 'tiny', 'base', 'small', 'medium', 'large'
    # 'base' es un buen equilibrio para empezar. 'tiny' es muy rápido.
    WHISPER_MODEL = whisper.load_model("base")
    print("Modelo Whisper 'base' cargado.")
except Exception as e:
    WHISPER_MODEL = None
    print(f"Error al cargar el modelo Whisper: {e}. La transcripción de voz no estará disponible.")

def record_audio(filename="output.wav", duration=5, samplerate=44100):
    """
    Graba audio del micrófono durante una duración especificada.
    Guarda el audio en un archivo WAV.
    """
    if not sd.query_devices(kind='input'):
        raise ValueError("No se encontraron dispositivos de entrada de audio (micrófonos). Asegúrate de tener uno conectado.")

    print(f"Grabando audio durante {duration} segundos...")
    try:
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()  # Espera a que la grabación termine
        wavio.write(filename, audio_data, samplerate, sampwidth=2) # wavio para mejor compatibilidad
        print(f"Grabación guardada en {filename}")
        return filename
    except Exception as e:
        raise Exception(f"Error al grabar audio: {e}. Asegúrate de que el micrófono esté correctamente configurado y no en uso.")

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe un archivo de audio a texto usando el modelo Whisper.
    """
    if not WHISPER_MODEL:
        return "ERROR: El modelo Whisper no está cargado. No se puede transcribir el audio."

    print(f"Transcribiendo audio de {audio_path}...")
    try:
        # result = WHISPER_MODEL.transcribe(audio_path, fp16=False) # fp16=False si tienes problemas con CUDA/GPU
        result = WHISPER_MODEL.transcribe(audio_path)
        transcription = result["text"]
        print(f"Transcripción: {transcription}")
        return transcription
    except Exception as e:
        return f"ERROR: Falló la transcripción del audio: {e}"

if __name__ == "__main__":
    # Ejemplo de uso
    try:
        output_file = "temp_recording.wav"
        recorded_file = record_audio(output_file, duration=5) # Graba 5 segundos
        if recorded_file:
            transcribed_text = transcribe_audio(recorded_file)
            print(f"Texto transcrito: {transcribed_text}")
            os.remove(recorded_file) # Limpiar archivo temporal
    except Exception as e:
        print(f"Error en la prueba de audio_utils: {e}")