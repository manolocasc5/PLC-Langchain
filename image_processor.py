import os
from PIL import Image
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from openai import APIConnectionError, RateLimitError
import base64
import traceback
import pytesseract # Importar pytesseract

# Configurar la ruta al ejecutable de Tesseract si no está en el PATH del sistema
# Solo necesario en algunos sistemas operativos (ej. Windows) si la instalación no lo añadió al PATH.
# Reemplaza 'C:/Program Files/Tesseract-OCR/tesseract.exe' con la ruta real en tu sistema.
# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Cargar variables de entorno
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ImageProcessor:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY debe estar configurado en el archivo .env")
        
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.vision_model = "gpt-4o" 

        try:
            print("DEBUG: Cargando modelo 'all-MiniLM-L6-v2' para embeddings...")
            self.sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Modelo 'all-MiniLM-L6-v2' cargado para embeddings.")
        except Exception as e:
            print(f"Error al cargar el modelo 'all-MiniLM-L6-v2': {e}")
            print("Asegúrate de tener conexión a internet o de haber descargado el modelo previamente.")
            self.sentence_transformer_model = None

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Codifica una imagen en formato base64.
        """
        print(f"DEBUG: Codificando imagen {image_path} a base64...")
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        print("DEBUG: Imagen codificada a base64.")
        return encoded_string

    def perform_ocr_on_image(self, image_path: str) -> str: # Hacemos público el método OCR
        """
        Extrae texto de una imagen usando Tesseract OCR.
        Este método es el que se llamará desde streamlit_app.py para la pestaña de OCR.
        """
        print(f"DEBUG: Ejecutando OCR en la imagen: {image_path}...")
        try:
            img = Image.open(image_path)
            # Para español, puedes especificar el idioma: lang='spa'
            # Para inglés: lang='eng'
            # Puedes combinar: lang='eng+spa'
            text = pytesseract.image_to_string(img, lang='spa') # Idioma español
            text = text.strip() # Limpiar espacios en blanco al inicio/final
            print(f"DEBUG: Texto extraído por OCR: '{text}'")
            return text
        except pytesseract.TesseractNotFoundError:
            print("ERROR: Tesseract OCR no encontrado. Asegúrate de que esté instalado y en tu PATH.")
            print("Para Windows, considera añadir: pytesseract.pytesseract.tesseract_cmd = r'RUTA/A/TESSERACT.EXE'")
            raise # Lanzar la excepción para que Streamlit la capture y la muestre al usuario
        except Exception as e:
            print(f"ERROR al ejecutar OCR en la imagen {image_path}: {e}")
            traceback.print_exc()
            return f"ERROR: No se pudo extraer texto con OCR. Detalles: {e}"

    def describe_image_with_ai(self, image_path: str) -> tuple[str, list, str, str, str]:
        """
        Genera una descripción detallada, palabras clave, clasificación de tipo de elemento,
        texto extraído por OCR y texto visible extraído por IA para una imagen.
        Args:
            image_path (str): Ruta al archivo de imagen.
        Returns:
            tuple: (description: str, keywords: list, element_type: str, ocr_text: str, ai_extracted_text: str)
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"La imagen no se encontró en: {image_path}")
            
            # --- Paso 1: Ejecutar OCR para obtener texto de forma programática ---
            # Ahora llamamos al método público que también puede ser usado por la pestaña de OCR
            ocr_text = self.perform_ocr_on_image(image_path) 

            # --- Paso 2: Preparar la imagen para la API de OpenAI ---
            base64_image = self._encode_image_to_base64(image_path)
            
            prompt_content = (
                "Eres un experto analista de interfaces de usuario (UI) y tu tarea es describir con máxima precisión "
                "el elemento visual en la siguiente imagen. Tu objetivo es proporcionar una descripción exhaustiva "
                "que permita identificar unívocamente el elemento y su función.\n\n"
                
                "Detalles a identificar y describir (en orden de prioridad):\n"
                "1.  **Tipo de Elemento UI:** Clasifica el elemento en una de las siguientes categorías estrictas: 'icono', 'pestaña', 'campo_texto', 'boton', 'desplegable', 'enlace', 'barra_desplazamiento', 'menu', 'ventana', 'fondo', 'otro'. Selecciona la más específica posible.\n"
                "2.  **Texto Visible Principal:** Identifica y transcribe textualmente cualquier texto visible directamente en el elemento (ej. 'Guardar', 'Cancelar', 'Inicio', 'Usuario', un número, etc.). Si no hay texto, indica 'Ninguno'.\n"
                "3.  **Propósito/Función:** Describe brevemente la acción que realiza el elemento o su significado semántico en el contexto de una UI.\n"
                "4.  **Características Visuales:** Colores predominantes, forma (rectangular, circular, etc.), tamaño relativo (si es posible inferirlo), bordes, sombreado, y otros detalles estéticos o de diseño relevantes.\n"
                "5.  **Contenido de Icono/Gráfico:** Si es un icono o gráfico, describe claramente su representación visual (ej. 'un disquete', 'una lupa', 'tres líneas horizontales').\n\n"

                "Genera también una lista de palabras clave relevantes para la búsqueda y clasificación. Estas deben incluir: "
                "el texto visible, el tipo de elemento UI, y términos descriptivos de su función o apariencia.\n\n"

                "Tu respuesta debe seguir este formato estricto. Si un campo no aplica (ej. 'Texto visible' si no hay), indica 'Ninguno'.\n"
                "Descripción: [Descripción detallada y concisa del elemento, incluyendo los puntos anteriores de forma narrativa.]\n"
                "Texto visible extraído por IA: [Texto directo visible en el elemento, o 'Ninguno']\n"
                "Palabras clave: [palabra1, palabra2, ...]\n"
                "Tipo de elemento: [Tipo de elemento de UI de la lista proporcionada]"
            )
            
            print(f"DEBUG: Enviando solicitud a OpenAI con modelo {self.vision_model}...")
            response = self.openai_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_content},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=500,
            )
            
            text_response = response.choices[0].message.content.strip()
            print(f"DEBUG: Respuesta cruda de OpenAI:\n{text_response}\n---")
            
            description = ""
            keywords = []
            element_type = "otro"
            ai_extracted_text = "Ninguno" # Inicializar el nuevo campo

            # Parsear la respuesta
            lines = text_response.split('\n')
            for line in lines:
                if line.startswith("Descripción:"):
                    description = line.replace("Descripción:", "").strip()
                elif line.startswith("Texto visible extraído por IA:"): # Nuevo campo a parsear
                    ai_extracted_text = line.replace("Texto visible extraído por IA:", "").strip()
                elif line.startswith("Palabras clave:"):
                    keywords_str = line.replace("Palabras clave:", "").strip()
                    if keywords_str:
                        keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                    else:
                        keywords = []
                elif line.startswith("Tipo de elemento:"):
                    element_type = line.replace("Tipo de elemento:", "").strip().lower()
            
            print(f"DEBUG: Descripción extraída: {description}")
            print(f"DEBUG: Texto visible extraído por IA: {ai_extracted_text}") # Debug del nuevo campo
            print(f"DEBUG: Palabras clave extraídas: {keywords}")
            print(f"DEBUG: Tipo de elemento extraído: {element_type}")
            
            # Devolver todos los valores, incluyendo el texto del OCR y el de la IA
            return description, keywords, element_type, ocr_text, ai_extracted_text
        except FileNotFoundError as fnfe:
            print(f"Error: {fnfe}")
            return "", [], "otro", "ERROR: Imagen no encontrada", "Ninguno"
        except RateLimitError:
            print("Error: Se ha alcanzado el límite de tasa de OpenAI. Por favor, espera y reintenta.")
            return "", [], "otro", ocr_text, "Ninguno" # Devolvemos ocr_text si se obtuvo
        except APIConnectionError as ace:
            print(f"Error de conexión a la API de OpenAI: {ace}. Verifica tu conexión a internet o la URL de la API.")
            return "", [], "otro", ocr_text, "Ninguno"
        except pytesseract.TesseractNotFoundError:
            # Captura aquí para un mensaje más amigable al usuario en la consola si este error ocurre
            # dentro de describe_image_with_ai.
            print("ERROR: Tesseract OCR no está instalado o no se encuentra en el PATH. "
                  "Por favor, instálalo desde https://tesseract-ocr.github.io/tessdoc/Installation.html "
                  "y asegúrate de que esté en tu PATH o configura pytesseract.pytesseract.tesseract_cmd.")
            return "", [], "otro", "ERROR: Tesseract OCR no encontrado", "Ninguno"
        except Exception as e:
            print(f"Error inesperado al describir la imagen con OpenAI: {e}")
            traceback.print_exc()
            return "", [], "otro", ocr_text, "Ninguno"

    def generate_embedding_from_text(self, text: str) -> list[float]:
        """
        Genera un embedding de texto de 384 dimensiones utilizando 'all-MiniLM-L6-v2'.
        """
        if self.sentence_transformer_model is None:
            raise RuntimeError("El modelo de Sentence Transformer no se cargó correctamente. No se pueden generar embeddings.")
        
        try:
            print(f"DEBUG: Generando embedding para el texto: '{text[:50]}...'")
            embedding = self.sentence_transformer_model.encode(text).tolist()
            if len(embedding) != 384:
                raise ValueError(f"El embedding generado no tiene la dimensión esperada (384), sino {len(embedding)}")
            print("DEBUG: Embedding generado exitosamente.")
            return embedding
        except Exception as e:
            print(f"Error al generar embedding de texto: {e}")
            traceback.print_exc()
            return [0.0] * 384 

# Ejemplo de uso (para pruebas directas de este módulo)
if __name__ == "__main__":
    print("--- Probando ImageProcessor (image_processor.py) con OpenAI y OCR ---")
    processor = ImageProcessor()

    # Asegúrate de que este archivo exista y sea un recorte de UI real.
    # Por ejemplo, puedes tomar una captura de pantalla de un icono o botón
    # y guardarlo como 'papelera_reciclaje.png' o similar.
    # Un buen ejemplo sería un botón con texto claro, como "Aceptar.png"
    # Puedes usar las imágenes que me has subido antes, por ejemplo:
    # 'image_01a69f.png' (icono de Excel)
    # 'image_0204d7.png' (un botón)
    test_clipping_path = "image_0204d7.png" # <--- ¡CÁMBIALO POR UNA DE TUS IMÁGENES!

    if os.path.exists(test_clipping_path):
        print(f"\nProcesando imagen: {test_clipping_path}")
        description, keywords, element_type, ocr_text, ai_extracted_text = processor.describe_image_with_ai(test_clipping_path)
        
        print("\n--- Resultado combinado ---")
        print(f"    Descripción (IA): {description}")
        print(f"    Texto visible (IA): {ai_extracted_text}")
        print(f"    Texto visible (OCR): {ocr_text}") # Nuevo campo en la salida
        print(f"    Palabras clave: {keywords}")
        print(f"    Tipo de elemento: {element_type}")

        # Puedes decidir cuál texto es más fiable o combinarlos para el embedding
        # Por ejemplo, priorizar OCR si tiene contenido y AI_extracted_text es "Ninguno"
        text_for_embedding = description
        if ocr_text and ocr_text.strip() != "" and "ERROR" not in ocr_text: # Se añadió .strip() para considerar cadenas vacías de espacios
            # Si el OCR extrajo algo y la IA no, o si el OCR es significativamente mejor, prioriza o combina.
            # Por simplicidad, aquí lo añadimos a la descripción para el embedding si es relevante.
            # Una estrategia más avanzada podría comparar la calidad o la longitud.
            if ai_extracted_text.lower() == "ninguno" or len(ocr_text) > len(ai_extracted_text): # Se añadió .lower() para "Ninguno"
                text_for_embedding = f"{description}. Texto detectado: {ocr_text}"
            elif ocr_text not in description: # Si el texto OCR no está ya en la descripción, añádelo
                text_for_embedding = f"{description}. OCR: {ocr_text}"
        
        print(f"\nDEBUG: Texto usado para embedding: '{text_for_embedding[:100]}...'")

        if text_for_embedding:
            try:
                embedding = processor.generate_embedding_from_text(text_for_embedding)
                print(f"    Embedding generado (primeros 5 valores): {embedding[:5]}...")
                print(f"    Longitud del embedding: {len(embedding)}")
            except RuntimeError as re:
                print(f"Error al generar embedding: {re}")
        else:
            print("No se generó descripción o texto suficiente para crear embedding.")
    else:
        print(f"\n¡ATENCIÓN! El archivo de prueba '{test_clipping_path}' no se encontró.")
        print("Asegúrate de tener una imagen de recorte de UI en el mismo directorio que este script o especifica una ruta completa.")
        print("Puedes usar 'image_01a69f.png' o 'image_0204d7.png' si las has subido al entorno.")