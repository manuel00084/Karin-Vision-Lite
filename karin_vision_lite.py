"""
karin_vision_lite.py - Detector de visión ligero para Karin VTuber usando HOG + SVM
Especializado en detección de personas para comentarios de juegos en tiempo real
Muy rápido y de bajo consumo de recursos
"""
import cv2
import numpy as np
import threading

class KarinVisionLiteDetector:
    def __init__(self, win_stride=(8, 8), padding=(16, 16), scale=1.05):
        """
        Inicializa el detector de visión ligero Karin Vision Lite usando HOG + SVM
        
        Args:
            win_stride: Tamaño de paso para la ventana deslizante (default: (8,8))
            padding: Relleno para la ventana deslizante (default: (16,16))
            scale: Factor de escala para la pirámide de imágenes (default: 1.05)
        """
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.hog = cv2.HOGDescriptor()
        # El detector SVM preentrenado para personas está incluido en OpenCV
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._initialized = True
        print("Karin Vision Lite Detector inicializado (usando detector SVM preentrenado de OpenCV)")
    
    def detect(self, frame, target_classes=None):
        """
        Detecta personas en un frame usando HOG + SVM
        
        Args:
            frame: Imagen de entrada (numpy array en formato BGR)
            target_classes: Lista de nombres de clases a detectar (por ahora solo soporta "person")
            
        Returns:
            Lista de detecciones, cada una es un dict con:
            - 'class_id': ID de la clase (0 para persona)
            - 'class_name': Nombre de la clase
            - 'confidence': Confianza de la detección (0-1)
            - 'box': [x, y, width, height] en coordenadas del frame original
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Detectar personas
            # Devuelve las coordenadas de las cajas y los pesos (confianza)
            boxes, weights = self.hog.detectMultiScale(
                frame, 
                winStride=self.win_stride,
                padding=self.padding,
                scale=self.scale
            )
            
            results = []
            for (x, y, w, h), weight in zip(boxes, weights):
                # El peso es una medida de confianza (mayor es mejor)
                # Normalizamos a un rango 0-1 para mantener consistencia
                # Los pesos de HOG pueden ser >1, así que limitamos a un rango razonable
                confidence = min(0.95, max(0.1, weight * 0.4))  # Ajuste empirico para mejor rango
                
                # Si se especificaron clases objetivo, filtramos
                if target_classes and "person" not in target_classes:
                    continue
                
                results.append({
                    'class_id': 0,  # Persona
                    'class_name': 'person',
                    'confidence': float(confidence),
                    'box': [int(x), int(y), int(w), int(h)]
                })
            
            return results
            
        except Exception as e:
            print(f"Error durante la detección Karin Vision Lite: {e}")
            return []

# Instancia global para reutilización (singleton)
_detector_instance = None
_detector_lock = threading.Lock()

def get_karin_vision_lite(win_stride=(8, 8), padding=(16, 16), scale=1.05):
    """
    Obtiene una instancia compartida del detector Karin Vision Lite (singleton pattern)
    
    Args:
        win_stride: Tamaño de paso para la ventana deslizante
        padding: Relleno para la ventana deslizante
        scale: Factor de escala para la pirámide de imágenes
        
    Returns:
        Instancia de KarinVisionLiteDetector
    """
    global _detector_instance
    with _detector_lock:
        if _detector_instance is None:
            _detector_instance = KarinVisionLiteDetector(win_stride, padding, scale)
        return _detector_instance

def detect_persons_karin_vision_lite(frame, win_stride=(8, 8), padding=(16, 16), scale=1.05):
    """
    Función de conveniencia para detección rápida de personas usando Karin Vision Lite
    
    Args:
        frame: Imagen de entrada (BGR numpy array)
        win_stride, padding, scale: Parámetros para HOG
        
    Returns:
        Lista de detecciones de personas (ver KarinVisionLiteDetector.detect)
    """
    detector = get_karin_vision_lite(win_stride, padding, scale)
    return detector.detect(frame, target_classes=["person"])

# Ejemplo de uso (comentado para evitar ejecuciones accidentales)
if __name__ == "__main__":
    # Este bloque solo se ejecuta si el archivo se ejecuta directamente
    import time
    
    # Simular una captura de cámara o carga de imagen
    print("Inicializando Karin Vision Lite Detector...")
    detector = KarinVisionLiteDetector()
    
    # Crear un frame de prueba (en la práctica, esto vendría de su captura de pantalla)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("Ejecutando detección de prueba...")
    start_time = time.time()
    detections = detector.detect(test_frame, target_classes=["person"])
    end_time = time.time()
    
    print(f"Detección completada en {end_time - start_time:.3f} segundos")
    print(f"Se encontraron {len(detections)} personas:")
    for det in detections[:5]:  # Mostrar máximo 5 detecciones
        print(f"  - {det['class_name']}: {det['confidence']:.2f} en {det['box']}")