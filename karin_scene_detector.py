"""
karin_scene_detector.py - Detector de escenarios ligero para Karin VTuber
Clasifica el paisaje/fondo de videojuegos usando análisis de color y textura
Muy rápido y de bajo consumo de recursos
"""
import cv2
import numpy as np
import threading

class KarinSceneDetector:
    """
    Detector de escenarios de videojuegos basado en análisis de color y textura.
    Clasifica el fondo/paisaje en categorías como: bosque, ciudad, desierto, nieve, agua, etc.
    """
    
    SCENE_TYPES = {
        "forest": "Bosque",
        "city": "Ciudad",
        "desert": "Desierto",
        "snow": "Nieve",
        "water": "Agua",
        "sky": "Cielo",
        "night": "Noche",
        "indoor": "Interior",
        "sunset": "Atardecer",
        "grassland": "Pastizal",
        "mountain": "Montaña",
        "unknown": "Desconocido"
    }
    
    def __init__(self):
        self._initialized = True
        print("Karin Scene Detector inicializado (análisis de color + textura)")
    
    def _extract_color_features(self, frame):
        """Extrae características de color del frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        h, w = frame.shape[:2]
        total_pixels = h * w
        
        # Máscaras de color en HSV
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        blue_mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([130, 255, 255]))
        brown_mask = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([25, 255, 255]))
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 40]), np.array([180, 30, 200]))
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        orange_mask = cv2.inRange(hsv, np.array([5, 80, 80]), np.array([25, 255, 255]))
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
        
        # Porcentajes de cada color
        features = {
            "green_pct": np.sum(green_mask > 0) / total_pixels,
            "blue_pct": np.sum(blue_mask > 0) / total_pixels,
            "brown_pct": np.sum(brown_mask > 0) / total_pixels,
            "gray_pct": np.sum(gray_mask > 0) / total_pixels,
            "white_pct": np.sum(white_mask > 0) / total_pixels,
            "orange_pct": np.sum(orange_mask > 0) / total_pixels,
            "dark_pct": np.sum(dark_mask > 0) / total_pixels,
        }
        
        # Promedio de canales LAB para luminosidad
        features["avg_luminance"] = np.mean(lab[:, :, 0]) / 255.0
        
        return features
    
    def _extract_texture_features(self, frame):
        """Extrae características de textura usando bordes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detección de bordes con Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        
        # Varianza de la imagen en escala de grises (textura)
        gray_variance = np.var(gray) / (255.0 ** 2)
        
        return {
            "edge_density": edge_density,
            "texture_variance": gray_variance
        }
    
    def _analyze_sky_region(self, frame):
        """Analiza la región superior del frame para detectar cielo"""
        h, w = frame.shape[:2]
        sky_region = frame[0:h//4, :, :]
        hsv_sky = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
        
        blue_sky_mask = cv2.inRange(hsv_sky, np.array([90, 30, 100]), np.array([130, 255, 255]))
        sky_blue_pct = np.sum(blue_sky_mask > 0) / (sky_region.shape[0] * sky_region.shape[1])
        
        return sky_blue_pct
    
    def classify(self, frame):
        """
        Clasifica el escenario/paisaje del frame
        
        Args:
            frame: Imagen de entrada (numpy array en formato BGR)
            
        Returns:
            Dict con:
            - 'scene': Tipo de escenario (str)
            - 'scene_name': Nombre en español (str)
            - 'confidence': Confianza de la clasificación (0-1)
            - 'details': Características extraídas (dict)
        """
        if frame is None or frame.size == 0:
            return {"scene": "unknown", "scene_name": "Desconocido", "confidence": 0.0, "details": {}}
        
        try:
            # Redimensionar para procesamiento rápido
            h, w = frame.shape[:2]
            if h > 320 or w > 480:
                frame = cv2.resize(frame, (480, 320))
            
            color_features = self._extract_color_features(frame)
            texture_features = self._extract_texture_features(frame)
            sky_blue = self._analyze_sky_region(frame)
            
            details = {**color_features, **texture_features, "sky_blue_pct": sky_blue}
            
            # Reglas de clasificación basadas en características
            scene, confidence = self._classify_scene(details)
            
            return {
                "scene": scene,
                "scene_name": self.SCENE_TYPES.get(scene, "Desconocido"),
                "confidence": confidence,
                "details": details
            }
            
        except Exception as e:
            print(f"Error durante clasificación de escena: {e}")
            return {"scene": "unknown", "scene_name": "Desconocido", "confidence": 0.0, "details": {}}
    
    def _classify_scene(self, features):
        """
        Clasifica la escena usando reglas basadas en características de color y textura
        Retorna (scene_type, confidence)
        """
        scores = {}
        
        # Bosque: mucho verde, textura media-alta
        scores["forest"] = features["green_pct"] * 0.7 + (0.3 if features["texture_variance"] > 0.05 else 0.0)
        
        # Agua: mucho azul, baja textura
        scores["water"] = features["blue_pct"] * 0.8 + (0.2 if features["texture_variance"] < 0.03 else 0.0)
        
        # Nieve: mucho blanco, luminosidad alta
        scores["snow"] = features["white_pct"] * 0.7 + (0.3 if features["avg_luminance"] > 0.7 else 0.0)
        
        # Desierto: marrón/naranja dominante, textura baja-media
        scores["desert"] = (features["brown_pct"] + features["orange_pct"]) * 0.8 + (0.2 if features["edge_density"] < 0.15 else 0.0)
        
        # Ciudad: gris alto, densidad de bordes alta
        scores["city"] = features["gray_pct"] * 0.5 + features["edge_density"] * 0.5
        
        # Noche: muy oscuro
        scores["night"] = features["dark_pct"] * 0.8 + (0.2 if features["avg_luminance"] < 0.25 else 0.0)
        
        # Cielo: azul en región superior + azul general
        scores["sky"] = features["sky_blue_pct"] * 0.6 + features["blue_pct"] * 0.4
        
        # Atardecer: naranja + luminosidad media
        scores["sunset"] = features["orange_pct"] * 0.6 + (0.4 if 0.3 < features["avg_luminance"] < 0.6 else 0.0)
        
        # Pastizal: verde pero menos textura que bosque
        scores["grassland"] = features["green_pct"] * 0.6 + (0.4 if features["texture_variance"] < 0.05 else 0.0)
        
        # Montaña: gris + textura alta + algo de verde
        scores["mountain"] = features["gray_pct"] * 0.3 + features["green_pct"] * 0.3 + features["edge_density"] * 0.4
        
        # Interior: gris + oscuridad + baja densidad de bordes
        scores["indoor"] = features["gray_pct"] * 0.4 + (0.3 if features["avg_luminance"] < 0.5 else 0.0) + (0.3 if features["edge_density"] < 0.1 else 0.0)
        
        # Obtener el máximo
        best_scene = max(scores, key=scores.get)
        best_score = scores[best_scene]
        
        # Umbral mínimo de confianza
        if best_score < 0.15:
            return "unknown", 0.0
        
        # Normalizar confianza a 0-1
        confidence = min(0.95, best_score)
        
        return best_scene, confidence


# Instancia global (singleton)
_detector_instance = None
_detector_lock = threading.Lock()

def get_karin_scene_detector():
    """Obtiene una instancia compartida del detector de escenas"""
    global _detector_instance
    with _detector_lock:
        if _detector_instance is None:
            _detector_instance = KarinSceneDetector()
        return _detector_instance

def classify_scene(frame):
    """Función de conveniencia para clasificar una escena"""
    detector = get_karin_scene_detector()
    return detector.classify(frame)


if __name__ == "__main__":
    import time
    
    print("Inicializando Karin Scene Detector...")
    detector = KarinSceneDetector()
    
    # Frame de prueba con colores verdes (simula bosque)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:, :, 1] = 120  # Verde
    test_frame[:, :, 0] = 30   # Poco azul
    test_frame[:, :, 2] = 30   # Poco rojo
    
    print("Clasificando escena de prueba...")
    start_time = time.time()
    result = detector.classify(test_frame)
    end_time = time.time()
    
    print(f"Clasificación completada en {end_time - start_time:.3f} segundos")
    print(f"Escenario: {result['scene_name']} ({result['scene']})")
    print(f"Confianza: {result['confidence']:.2f}")
