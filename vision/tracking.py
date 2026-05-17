import cv2
import numpy as np
from .template import ColorDetector, SceneAnalyzer
from .motion import MotionDetector as MotionTrackerDetector


class ORBTracker:
    def __init__(self, n_features=500):
        self._orb = cv2.ORB_create(nfeatures=n_features)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self._orb.detectAndCompute(gray, None)

    @staticmethod
    def draw_keypoints(frame, keypoints):
        return cv2.drawKeypoints(frame, keypoints, None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def match_templates(self, frame, templates):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        kp_frame, des_frame = self._orb.detectAndCompute(gray, None)
        results = []
        if des_frame is None or len(kp_frame) < 4:
            return results
        for tpl in templates:
            if "descriptors" not in tpl:
                continue
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(tpl["descriptors"], des_frame)
            if len(matches) > 6:
                pts = [kp_frame[m.trainIdx].pt for m in matches]
                xs = [int(p[0]) for p in pts]
                ys = [int(p[1]) for p in pts]
                x_min, x_max = max(0, min(xs)), min(w, max(xs))
                y_min, y_max = max(0, min(ys)), min(h, max(ys))
                bw, bh = x_max - x_min, y_max - y_min
                if bw > 10 and bh > 10:
                    results.append({"class_name": f"personaje_{tpl['name']}",
                                    "confidence": min(0.9, len(matches) / 30.0),
                                    "box": [x_min, y_min, bw, bh]})
        return results


class UIDetector:
    def detect(self, frame):
        h, w = frame.shape[:2]
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_half = cv2.resize(gray, (w // 2, h // 2))
        edges = cv2.Canny(gray_half, 80, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            x2, y2, bw2, bh2 = x * 2, y * 2, bw * 2, bh * 2
            if bw2 < 30 or bh2 < 15 or bw2 > w * 0.4 or bh2 > h * 0.15:
                continue
            if cv2.contourArea(cnt) / max(bw * bh, 1) > 0.7:
                _, std = cv2.meanStdDev(frame[y2:y2 + bh2, x2:x2 + bw2])
                if np.mean(std) < 60:
                    results.append({"class_name": "boton_ui", "confidence": 0.55,
                                    "box": [x2, y2, bw2, bh2]})

        gray_tiny = cv2.resize(gray, (64, 48))
        fast = cv2.FastFeatureDetector_create(threshold=25)
        kp = fast.detect(gray_tiny, None)
        if len(kp) > 20 and np.std([p.pt[0] for p in kp]) > 20 and np.std([p.pt[1] for p in kp]) > 15:
            results.append({"class_name": "inventario_ui", "confidence": 0.5, "box": [0, 0, w, h]})

        hsv_half = cv2.cvtColor(cv2.resize(frame, (w // 2, h // 2)), cv2.COLOR_BGR2HSV)
        menu_mask = cv2.inRange(hsv_half, np.array([0, 0, 0]), np.array([180, 80, 100]))
        menu_mask = cv2.dilate(menu_mask, None, iterations=2)
        menu_contours, _ = cv2.findContours(menu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in menu_contours:
            if cv2.contourArea(cnt) > (w // 2) * (h // 2) * 0.3:
                x, y, bw, bh = cv2.boundingRect(cnt)
                results.append({"class_name": "menu_ui", "confidence": 0.5,
                                "box": [x * 2, y * 2, bw * 2, bh * 2]})

        return results


class LifeBarDetector:
    COLORS = [
        ("vida_verde", np.array([40, 80, 80]), np.array([80, 255, 255])),
        ("vida_roja", np.array([0, 100, 100]), np.array([10, 255, 255])),
        ("vida_roja2", np.array([170, 100, 100]), np.array([180, 255, 255])),
        ("vida_azul", np.array([100, 80, 80]), np.array([130, 255, 255])),
        ("vida_amarilla", np.array([20, 100, 100]), np.array([35, 255, 255])),
    ]

    def __init__(self, min_w=15, min_h=3):
        self._min_w = min_w
        self._min_h = min_h

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        results = []

        for name, lo, hi in self.COLORS:
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                rect_area = bw * bh
                if not (bw > bh * 2 and bw > self._min_w and bh >= self._min_h and bw < w * 0.4 and bh < h * 0.06):
                    continue
                if rect_area > 0 and area / rect_area < 0.15:
                    continue
                results.append({"class_name": name, "confidence": 0.6, "box": [x, y, bw, bh]})

        return results

    def analyze_hp_percent(self, hp_roi):
        if hp_roi is None or hp_roi.size == 0:
            return 100
        mask = cv2.inRange(cv2.cvtColor(hp_roi, cv2.COLOR_BGR2HSV),
                           np.array([0, 100, 100]), np.array([10, 255, 255]))
        total = hp_roi.shape[0] * hp_roi.shape[1]
        return (np.count_nonzero(mask) / total) * 100 if total else 100


class DetectionPersistence:
    def __init__(self):
        self._persistence = {}
        self._min_frames = {"default": 2, "personaje_": 3, "escenario_": 5,
                            "boton_ui": 5, "menu_ui": 5, "minimapa": 5,
                            "vida_": 2, "patron_repetido": 4, "objeto_": 2}

    def filter(self, detections):
        current = {f"{d['class_name']}_{tuple(d['box'])}": d for d in detections}

        for k in list(self._persistence.keys()):
            if k not in current:
                self._persistence[k] -= 1
                if self._persistence[k] <= 0:
                    del self._persistence[k]

        for k, d in current.items():
            self._persistence[k] = self._persistence.get(k, 0) + 1

        result = []
        for k, d in current.items():
            pval = self._persistence[k]
            threshold = 2
            for prefix, t in self._min_frames.items():
                if d["class_name"].startswith(prefix):
                    threshold = t
                    break
            if pval >= threshold:
                result.append(d)
        return result


class VisionEngine:
    def __init__(self):
        self._detectors = {}
        self._persistence = DetectionPersistence()

    def analyze(self, frame, full_scan=False):
        if frame is None or frame.size == 0:
            return []

        results = []

        if "color" not in self._detectors:
            self._detectors["color"] = ColorDetector()
        results.extend(self._detectors["color"].detect(frame))

        if "motion" not in self._detectors:
            self._detectors["motion"] = MotionTrackerDetector()
        _, _, regions = self._detectors["motion"].detect(frame)
        for r in regions:
            results.append({"class_name": "movimiento", "confidence": 0.7, "box": r["box"]})

        if "scene" not in self._detectors:
            self._detectors["scene"] = SceneAnalyzer()
        for label in self._detectors["scene"].analyze(frame):
            h, w = frame.shape[:2]
            results.append({"class_name": label, "confidence": 0.5, "box": [0, 0, w, h]})

        if "lifebars" not in self._detectors:
            self._detectors["lifebars"] = LifeBarDetector()
        results.extend(self._detectors["lifebars"].detect(frame))

        if full_scan:
            if "ui" not in self._detectors:
                self._detectors["ui"] = UIDetector()
            results.extend(self._detectors["ui"].detect(frame))

        results = self._persistence.filter(results)
        return results
