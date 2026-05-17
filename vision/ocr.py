

try:
    from paddleocr import PaddleOCR
    PADDLE_OK = True
except ImportError:
    PADDLE_OK = False

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_OK = True
except ImportError:
    RAPID_OK = False

import numpy as np
import cv2


class GameOCREngine:
    def __init__(self, engine="auto", lang="es", conf_min=0.3):
        self._engine_name = engine
        self._lang = lang
        self._conf_min = conf_min
        self._paddle = None
        self._rapid = None

    def _init_paddle(self):
        if PADDLE_OK and self._paddle is None:
            try:
                self._paddle = PaddleOCR(use_angle_cls=True, lang=self._lang, show_log=False)
                return True
            except Exception:
                return False
        return PADDLE_OK and self._paddle is not None

    def _init_rapid(self):
        if RAPID_OK and self._rapid is None:
            try:
                self._rapid = RapidOCR()
                return True
            except Exception:
                return False
        return RAPID_OK and self._rapid is not None

    def read(self, image, conf_min=None):
        if conf_min is None:
            conf_min = self._conf_min

        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            arr = image
        else:
            arr = np.array(image.convert("RGB"))
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        results = []

        if self._engine_name == "auto" or self._engine_name == "paddle":
            if self._init_paddle():
                try:
                    paddle_result = self._paddle.ocr(arr, cls=True)
                    if paddle_result and paddle_result[0]:
                        for line in paddle_result[0]:
                            text = line[1][0]
                            conf = line[1][1]
                            if conf >= conf_min and len(text.strip()) > 1:
                                results.append({"text": text, "confidence": float(conf)})
                except Exception:
                    pass

        if (self._engine_name == "auto" and not results) or self._engine_name == "rapid":
            if self._init_rapid():
                try:
                    rapid_result, elapse = self._rapid(arr)
                    if rapid_result:
                        for box, text, conf in rapid_result:
                            if conf is not None and conf >= conf_min and len(text.strip()) > 1:
                                results.append({"text": text, "confidence": float(conf)})
                except Exception:
                    pass

        return results

    def read_text(self, image, conf_min=None):
        results = self.read(image, conf_min)
        texts = [r["text"] for r in results if len(r["text"].strip()) > 2]
        return " ".join(texts)


