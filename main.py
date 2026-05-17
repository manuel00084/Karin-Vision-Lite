"""
Karin Vision Lite — Sistema de Visión para Juegos (bajo consumo)

Modos:
  vision  — Análisis visual en vivo con ventana OpenCV
  ocr     — OCR de texto en pantalla
  monitor — Escaneo periódico + log de detecciones
"""
import time
import threading
import argparse
import cv2
import numpy as np

from vision.capture import ScreenCapture
from vision.motion import MotionDetector
from vision.template import SceneAnalyzer
from vision.ocr import GameOCREngine
from vision.tracking import UIDetector, LifeBarDetector, VisionEngine


class KarinVisionLite:
    def __init__(self):
        self._running = False
        self._thread = None
        self._mode = "vision"
        self._log_fn = print

        self.capture = ScreenCapture(target_fps=20)
        self.motion = MotionDetector()
        self.scene = SceneAnalyzer()
        self.ocr = GameOCREngine(engine="auto", lang="es")
        self.ui = UIDetector()
        self.lifebars = LifeBarDetector()
        self.engine = VisionEngine()

        self._frame_count = 0

    def set_logger(self, log_fn):
        self._log_fn = log_fn

    def set_mode(self, mode):
        if mode in ("ocr", "vision", "monitor"):
            self._mode = mode

    def log(self, msg):
        self._log_fn(f"[KVL] {msg}")

    # ── OCR Mode ──────────────────────────────────────────────────────

    def _run_ocr_mode(self):
        last_ocr = 0
        interval = 3.0

        while self._running:
            try:
                frame = self.capture.grab()
                if frame is None:
                    time.sleep(0.5)
                    continue

                now = time.time()
                if now - last_ocr < interval:
                    time.sleep(0.3)
                    continue

                last_ocr = now
                text = self.ocr.read_text(frame)
                if text:
                    self.log(f"OCR: {text}")

            except Exception as e:
                self.log(f"Error: {e}")
                time.sleep(1)

    # ── Vision Mode ───────────────────────────────────────────────────

    def _run_vision_mode(self):
        fps_counter = 0
        fps_time = time.time()
        full_scan_counter = 0

        while self._running:
            try:
                frame = self.capture.grab()
                if frame is None:
                    time.sleep(0.1)
                    continue

                fps_counter += 1
                self._frame_count += 1
                if time.time() - fps_time >= 2.0:
                    self.log(f"FPS: {fps_counter / 2:.1f}")
                    fps_counter = 0
                    fps_time = time.time()

                debug = frame.copy()

                # Lightweight: motion + scene (every frame)
                motion, score, regions = self.motion.detect(frame)
                for r in regions:
                    x, y, bw, bh = r["box"]
                    cv2.rectangle(debug, (x, y), (x + bw, y + bh), (0, 255, 255), 2)

                # Scene labels (runs on 64x48, cheap)
                scene_labels = self.scene.analyze(frame)
                if scene_labels:
                    cv2.putText(debug, scene_labels[0].replace("escenario_", ""),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Transition detection (cheap)
                trans, _ = self.motion.detect_transition(frame)
                if trans:
                    full_scan_counter = 0

                # Life bars (mid-weight)
                if self._frame_count % 3 == 0:
                    life_dets = self.lifebars.detect(frame)
                    for lb in life_dets:
                        x, y, bw, bh = lb["box"]
                        cv2.rectangle(debug, (x, y), (x + bw, y + bh), (0, 255, 0), 1)

                # Full scan every ~30 frames or on transition
                full_scan_counter += 1
                if full_scan_counter > 30 or trans:
                    full_scan_counter = 0
                    flow = self.motion.detect_optical_flow(frame)
                    if flow.get("direction"):
                        cv2.putText(debug, flow["direction"],
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    ui_dets = self.ui.detect(frame)
                    for u in ui_dets:
                        x, y, bw, bh = u["box"]
                        cv2.rectangle(debug, (x, y), (x + bw, y + bh), (255, 0, 255), 1)

                cv2.imshow("Karin Vision Lite", debug)

                slowed = cv2.waitKey(1)
                if slowed == 27:
                    self._running = False

            except Exception as e:
                self.log(f"Error: {e}")
                time.sleep(0.5)

    # ── Monitor Mode ──────────────────────────────────────────────────

    def _run_monitor_mode(self):
        last_scan = 0

        while self._running:
            try:
                frame = self.capture.grab()
                if frame is None:
                    time.sleep(1)
                    continue

                now = time.time()
                if now - last_scan < 4.0:
                    time.sleep(0.5)
                    continue

                last_scan = now

                motion, _, _ = self.motion.detect(frame)
                scene = self.scene.analyze(frame)
                flow = self.motion.detect_optical_flow(frame)

                parts = []
                if scene:
                    parts.append(scene[0].replace("escenario_", ""))
                if motion:
                    parts.append("movimiento")
                if flow.get("direction"):
                    parts.append(flow["direction"])

                self.log(" | ".join(parts) if parts else "sin detecciones")

            except Exception as e:
                self.log(f"Error: {e}")
                time.sleep(2)

    # ── Control ───────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        modes = {"ocr": self._run_ocr_mode, "vision": self._run_vision_mode,
                 "monitor": self._run_monitor_mode}
        target = modes.get(self._mode, self._run_vision_mode)
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()
        self.log(f"Iniciado (modo: {self._mode})")

    def stop(self):
        self._running = False
        self.capture.release()
        cv2.destroyAllWindows()

    def wait(self):
        if self._thread and self._thread.is_alive():
            self._thread.join()


def main():
    parser = argparse.ArgumentParser(description="Karin Vision Lite")
    parser.add_argument("--mode", choices=["ocr", "vision", "monitor"],
                        default="vision")
    args = parser.parse_args()

    kvl = KarinVisionLite()
    kvl.set_mode(args.mode)

    print(f"Karin Vision Lite — {args.mode}")
    print("ESC o Ctrl+C para salir.\n")

    try:
        kvl.start()
        kvl.wait()
    except KeyboardInterrupt:
        pass
    finally:
        kvl.stop()


if __name__ == "__main__":
    main()
