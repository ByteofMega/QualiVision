# ===== DETECTOR =====
import cv2
from ultralytics import YOLO
from pyzbar.pyzbar import decode
from tracker import ByteTracker
from config import *


class ObjectDetector:
    def __init__(self, model_path=DEFAULT_MODEL):
        """Инициализирует модель YOLO и трекер."""
        self.model = YOLO(model_path)
        self.tracker = ByteTracker()
        print(f"✅ Модель загружена: {model_path}")

    def load_model(self, model_path):
        """Загружает новую модель и сбрасывает трекер."""
        self.model = YOLO(model_path)
        self.tracker = ByteTracker()  # Сброс трекера при смене модели
        print(f"✅ Модель переключена: {model_path}")

    def predict(self, frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD):
        """Выполняет предсказание YOLO на кадре."""
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            save=False,
            show=False,
            verbose=False
        )
        return results

    def get_barcode_data(self, frame, box):
        """
        Вырезает область, улучшает контраст и декодирует штрихкод.
        Используется только для задач сканирования штрихкодов.
        """
        x1, y1, x2, y2 = map(int, box)
        h_f, w_f = frame.shape[:2]

        # Добавляем padding для захвата тихой зоны
        pad = BARCODE_PADDING
        px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
        px2, py2 = min(w_f, x2 + pad), min(h_f, y2 + pad)

        crop = frame[py1:py2, px1:px2]
        if crop.size == 0:
            return None

        # Конвертация в серый
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Попытка 1: Обычное декодирование
        decoded = decode(gray)

        # Попытка 2: Бинаризация Otsu
        if not decoded:
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            decoded = decode(thresh)

        # Попытка 3: Инверсия (белый текст на черном фоне)
        if not decoded:
            thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            decoded = decode(thresh_inv)

        return decoded[0].data.decode('utf-8') if decoded else None

    def process_detections(self, results, profile=None, previous_count=0, current_count=0):
        """
        Универсальная обработка детекций на основе профиля задачи.

        Args:
            results: результаты YOLO predict
            profile: объект TaskProfile с настройками (опционально)
            previous_count: предыдущее количество объектов (для трекера)
            current_count: текущее количество в зоне (для трекера)

        Returns:
            tracker_ids: список ID от трекера
            boxes: список координат боксов
        """
        boxes = []

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for detection in results[0].boxes:
                # Если профиль задан и есть фильтр по классу
                if profile and profile.target_class is not None:
                    if int(detection.cls) != profile.target_class:
                        continue

                box = detection.xyxy[0].cpu().numpy()
                boxes.append(box)

        if DEBUG:
            print(f"Detector: {len(boxes)} объектов найдено")

        # Применяем трекинг
        tracker_ids = self.tracker.update(boxes, previous_count, current_count)

        # Возвращаем только те боксы, для которых есть ID
        return tracker_ids, boxes[:len(tracker_ids)]
