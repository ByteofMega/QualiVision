# ===== UTILS =====
import cv2
import numpy as np
import time
from config import *


def initialize_camera(camera_id=CAMERA_ID, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=FPS):
    """Инициализирует камеру с заданными параметрами."""
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру {camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def calculate_iou(box1, box2):
    """Вычисляет IoU (Intersection over Union) двух bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def is_in_counting_zone(box, counting_zone):
    """Проверяет, находится ли bounding box в зоне подсчета."""
    x1, y1, x2, y2 = map(int, box)
    zx1, zy1, zx2, zy2 = counting_zone

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    in_zone = zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2

    if DEBUG:
        print(f"Center: ({center_x:.1f},{center_y:.1f}), Zone: [{zx1}-{zx2}], In: {in_zone}")

    return in_zone


def get_counting_zone_coordinates(frame_width, frame_height, x_start_ratio, x_end_ratio):
    """Возвращает координаты зоны подсчёта (x1, y1, x2, y2)."""
    return (
        int(frame_width * x_start_ratio),
        0,
        int(frame_width * x_end_ratio),
        frame_height
    )


def draw_counting_zone(frame, x_start_ratio, x_end_ratio,
                       color=COUNTING_ZONE_COLOR, alpha=COUNTING_ZONE_ALPHA):
    """Рисует зону подсчёта. Возвращает (frame, zone_coords)."""
    h, w = frame.shape[:2]
    x1, y1 = int(w * x_start_ratio), 0
    x2, y2 = int(w * x_end_ratio), h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, "COUNTING ZONE", (x1 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame, (x1, y1, x2, y2)


def draw_box(frame, box, color, thickness=2):
    """Рисует один bounding box (для режима штрихкодов)."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_boxes(frame, boxes, tracker_ids, counting_zone, box_color=BOX_COLOR):
    """Рисует все боксы с ID и статусом зоны (режим винтиков)."""
    annotated = frame.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        in_zone = is_in_counting_zone(box, counting_zone)
        color = SUCCESS_COLOR if in_zone else box_color

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if i < len(tracker_ids):
            label = f"#{tracker_ids[i]} {'IN ZONE' if in_zone else 'outside'}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if DEBUG:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(annotated, (cx, cy), 3, color, -1)

    return annotated


def add_text(frame, text, position=(10, 30), color=TEXT_COLOR, scale=0.9, thickness=2):
    """Добавляет текст на кадр."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return frame


def draw_fps(frame, fps):
    """Рисует FPS на кадре. fps — уже посчитанное значение."""
    if frame is not None and isinstance(frame, np.ndarray):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


# ===== УНИВЕРСАЛЬНАЯ ФУНКЦИЯ ДЛЯ СИСТЕМЫ ПРОФИЛЕЙ =====

def draw_boxes_universal(frame, boxes, tracker_ids, profile, counting_zone, barcode_data=None):
    """
    Универсальная отрисовка боксов на основе профиля задачи.

    Args:
        frame: кадр для отрисовки
        boxes: список bounding boxes
        tracker_ids: ID объектов от трекера
        profile: объект TaskProfile с настройками
        counting_zone: координаты зоны подсчета (x1, y1, x2, y2)
        barcode_data: список данных штрихкодов (опционально)

    Returns:
        annotated_frame: кадр с отрисованными боксами и метками
    """
    annotated = frame.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # Определяем, находится ли объект в зоне
        if profile.zone_enabled:
            in_zone = is_in_counting_zone(box, counting_zone)
        else:
            in_zone = True  # Если зона отключена, все объекты "в зоне"

        # Выбираем цвет бокса
        color = profile.box_color_in_zone if in_zone else profile.box_color_default

        # Рисуем бокс
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Формируем и рисуем метку
        if profile.show_labels and i < len(tracker_ids):
            label = profile.label_format
            label = label.replace('{id}', str(tracker_ids[i]))

            # Подстановка данных штрихкода (если есть)
            if barcode_data and i < len(barcode_data) and barcode_data[i]:
                label = label.replace('{data}', barcode_data[i][:20])
            else:
                # Убираем плейсхолдер, если данных нет
                label = label.replace(' | {data}', '').replace('{data}', '')

            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Отладочная информация (если DEBUG включен)
        if DEBUG:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(annotated, (cx, cy), 3, color, -1)
            cv2.putText(annotated, f"({cx},{cy})", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return annotated
