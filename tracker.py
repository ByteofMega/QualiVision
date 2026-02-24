# ===== TRACKER =====
from collections import deque
import numpy as np
from utils import calculate_iou
from config import *


class ByteTracker:
    def __init__(self, max_distance=MAX_DISTANCE, history_size=HISTORY_SIZE, iou_threshold=IOU_THRESHOLD_TRACKING):
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold
        self.history_size = history_size
        self.next_id = 1
        self.tracks = {}

    def calculate_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def calculate_distance(self, center1, center2):
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    def is_same_screw(self, new_box, track_history):
        for historical_box in track_history:
            if calculate_iou(new_box, historical_box) > self.iou_threshold:
                return True
        return False

    # Добавлены аргументы со значениями по умолчанию
    def update(self, boxes, previous_count=0, current_count=0):
        if not boxes:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['frames_since_update'] += 1
                if self.tracks[track_id]['frames_since_update'] > MAX_AGE:
                    del self.tracks[track_id]
            return []

        current_centers = [self.calculate_center(box) for box in boxes]
        tracker_ids = []
        matched_detections = set()

        for track_id, track_data in list(self.tracks.items()):
            track_center = self.calculate_center(track_data['current_box'])
            best_match_idx = -1
            min_distance = float('inf')

            for i, det_center in enumerate(current_centers):
                if i in matched_detections:
                    continue
                distance = self.calculate_distance(track_center, det_center)
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    best_match_idx = i

            if best_match_idx != -1:
                new_box = boxes[best_match_idx]
                track_data['current_box'] = new_box
                track_data['frames_since_update'] = 0
                track_data['history'].append(new_box)
                if len(track_data['history']) > self.history_size:
                    track_data['history'].popleft()
                tracker_ids.append(track_id)
                matched_detections.add(best_match_idx)
            else:
                track_data['frames_since_update'] += 1
                if track_data['frames_since_update'] > MAX_AGE:
                    del self.tracks[track_id]

        for i in range(len(boxes)):
            if i not in matched_detections:
                new_box = boxes[i]
                is_old_screw = False

                # Логика трекинга винтиков
                count_diff = abs(current_count - previous_count)
                if count_diff <= 1:
                    for track_id, track_data in self.tracks.items():
                        if self.is_same_screw(new_box, track_data['history']):
                            is_old_screw = True
                            track_data['current_box'] = new_box
                            track_data['frames_since_update'] = 0
                            track_data['history'].append(new_box)
                            if len(track_data['history']) > self.history_size:
                                track_data['history'].popleft()
                            tracker_ids.append(track_id)
                            break

                if not is_old_screw:
                    new_track_id = self.next_id
                    self.next_id += 1
                    self.tracks[new_track_id] = {
                        'current_box': new_box,
                        'frames_since_update': 0,
                        'history': deque([new_box], maxlen=self.history_size)
                    }
                    tracker_ids.append(new_track_id)

        return tracker_ids
