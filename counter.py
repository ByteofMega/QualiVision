# ===== COUNTER =====
from utils import is_in_counting_zone
from config import *
import time


class ObjectCounter:
    """Счётчик для штрихкодов"""
    def __init__(self):
        self.scanned_items = {}
        self.total_unique_count = 0

    def update(self, tracker_id, data, box, counting_zone):
        in_zone = is_in_counting_zone(box, counting_zone)
        if not in_zone:
            return None
        if data:
            if tracker_id not in self.scanned_items:
                self.scanned_items[tracker_id] = {'data': data, 'timestamp': time.time()}
                self.total_unique_count += 1
                print(f"[НОВЫЙ] ID #{tracker_id} | {data} | Всего: {self.total_unique_count}")
            else:
                self.scanned_items[tracker_id]['timestamp'] = time.time()
            return self.scanned_items[tracker_id]['data']
        if tracker_id in self.scanned_items:
            return self.scanned_items[tracker_id]['data']
        return None

    def reset(self):
        self.scanned_items = {}
        self.total_unique_count = 0


class ScrewCounter:
    """Счётчик для винтиков"""
    def __init__(self):
        self.total_count = 0
        self.counted_tracker_ids = set()

    def update_count(self, tracker_ids, boxes, counting_zone):
        current_in_zone = 0
        for tracker_id, box in zip(tracker_ids, boxes):
            if is_in_counting_zone(box, counting_zone):
                current_in_zone += 1
                if tracker_id not in self.counted_tracker_ids:
                    self.counted_tracker_ids.add(tracker_id)
                    self.total_count += 1
                    print(f"✅ Новый винтик ID #{tracker_id} | Всего: {self.total_count}")
            else:
                if DEBUG:
                    print(f"❌ Винтик #{tracker_id} вне зоны")
        if DEBUG:
            print(f"Counter: {current_in_zone} в зоне, {len(self.counted_tracker_ids)} подсчитано")
        return self.total_count

    def reset(self):
        self.total_count = 0
        self.counted_tracker_ids = set()
        print("🔄 Счётчик сброшен")
