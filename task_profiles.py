# ===== TASK PROFILES =====
import json
import os


class TaskProfile:
    """Класс для загрузки и хранения профиля задачи из JSON"""

    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Основная информация
        self.task_name = data['task_name']
        self.model_path = data['model_path']
        self.task_type = data['task_type']
        self.config_path = config_path  # Сохраняем путь к конфигу

        # Зона подсчета
        zone = data['counting_zone']
        self.zone_enabled = zone['enabled']
        self.zone_x_start = zone['x_start']
        self.zone_x_end = zone['x_end']

        # Визуализация
        vis = data['visualization']
        self.show_labels = vis['show_labels']
        self.label_format = vis['label_format']
        self.box_color_default = tuple(vis['box_color_default'])
        self.box_color_in_zone = tuple(vis['box_color_in_zone'])
        self.zone_color = tuple(vis['zone_color'])

        # Детекция
        det = data['detection']
        self.confidence = det['confidence']
        self.iou = det['iou']
        self.target_class = det.get('target_class', None)

        # Обработка
        proc = data['processing']
        self.use_barcode_reading = proc['use_barcode_reading']
        self.use_tracking = proc['use_tracking']
        self.tracker_max_age = proc['tracker_max_age']
        self.tracker_max_distance = proc['tracker_max_distance']

    def __str__(self):
        return f"TaskProfile({self.task_name}, {self.model_path})"


def load_all_profiles(profiles_dir="task_profiles"):
    """Загружает все профили из указанной директории"""
    profiles = []

    if not os.path.exists(profiles_dir):
        os.makedirs(profiles_dir)
        print(f"⚠️ Папка {profiles_dir} создана. Добавьте JSON профили.")
        return profiles

    for file in os.listdir(profiles_dir):
        if file.endswith('.json'):
            try:
                profile_path = os.path.join(profiles_dir, file)
                profile = TaskProfile(profile_path)
                profiles.append(profile)
                print(f"✅ Загружен профиль: {profile.task_name}")
            except Exception as e:
                print(f"❌ Ошибка загрузки профиля {file}: {e}")

    if not profiles:
        print("⚠️ Профили не найдены. Создайте .json файлы в папке task_profiles/")

    return profiles
