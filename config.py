# ===== CONFIG =====

# ============ ЦВЕТА ИНТЕРФЕЙСА (черно-зелено-белая схема) ============
BG_MAIN = '#0a0a0a'           # Основной фон - почти черный
BG_PANEL = '#1a1a1a'          # Панели - темно-серый
BG_SIDEBAR = '#141414'        # Боковая панель
ACCENT_GREEN = '#00ff41'      # Яркий неоновый зеленый
ACCENT_GREEN_HOVER = '#00cc33'  # Зеленый при наведении
TEXT_WHITE = '#ffffff'        # Белый текст
TEXT_GRAY = '#b0b0b0'         # Серый текст
BORDER_COLOR = '#2a2a2a'      # Цвет границ
SUCCESS_COLOR = (0, 255, 65)  # Зеленый для OpenCV (BGR)
ERROR_COLOR = '#ff1744'       # Красный для ошибок

# ============ ЦВЕТА ДЛЯ ДЕТЕКЦИИ (OpenCV BGR) ============
BOX_COLOR = (255, 0, 0)       # Синий для боксов вне зоны
COUNTING_ZONE_COLOR = (65, 255, 0)  # Зеленый для зоны
TEXT_COLOR = (255, 255, 255)  # Белый текст

# ============ ПАРАМЕТРЫ КАМЕРЫ ============
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# ============ ПАРАМЕТРЫ ОТОБРАЖЕНИЯ ============
DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 700

# ============ ПАРАМЕТРЫ YOLO ============
DEFAULT_MODEL = "models/barcodes.pt"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# ============ ПАРАМЕТРЫ ЗОНЫ ПОДСЧЕТА ============
COUNTING_ZONE_X_START = 0.1
COUNTING_ZONE_X_END = 0.9
COUNTING_ZONE_ALPHA = 0.2

# ============ ПАРАМЕТРЫ ШТРИХКОДОВ ============
BARCODE_PADDING = 20

# ============ ПАРАМЕТРЫ ТРЕКЕРА ============
MAX_DISTANCE = 100              # Максимальное расстояние для сопоставления объектов
HISTORY_SIZE = 30               # Размер истории для трекера (количество кадров)
IOU_THRESHOLD_TRACKING = 0.3    # Порог IoU для трекинга
MAX_AGE = 30                    # Максимальное количество кадров без обнаружения

# ============ ПУТИ ============
MODELS_FOLDER = "models"
CONFIG_FILE = "models_config.json"

# ============ ОТЛАДКА ============
DEBUG = False
