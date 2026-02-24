# ===== MAIN APP =====

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import json
import os
import shutil
from pathlib import Path

# Импортируем модули проекта
from detector import ObjectDetector
from counter import ObjectCounter, ScrewCounter
from utils import (
    draw_counting_zone, draw_fps, get_counting_zone_coordinates,
    draw_boxes_universal, is_in_counting_zone
)
from config import *
from task_profiles import load_all_profiles

# ==============================================================================
# ↓↓↓ ВСТАВЬТЕ ПУТЬ ДО КАРТИНКИ ЛОГОТИПА ЗДЕСЬ (СТРОКА 22) ↓↓↓
LOGO_PATH = "QualiVision_logo_no_bg.png"


# ==============================================================================


class YOLOApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("QualiVision")
        self.root.geometry("1400x900")
        self.root.configure(bg=BG_MAIN)
        self.root.resizable(True, True)

        # Флаги состояния
        self.camera_running = False
        self.is_detecting = True
        self.is_recording = False
        self.info_panel_visible = False
        self.is_fullscreen = False
        self.recording_path = os.getcwd()  # Путь для сохранения видео

        self.last_frame = None
        self.frame_lock = threading.Lock()
        self.cap = None
        self.video_writer = None

        # Инициализация ИИ и счётчиков
        self.detector = ObjectDetector()
        self.counter = ObjectCounter()
        self.screw_counter = ScrewCounter()

        # Система профилей
        self.profiles = load_all_profiles()
        self.current_profile = None

        if self.profiles:
            self.current_profile = self.profiles[0]
            self.detector.load_model(self.current_profile.model_path)
        else:
            print("⚠️ Профили не загружены! Создайте .json файлы в task_profiles/")

        self.current_model_name = "Стандартная модель"
        self.current_model_path = DEFAULT_MODEL
        self.user_models = []

        self.load_models_config()
        self.setup_ui()

        # Горячие клавиши
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
        self.root.bind('<Escape>', lambda e: self.exit_fullscreen())

        self.start_camera()

    def setup_ui(self):
        # === МЕНЮ КАК В IDE ===
        menubar = tk.Menu(
            self.root,
            bg=BG_PANEL,
            fg=TEXT_WHITE,
            activebackground=ACCENT_GREEN,
            activeforeground='#000000',
            font=('Segoe UI', 10),
            tearoff=0
        )
        self.root.config(menu=menubar)

        # Меню File
        file_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=BG_SIDEBAR,
            fg=TEXT_WHITE,
            activebackground=ACCENT_GREEN,
            activeforeground='#000000',
            font=('Segoe UI', 10)
        )
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="▶ Запустить детекцию", command=self.start_detection)
        file_menu.add_command(label="■ Остановить детекцию", command=self.stop_detection)
        file_menu.add_separator()
        file_menu.add_command(label="⬤ Начать запись", command=self.start_recording_menu)
        file_menu.add_command(label="⬛ Остановить запись", command=self.stop_recording_menu)
        file_menu.add_separator()
        file_menu.add_command(
            label="📁 Выбрать папку для сохранения видео",
            command=self.choose_recording_path
        )
        file_menu.add_separator()
        file_menu.add_command(label="📷 Сделать снимок", command=self.take_screenshot)
        file_menu.add_separator()
        file_menu.add_command(label="🔄 Сбросить статистику", command=self.reset_stats)
        file_menu.add_separator()
        file_menu.add_command(label="🚪 Выход", command=self.on_closing)

        # Меню View
        view_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=BG_SIDEBAR,
            fg=TEXT_WHITE,
            activebackground=ACCENT_GREEN,
            activeforeground='#000000',
            font=('Segoe UI', 10)
        )
        menubar.add_cascade(label="View", menu=view_menu)

        # Подменю выбора профиля
        profile_submenu = tk.Menu(
            view_menu,
            tearoff=0,
            bg=BG_SIDEBAR,
            fg=TEXT_WHITE,
            activebackground=ACCENT_GREEN,
            activeforeground='#000000',
            font=('Segoe UI', 10)
        )
        view_menu.add_cascade(label="📋 Выбрать профиль", menu=profile_submenu)

        if self.profiles:
            for profile in self.profiles:
                profile_submenu.add_command(
                    label=profile.task_name,
                    command=lambda p=profile.task_name: self.switch_profile(p)
                )
        else:
            profile_submenu.add_command(
                label="⚠ Нет доступных профилей",
                state=tk.DISABLED
            )

        view_menu.add_separator()

        # Подменю загрузки моделей
        model_submenu = tk.Menu(
            view_menu,
            tearoff=0,
            bg=BG_SIDEBAR,
            fg=TEXT_WHITE,
            activebackground=ACCENT_GREEN,
            activeforeground='#000000',
            font=('Segoe UI', 10)
        )
        view_menu.add_cascade(label="📦 Загруженные модели", menu=model_submenu)

        model_submenu.add_command(
            label="➕ Загрузить новую модель",
            command=self.upload_model
        )
        model_submenu.add_separator()
        model_submenu.add_command(
            label="🔍 Стандартная модель (Штрихкоды)",
            command=lambda: self.load_standard_model()
        )

        for model in self.user_models:
            model_submenu.add_command(
                label=f"📦 {model['name']}",
                command=lambda m=model: self.load_user_model(m)
            )

        view_menu.add_separator()
        view_menu.add_command(
            label="🖥️ Полноэкранный режим (F11)",
            command=self.toggle_fullscreen
        )

        # Меню Help
        help_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=BG_SIDEBAR,
            fg=TEXT_WHITE,
            activebackground=ACCENT_GREEN,
            activeforeground='#000000',
            font=('Segoe UI', 10)
        )
        menubar.add_cascade(label="Help", menu=help_menu)

        help_menu.add_command(label="📖 Документация", command=self.show_documentation)
        help_menu.add_command(label="💡 Как использовать", command=self.show_usage_guide)
        help_menu.add_separator()
        help_menu.add_command(label="⌨️ Горячие клавиши", command=self.show_hotkeys)
        help_menu.add_separator()
        help_menu.add_command(label="ℹ️ О программе", command=self.show_about)

        # === ВЕРХНЯЯ ПАНЕЛЬ ===
        self.top_panel = tk.Frame(
            self.root,
            bg=BG_PANEL,
            height=70,
            relief=tk.FLAT,
            bd=0
        )
        self.top_panel.pack(fill=tk.X, padx=0, pady=0)
        self.top_panel.pack_propagate(False)

        # Заголовок по центру
        title_frame = tk.Frame(self.top_panel, bg=BG_PANEL)
        title_frame.pack(side=tk.LEFT, expand=True, padx=20, pady=15)

        tk.Label(
            title_frame,
            text='QualiVision',
            font=('Segoe UI', 20, 'bold'),
            bg=BG_PANEL,
            fg=ACCENT_GREEN
        ).pack()

        # Правая часть - ЛОГОТИП
        right_controls = tk.Frame(self.top_panel, bg=BG_PANEL)
        right_controls.pack(side=tk.RIGHT, padx=20, pady=10)

        try:
            # Загружаем логотип из указанного пути
            logo_img = Image.open(LOGO_PATH)
            # Масштабируем до высоты 50 пикселей, сохраняя пропорции
            ratio = 50.0 / logo_img.height
            new_width = int(logo_img.width * ratio)
            logo_img = logo_img.resize((new_width, 50), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            tk.Label(right_controls, image=self.logo_photo, bg=BG_PANEL).pack()
        except Exception:
            # Если логотип не найден, показываем заглушку текста
            tk.Label(
                right_controls,
                text="[ЛОГОТИП]",
                font=('Segoe UI', 14, 'bold'),
                bg=BG_PANEL,
                fg=TEXT_GRAY
            ).pack(pady=10)

        # === КОНТЕЙНЕР ДЛЯ ОСНОВНОГО КОНТЕНТА ===
        self.content_container = tk.Frame(self.root, bg=BG_MAIN)
        self.content_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=(10, 15))

        # === ПРАВАЯ ПАНЕЛЬ ИНФОРМАЦИИ (СКРЫТА ПО УМОЛЧАНИЮ) ===
        self.info_frame = tk.Frame(
            self.content_container,
            bg=BG_SIDEBAR,
            bd=0,
            relief=tk.FLAT,
            highlightthickness=2,
            highlightbackground=ACCENT_GREEN
        )

        # === КНОПКА-СТРЕЛКА ДЛЯ ПЕРЕКЛЮЧЕНИЯ ПАНЕЛИ ===
        self.toggle_frame = tk.Frame(self.content_container, bg=BG_SIDEBAR, width=25)
        self.toggle_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.toggle_frame.pack_propagate(False)

        self.toggle_btn = tk.Button(
            self.toggle_frame,
            text="<",
            font=('Consolas', 14, 'bold'),
            bg=BG_SIDEBAR,
            fg=TEXT_WHITE,
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            command=self.toggle_info_panel,
            activebackground=ACCENT_GREEN,
            activeforeground='#000000'
        )
        self.toggle_btn.pack(expand=True, fill=tk.BOTH)

        # === КОНТЕЙНЕР КАМЕРЫ (Без зелёной окантовки) ===
        self.camera_container = tk.Frame(
            self.content_container,
            bg='#000000',
            bd=0,
            relief=tk.FLAT
        )
        self.camera_container.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.video_label = tk.Label(self.camera_container, bg='#000000')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # --- НАПОЛНЕНИЕ ПАНЕЛИ ИНФОРМАЦИИ ---

        # Заголовок панели
        info_header = tk.Frame(self.info_frame, bg=ACCENT_GREEN, height=50)
        info_header.pack(fill=tk.X)
        info_header.pack_propagate(False)

        tk.Label(
            info_header,
            text="ИНФОРМАЦИЯ",
            font=('Segoe UI', 12, 'bold'),
            bg=ACCENT_GREEN,
            fg='#000000'
        ).pack(pady=13)

        # Текущий профиль
        profile_section = tk.Frame(self.info_frame, bg=BG_SIDEBAR)
        profile_section.pack(fill=tk.X, padx=20, pady=(25, 15))

        tk.Label(
            profile_section,
            text="ТЕКУЩИЙ ПРОФИЛЬ",
            font=('Segoe UI', 9, 'bold'),
            bg=BG_SIDEBAR,
            fg=TEXT_GRAY
        ).pack(anchor=tk.W)

        profile_name = self.current_profile.task_name if self.current_profile else "Не выбран"
        self.model_display = tk.Label(
            profile_section,
            text=profile_name,
            font=('Segoe UI', 12, 'bold'),
            bg=BG_SIDEBAR,
            fg=ACCENT_GREEN,
            wraplength=330,
            justify=tk.LEFT
        )
        self.model_display.pack(anchor=tk.W, pady=(5, 0))

        # Разделитель
        tk.Frame(self.info_frame, bg=BORDER_COLOR, height=1).pack(fill=tk.X, padx=20, pady=15)

        # Статус детекции
        detection_section = tk.Frame(self.info_frame, bg=BG_SIDEBAR)
        detection_section.pack(fill=tk.X, padx=20, pady=15)

        tk.Label(
            detection_section,
            text="СТАТУС ДЕТЕКЦИИ",
            font=('Segoe UI', 9, 'bold'),
            bg=BG_SIDEBAR,
            fg=TEXT_GRAY
        ).pack(anchor=tk.W)

        self.detection_status = tk.Label(
            detection_section,
            text="▶ Детекция активна",
            font=('Segoe UI', 10),
            bg=BG_SIDEBAR,
            fg=ACCENT_GREEN
        )
        self.detection_status.pack(anchor=tk.W, pady=(5, 0))

        # Разделитель
        tk.Frame(self.info_frame, bg=BORDER_COLOR, height=1).pack(fill=tk.X, padx=20, pady=15)

        # Путь сохранения видео
        path_section = tk.Frame(self.info_frame, bg=BG_SIDEBAR)
        path_section.pack(fill=tk.X, padx=20, pady=15)

        tk.Label(
            path_section,
            text="ПАПКА ДЛЯ ЗАПИСИ",
            font=('Segoe UI', 9, 'bold'),
            bg=BG_SIDEBAR,
            fg=TEXT_GRAY
        ).pack(anchor=tk.W)

        self.path_display = tk.Label(
            path_section,
            text=self.recording_path,
            font=('Segoe UI', 9),
            bg=BG_SIDEBAR,
            fg=TEXT_WHITE,
            wraplength=330,
            justify=tk.LEFT
        )
        self.path_display.pack(anchor=tk.W, pady=(5, 0))

        # Разделитель
        tk.Frame(self.info_frame, bg=BORDER_COLOR, height=1).pack(fill=tk.X, padx=20, pady=15)

        # Статус записи
        recording_section = tk.Frame(self.info_frame, bg=BG_SIDEBAR)
        recording_section.pack(fill=tk.X, padx=20, pady=15)

        tk.Label(
            recording_section,
            text="СТАТУС ЗАПИСИ",
            font=('Segoe UI', 9, 'bold'),
            bg=BG_SIDEBAR,
            fg=TEXT_GRAY
        ).pack(anchor=tk.W)

        self.recording_status = tk.Label(
            recording_section,
            text="⚫ Запись не ведется",
            font=('Segoe UI', 10),
            bg=BG_SIDEBAR,
            fg=TEXT_WHITE
        )
        self.recording_status.pack(anchor=tk.W, pady=(5, 0))

        # Разделитель
        tk.Frame(self.info_frame, bg=BORDER_COLOR, height=1).pack(fill=tk.X, padx=20, pady=15)

        # Статистика
        stats_section = tk.Frame(self.info_frame, bg=BG_SIDEBAR)
        stats_section.pack(fill=tk.X, padx=20, pady=15)

        tk.Label(
            stats_section,
            text="СТАТИСТИКА",
            font=('Segoe UI', 9, 'bold'),
            bg=BG_SIDEBAR,
            fg=TEXT_GRAY
        ).pack(anchor=tk.W)

        # Всего объектов
        total_frame = tk.Frame(stats_section, bg=BG_SIDEBAR)
        total_frame.pack(fill=tk.X, pady=(10, 5))

        tk.Label(
            total_frame,
            text="Всего обнаружено:",
            font=('Segoe UI', 10),
            bg=BG_SIDEBAR,
            fg=TEXT_WHITE
        ).pack(side=tk.LEFT)

        self.stats_label = tk.Label(
            total_frame,
            text="0",
            font=('Segoe UI', 10, 'bold'),
            bg=BG_SIDEBAR,
            fg=ACCENT_GREEN
        )
        self.stats_label.pack(side=tk.RIGHT)

        # В зоне подсчёта
        zone_frame = tk.Frame(stats_section, bg=BG_SIDEBAR)
        zone_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            zone_frame,
            text="В зоне подсчета:",
            font=('Segoe UI', 10),
            bg=BG_SIDEBAR,
            fg=TEXT_WHITE
        ).pack(side=tk.LEFT)

        self.counter_label = tk.Label(
            zone_frame,
            text="0",
            font=('Segoe UI', 10, 'bold'),
            bg=BG_SIDEBAR,
            fg=ACCENT_GREEN
        )
        self.counter_label.pack(side=tk.RIGHT)

        # === НИЖНЯЯ ПАНЕЛЬ СТАТУСА ===
        self.bottom_panel = tk.Frame(
            self.root,
            bg=BG_PANEL,
            height=40,
            relief=tk.FLAT,
            bd=0
        )
        self.bottom_panel.pack(fill=tk.X, padx=0, pady=0)
        self.bottom_panel.pack_propagate(False)

        status_container = tk.Frame(self.bottom_panel, bg=BG_PANEL)
        status_container.pack(side=tk.LEFT, padx=20, pady=10)

        tk.Label(
            status_container,
            text="●",
            font=('Arial', 14),
            bg=BG_PANEL,
            fg=ACCENT_GREEN
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.status_label = tk.Label(
            status_container,
            text="Система готова к работе",
            font=('Segoe UI', 9),
            bg=BG_PANEL,
            fg=TEXT_WHITE
        )
        self.status_label.pack(side=tk.LEFT)

    # === ПОЛНОЭКРАННЫЙ РЕЖИМ ===

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)

        if self.is_fullscreen:
            # Скрываем все лишние панели
            self.top_panel.pack_forget()
            self.bottom_panel.pack_forget()
            self.toggle_frame.pack_forget()

            if self.info_panel_visible:
                self.info_frame.pack_forget()
                self.info_panel_visible = False
                self.toggle_btn.configure(text="<")

            self.content_container.pack_configure(padx=0, pady=0)
            self.status_label.configure(text="Полноэкранный режим (ESC для выхода)")
        else:
            # Восстанавливаем интерфейс
            self.top_panel.pack(fill=tk.X, padx=0, pady=0, before=self.content_container)
            self.bottom_panel.pack(fill=tk.X, padx=0, pady=0)

            # Возвращаем стрелочку на место
            self.toggle_frame.pack(side=tk.RIGHT, fill=tk.Y)

            self.content_container.pack_configure(padx=20, pady=(10, 15))
            self.status_label.configure(text="Обычный режим")

    def exit_fullscreen(self):
        if self.is_fullscreen:
            self.toggle_fullscreen()

    # === ПАНЕЛЬ INFO ===

    def toggle_info_panel(self):
        if self.is_fullscreen:
            messagebox.showinfo("Информация", "Панель информации недоступна в полноэкранном режиме")
            return

        if self.info_panel_visible:
            # Закрываем панель
            self.info_frame.pack_forget()
            self.info_panel_visible = False
            self.toggle_btn.configure(text="<")
            self.status_label.configure(text="Панель информации скрыта")
        else:
            # Открываем панель. Динамически перепаковываем элементы
            self.toggle_frame.pack_forget()  # Убираем стрелочку

            self.info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))  # Панель в самый правый край
            self.info_frame.config(width=370)
            self.info_frame.pack_propagate(False)

            self.toggle_frame.pack(side=tk.RIGHT, fill=tk.Y)  # Возвращаем стрелочку теперь левее панели

            self.info_panel_visible = True
            self.toggle_btn.configure(text=">")
            self.status_label.configure(text="Панель информации открыта")

    # === ДЕТЕКЦИЯ ===

    def start_detection(self):
        if not self.is_detecting:
            self.is_detecting = True
            self.detection_status.configure(text="▶ Детекция активна", fg=ACCENT_GREEN)
            self.status_label.configure(text="Детекция запущена")
            messagebox.showinfo("Детекция", "Детекция объектов запущена")

    def stop_detection(self):
        if self.is_detecting:
            self.is_detecting = False
            self.detection_status.configure(text="■ Детекция остановлена", fg=ERROR_COLOR)
            self.status_label.configure(text="Детекция остановлена")
            messagebox.showinfo("Детекция", "Детекция объектов остановлена")

    # === FILE: ЗАПИСЬ И СКРИНШОТЫ ===

    def choose_recording_path(self):
        path = filedialog.askdirectory(
            title="Выберите папку для сохранения видео",
            initialdir=self.recording_path
        )
        if path:
            self.recording_path = path
            self.path_display.configure(text=path)
            self.status_label.configure(text=f"Папка для записи: {path}")
            messagebox.showinfo("Путь сохранения", f"Видео будут сохраняться в:\n{path}")

    def start_recording_menu(self):
        if not self.is_recording:
            self.is_recording = True
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"recording_{timestamp}.mp4"
            filepath = os.path.join(self.recording_path, filename)

            self.video_writer = cv2.VideoWriter(
                filepath,
                fourcc,
                FPS,
                (FRAME_WIDTH, FRAME_HEIGHT)
            )
            self.recording_status.configure(text=f"🔴 Запись: {filename}", fg=ERROR_COLOR)
            self.status_label.configure(text=f"Запись видео начата: {filename}")
            messagebox.showinfo("Запись", f"Запись видео начата\nФайл: {filepath}")

    def stop_recording_menu(self):
        if self.is_recording:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording_status.configure(text="⚫ Запись не ведется", fg=TEXT_WHITE)
            self.status_label.configure(text="Запись остановлена")
            messagebox.showinfo("Запись", "Запись видео остановлена и сохранена")
        else:
            messagebox.showwarning("Предупреждение", "Запись не ведется")

    def reset_stats(self):
        self.counter.reset()
        self.screw_counter.reset()
        self.stats_label.configure(text="0")
        self.counter_label.configure(text="0")
        print("🔄 Счётчик сброшен")

    def take_screenshot(self):
        if self.last_frame is not None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"screenshot_{timestamp}.jpg"
            filepath = os.path.join(self.recording_path, filename)
            cv2.imwrite(filepath, self.last_frame)
            self.status_label.configure(text=f"Снимок сохранен: {filename}")
            messagebox.showinfo("Снимок", f"Снимок сохранен:\n{filepath}")
        else:
            messagebox.showwarning("Ошибка", "Нет доступного кадра для сохранения")

    # === VIEW: МОДЕЛИ И ПРОФИЛИ ===

    def load_standard_model(self):
        self.current_model_path = DEFAULT_MODEL
        self.current_model_name = "Стандартная модель"
        self.detector.load_model(DEFAULT_MODEL)
        if self.current_profile:
            self.model_display.configure(text=self.current_profile.task_name)
        messagebox.showinfo("Модель", "Переключено на стандартную модель")

    def load_user_model(self, model):
        self.current_model_path = model['path']
        self.current_model_name = model['name']
        self.detector.load_model(model['path'])
        if self.current_profile:
            self.model_display.configure(text=f"{self.current_profile.task_name} ({model['name']})")
        messagebox.showinfo("Модель", f"Загружена модель: {model['name']}")

    def switch_profile(self, selected_name):
        for profile in self.profiles:
            if profile.task_name == selected_name:
                self.current_profile = profile
                self.detector.load_model(profile.model_path)
                self.model_display.configure(text=profile.task_name)
                self.reset_stats()
                self.status_label.configure(text=f"Активирован профиль: {profile.task_name}")
                break

    def upload_model(self):
        path = filedialog.askopenfilename(
            title="Выберите YOLO модель",
            filetypes=[("PyTorch", "*.pt"), ("Все", "*.*")]
        )
        if not path:
            return
        name = Path(path).stem
        new_path = os.path.join(MODELS_FOLDER, f"{name}.pt")

        try:
            shutil.copy2(path, new_path)
        except shutil.SameFileError:
            pass

        self.user_models.append({'name': name, 'path': new_path})
        self.save_models_config()
        messagebox.showinfo("Успешно", f"Модель '{name}' загружена!")

    # === HELP ===

    def show_documentation(self):
        doc_text = """
QualiVision - Документация

ОСНОВНЫЕ ВОЗМОЖНОСТИ:
• Детекция объектов в режиме реального времени
• Подсчет объектов в заданной зоне
• Сканирование штрихкодов (для соответствующих профилей)
• Запись видео с детекцией
• Создание скриншотов
• Полноэкранный режим
• Выбор папки для сохранения видео

ПРОФИЛИ ЗАДАЧ:
Система поддерживает различные профили для разных задач.
Каждый профиль настраивается через JSON-файлы в папке task_profiles/

УПРАВЛЕНИЕ:
File  - детекция, запись, снимки, путь сохранения
View  - профили, модели, полноэкранный режим
Help  - документация и справка
"""
        messagebox.showinfo("Документация", doc_text)

    def show_usage_guide(self):
        guide_text = """
КАК ИСПОЛЬЗОВАТЬ QualiVision:

1. Выбор профиля:
   View → Выбрать профиль → нужная задача

2. Папка для записи:
   File → Выбрать папку для сохранения видео

3. Детекция:
   File → Запустить детекцию / Остановить детекцию

4. Запись:
   File → Начать запись / Остановить запись

5. Скриншот:
   File → Сделать снимок

6. Статистика:
   Нажмите на стрелочку "<" справа экрана

7. Полноэкранный режим:
   View → Полноэкранный режим или F11 (ESC — выход)

8. Сброс статистики:
   File → Сбросить статистику

9. Пользовательские модели:
   View → Загруженные модели → Загрузить новую модель
"""
        messagebox.showinfo("Руководство", guide_text)

    def show_hotkeys(self):
        hotkeys_text = """
ГОРЯЧИЕ КЛАВИШИ:

F11  - Переключить полноэкранный режим
ESC  - Выйти из полноэкранного режима

(Дополнительные горячие клавиши в разработке)
"""
        messagebox.showinfo("Горячие клавиши", hotkeys_text)

    def show_about(self):
        about_text = """
QualiVision
Интеллектуальная система визуального контроля

Версия: 1.0.0
Разработчик: Ваше имя/компания

ТЕХНОЛОГИИ:
• YOLOv8 для детекции объектов
• ByteTrack (или аналогичный трекер) для отслеживания
• pyzbar (при необходимости) для сканирования штрихкодов
• OpenCV для обработки видео
• Tkinter для интерфейса

© 2026 Все права защищены
"""
        messagebox.showinfo("О программе", about_text)

    # === КОНФИГ МОДЕЛЕЙ ===

    def save_models_config(self):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.user_models, f, ensure_ascii=False, indent=2)

    def load_models_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                self.user_models = json.load(f)

    # === КАМЕРА И ВИДЕОПОТОК ===

    def start_camera(self):
        self.camera_running = True
        self.cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.status_label.configure(text="Ошибка: Камера не найдена")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

        threading.Thread(target=self.video_loop, daemon=True).start()
        self.update_display()

    def video_loop(self):
        prev_time = time.time()

        while self.camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            if self.is_detecting and self.current_profile:
                profile = self.current_profile
                display_frame = frame.copy()

                # Зона подсчёта
                if profile.zone_enabled:
                    counting_zone = get_counting_zone_coordinates(
                        FRAME_WIDTH,
                        FRAME_HEIGHT,
                        profile.zone_x_start,
                        profile.zone_x_end
                    )
                    display_frame, _ = draw_counting_zone(
                        display_frame,
                        profile.zone_x_start,
                        profile.zone_x_end,
                        profile.zone_color,
                        COUNTING_ZONE_ALPHA
                    )
                else:
                    counting_zone = (0, 0, FRAME_WIDTH, FRAME_HEIGHT)

                # Предсказание
                results = self.detector.predict(
                    frame,
                    conf=profile.confidence,
                    iou=profile.iou
                )

                boxes_raw = []
                if results[0].boxes is not None:
                    for det in results[0].boxes:
                        if profile.target_class is None or int(det.cls) == profile.target_class:
                            boxes_raw.append(det.xyxy[0].cpu().numpy())

                current_in_zone = sum(
                    1 for b in boxes_raw if is_in_counting_zone(b, counting_zone)
                )

                tracker_ids, boxes = self.detector.process_detections(
                    results,
                    profile,
                    previous_count=(
                            self.counter.total_unique_count +
                            self.screw_counter.total_count
                    ),
                    current_count=current_in_zone
                )

                barcode_data = None
                if profile.use_barcode_reading:
                    barcode_data = []
                    for box in boxes:
                        barcode = self.detector.get_barcode_data(frame, box)
                        barcode_data.append(barcode)

                display_frame = draw_boxes_universal(
                    display_frame,
                    boxes,
                    tracker_ids,
                    profile,
                    counting_zone,
                    barcode_data
                )

                # Обновление счётчиков
                if profile.use_barcode_reading:
                    for i, (tid, box) in enumerate(zip(tracker_ids, boxes)):
                        data = barcode_data[i] if barcode_data and i < len(barcode_data) else None
                        self.counter.update(tid, data, box, counting_zone)
                    total = self.counter.total_unique_count
                else:
                    total = self.screw_counter.update_count(
                        tracker_ids,
                        boxes,
                        counting_zone
                    )

                self.root.after(
                    0,
                    lambda t=total: self.stats_label.configure(text=f"{t}")
                )
                self.root.after(
                    0,
                    lambda c=current_in_zone: self.counter_label.configure(text=f"{c}")
                )

                # FPS
                current_time = time.time()
                fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                display_frame = draw_fps(display_frame, fps)
                prev_time = current_time

                frame_to_show = display_frame
            else:
                frame_to_show = frame
                prev_time = time.time()

            with self.frame_lock:
                self.last_frame = frame_to_show
                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame_to_show)

            time.sleep(0.001)

    def update_display(self):
        if self.camera_running:
            with self.frame_lock:
                if self.last_frame is not None:
                    rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)

                    if self.is_fullscreen:
                        width = self.root.winfo_screenwidth()
                        height = self.root.winfo_screenheight()
                    else:
                        # Адаптивное масштабирование учитывает ширину кнопки (25px) и панели (370px)
                        width = 960 if self.info_panel_visible else 1330
                        height = 750

                    img = Image.fromarray(rgb).resize(
                        (width, height),
                        Image.Resampling.LANCZOS
                    )
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
            self.root.after(30, self.update_display)

    # === ЖИЗНЕННЫЙ ЦИКЛ ПРИЛОЖЕНИЯ ===

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.camera_running = False
        self.save_models_config()
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        self.root.destroy()


if __name__ == "__main__":
    app = YOLOApp()
    app.run()
