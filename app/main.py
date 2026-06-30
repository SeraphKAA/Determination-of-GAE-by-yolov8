import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
PREVIEW_MAX_SIZE = (480, 480)
CYRILLIC_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

MODELS = {
    "gender": {
        "path": APP_DIR / "best_gender.pt",
        "title": "Пол",
        "color": (0, 140, 255),
    },
    "age": {
        "path": APP_DIR / "best_age.pt",
        "title": "Возраст",
        "color": (40, 180, 99),
    },
    "ethnos": {
        "path": APP_DIR / "best_ethnos.pt",
        "title": "Этнос",
        "color": (220, 80, 80),
    },
}



def validate_image_path(source_path: Path) -> None:
    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Поддерживаются только JPG, JPEG и PNG.")


def resize_for_preview(image: np.ndarray, max_size: tuple[int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    max_width, max_height = max_size
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def numpy_bgr_to_qpixmap(image: np.ndarray, max_size: tuple[int, int]) -> QPixmap:
    rgb_image = cv2.cvtColor(resize_for_preview(image, max_size), cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_image.shape
    bytes_per_line = channels * width
    qimage = QImage(
        rgb_image.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(qimage.copy())


def _load_cyrillic_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in CYRILLIC_FONT_PATHS:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def draw_detections(base_image: np.ndarray, enabled_models: dict[str, YOLO]) -> tuple[np.ndarray, list[str]]:
    result_image = base_image.copy()
    text_lines: list[str] = []
    labels_to_draw: list[tuple[str, int, int, tuple[int, int, int]]] = []
    font = _load_cyrillic_font(16)

    for key, model in enabled_models.items():
        config = MODELS[key]
        results = model(
            base_image,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            agnostic_nms=True,
            verbose=False,
        )
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            text_lines.append(f"{config['title']}: объекты не найдены")
            continue

        model_lines: list[str] = []
        color = config["color"]

        for index, box in enumerate(boxes, start=1):
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            label = f"{config['title']}: {class_name} ({confidence * 100:.1f}%)"
            if len(boxes) > 1:
                model_lines.append(f"  {index}. {class_name} ({confidence * 100:.1f}%)")
            else:
                model_lines.append(f"{config['title']}: {class_name} ({confidence * 100:.1f}%)")

            text_bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = x1
            text_y = int(max(y1 - text_height - 10, 4))
            cv2.rectangle(
                result_image,
                (text_x, text_y),
                (text_x + text_width + 6, text_y + text_height + 6),
                color,
                -1,
            )
            labels_to_draw.append((label, text_x + 3, text_y + 2, color))

        if len(boxes) > 1:
            text_lines.append(f"{config['title']} ({len(boxes)}):")
            text_lines.extend(model_lines)
        else:
            text_lines.extend(model_lines)

    if labels_to_draw:
        pil_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        for label, text_x, text_y, _ in labels_to_draw:
            draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))
        result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return result_image, text_lines


class ModelLoaderThread(QThread):
    finished_loading = pyqtSignal(dict, list)

    def run(self) -> None:
        loaded: dict[str, YOLO] = {}
        missing: list[str] = []

        for key, config in MODELS.items():
            model_path = config["path"]
            if not model_path.exists():
                missing.append(model_path.name)
                continue
            loaded[key] = YOLO(str(model_path))

        self.finished_loading.emit(loaded, missing)


class DetectionThread(QThread):
    finished_detection = pyqtSignal(object, list)
    failed_detection = pyqtSignal(str)

    def __init__(self, image: np.ndarray, enabled_models: dict[str, YOLO]) -> None:
        super().__init__()
        self.image = image
        self.enabled_models = enabled_models

    def run(self) -> None:
        try:
            result_image, text_lines = draw_detections(self.image, self.enabled_models)
            self.finished_detection.emit(result_image, text_lines)
        except Exception as error:
            self.failed_detection.emit(str(error))


class ImagePanel(QGroupBox):
    def __init__(self, title: str, placeholder: str) -> None:
        super().__init__(title)
        self.placeholder = placeholder
        self._pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        self.label = QLabel(placeholder)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(320, 320)
        self.label.setStyleSheet("background-color: #f4f4f4; border: 1px solid #d0d0d0;")
        layout.addWidget(self.label)

    def set_image(self, image: np.ndarray) -> None:
        self._pixmap = numpy_bgr_to_qpixmap(image, PREVIEW_MAX_SIZE)
        self._update_label()

    def clear_image(self) -> None:
        self._pixmap = None
        self.label.setPixmap(QPixmap())
        self.label.setText(self.placeholder)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_label()

    def _update_label(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(scaled)
        self.label.setText("")


class DetectionApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Определение пола, возраста и этноса")
        self.setMinimumSize(980, 720)

        self.models: dict[str, YOLO] = {}
        self.original_image: np.ndarray | None = None
        self.selected_source_path: Path | None = None
        self.checkboxes: dict[str, QCheckBox] = {}
        self.is_busy = False
        self.detection_thread: DetectionThread | None = None

        self._build_ui()
        self._load_models()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        controls = QHBoxLayout()
        self.open_button = QPushButton("Выбрать изображение")
        self.open_button.clicked.connect(self.choose_image)
        controls.addWidget(self.open_button)

        controls.addWidget(QLabel("Показывать детекцию:"))
        for key, config in MODELS.items():
            checkbox = QCheckBox(config["title"])
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.refresh_detection)
            self.checkboxes[key] = checkbox
            controls.addWidget(checkbox)

        controls.addStretch()
        self.status_label = QLabel("Загрузка моделей...")
        controls.addWidget(self.status_label)
        root_layout.addLayout(controls)

        images_layout = QHBoxLayout()
        self.original_panel = ImagePanel("Исходное изображение", "Выберите JPG, JPEG или PNG")
        self.result_panel = ImagePanel("Результат детекции", "Результат появится после выбора файла")
        images_layout.addWidget(self.original_panel)
        images_layout.addWidget(self.result_panel)
        root_layout.addLayout(images_layout, stretch=1)

        results_group = QGroupBox("Текстовый результат")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(140)
        results_layout.addWidget(self.results_text)
        root_layout.addWidget(results_group)

    def _load_models(self) -> None:
        self.open_button.setEnabled(False)
        self.loader_thread = ModelLoaderThread()
        self.loader_thread.finished_loading.connect(self._on_models_loaded)
        self.loader_thread.start()

    def _on_models_loaded(self, loaded: dict[str, YOLO], missing: list[str]) -> None:
        self.models = loaded
        self.open_button.setEnabled(True)

        if missing:
            QMessageBox.warning(
                self,
                "Модели не найдены",
                "Не найдены файлы моделей:\n" + "\n".join(missing),
            )

        if self.models:
            self.status_label.setText("Модели загружены. Выберите изображение.")
        else:
            self.status_label.setText("Модели не загружены.")

    def choose_image(self) -> None:
        if self.is_busy:
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение",
            "",
            "Изображения (*.jpg *.jpeg *.png);;JPEG (*.jpg *.jpeg);;PNG (*.png)",
        )
        if not file_path:
            return

        try:
            source_path = Path(file_path)
            validate_image_path(source_path)
            image = cv2.imread(str(source_path))
            if image is None:
                raise ValueError("Не удалось прочитать выбранный файл.")

            self.selected_source_path = source_path
            self.original_image = image
            self.original_panel.set_image(image)
            self.status_label.setText(f"Файл: {source_path.name}")
            self.refresh_detection()
        except ValueError as error:
            QMessageBox.critical(self, "Ошибка", str(error))
        except OSError as error:
            QMessageBox.critical(self, "Ошибка", f"Не удалось обработать файл:\n{error}")

    def refresh_detection(self) -> None:
        if self.original_image is None or not self.models:
            return
        if self.is_busy:
            return

        enabled_models = {
            key: model
            for key, model in self.models.items()
            if self.checkboxes[key].isChecked()
        }

        if not enabled_models:
            self.result_panel.set_image(self.original_image)
            self.results_text.setPlainText("Включите хотя бы одну модель детекции.")
            return

        self.is_busy = True
        self.open_button.setEnabled(False)
        self.status_label.setText("Выполняется детекция...")

        self.detection_thread = DetectionThread(self.original_image.copy(), enabled_models)
        self.detection_thread.finished_detection.connect(self._on_detection_done)
        self.detection_thread.failed_detection.connect(self._on_detection_failed)
        self.detection_thread.finished.connect(self._on_detection_thread_finished)
        self.detection_thread.start()

    def _on_detection_done(self, result_image: np.ndarray, text_lines: list[str]) -> None:
        self.result_panel.set_image(result_image)
        self.results_text.setPlainText("\n".join(text_lines) if text_lines else "Нет данных.")
        if self.selected_source_path:
            self.status_label.setText(f"Готово: {self.selected_source_path.name}")

    def _on_detection_failed(self, error_message: str) -> None:
        self.status_label.setText("Ошибка детекции")
        QMessageBox.critical(self, "Ошибка детекции", error_message)

    def _on_detection_thread_finished(self) -> None:
        self.is_busy = False
        self.open_button.setEnabled(True)


def main() -> None:
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
