import datetime
import sys
import os
import time
import json
import logging
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QGridLayout,
                             QLineEdit, QVBoxLayout, QHBoxLayout, QGroupBox, QFileDialog,
                             QMessageBox, QCheckBox, QSizePolicy, QRadioButton, QSpinBox)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QDateTime, QTimer, QUrl
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from cloud_functions import predict_image, load_model
import cv2
import numpy as np

SETTINGS_FILE = "cloud_watcher_settings.json"

class PredictionThread(QThread):
    prediction_signal = pyqtSignal(str, QImage)
    error_signal = pyqtSignal(str)

    def __init__(self, file_path, model_path, image_size, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.model_path = model_path
        self.image_size = image_size
        self.running = True
        self.model = None

    def stop(self):
        self.running = False

    def run(self):
        self.model = load_model(self.model_path)
        if self.model is None:
            self.error_signal.emit(f"Failed to load model from {self.model_path}")
            return

        previous_mtime = None
        while self.running:
            if not os.path.exists(self.file_path):
                time.sleep(1)
                continue

            try:
                current_mtime = os.path.getmtime(self.file_path)
            except OSError as e:
                self.error_signal.emit(f"OS Error accessing file: {e}")
                time.sleep(5)
                continue

            if current_mtime != previous_mtime:
                try:
                    prediction = predict_image(self.model, self.file_path, self.image_size)
                    if prediction is not None:
                        img = cv2.imread(self.file_path)
                        if img is not None:
                            if len(img.shape) == 2:
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            else:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                h, w, ch = img.shape
                                bytesPerLine = ch * w
                                convert_to_qt_format = QImage(img.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
                            if prediction['label'] == 'No Clouds':
                                prediction['value'] = 1 - prediction['value']
                            self.prediction_signal.emit(f"Prediction: {prediction['label']} (Value: {prediction['value']:.4f})", convert_to_qt_format)
                        else:
                            self.error_signal.emit("Could not read image for display.")
                    else:
                        self.error_signal.emit("Prediction failed.")
                except Exception as e:
                    self.error_signal.emit(f"Prediction Error: {e}")
                previous_mtime = current_mtime
            time.sleep(1)


class WebpagePredictionThread(QThread):
    prediction_signal = pyqtSignal(str, QImage)
    error_signal = pyqtSignal(str)

    def __init__(self, url, model_path, image_size, update_frequency, parent=None):
        super().__init__(parent)
        self.url = url
        self.model_path = model_path
        self.image_size = image_size
        self.running = True
        self.model = None
        self.network_manager = QNetworkAccessManager()
        self.network_manager.finished.connect(self.handle_network_reply)
        self.update_frequency = update_frequency
        self.timer = QTimer()
        self.timer.timeout.connect(self.fetch_image)
        self.timer.start(self.update_frequency * 1000)

    def stop(self):
        self.running = False
        self.timer.stop()

    def fetch_image(self):
        if self.running:
            url = QUrl(self.url)
            request = QNetworkRequest(url)
            self.network_manager.get(request)

    def run(self):
        self.model = load_model(self.model_path)
        if self.model is None:
            self.error_signal.emit(f"Failed to load model from {self.model_path}")
            return
        #No loop needed anymore since the QTimer handles the calls now

    def handle_network_reply(self, reply):
        if reply.error() == QNetworkReply.NetworkError.NoError:
            image_data = reply.readAll()
            pixmap = QPixmap()
            if pixmap.loadFromData(image_data):
                image = pixmap.toImage()
                image_bytes = image.bits().asstring(image.sizeInBytes())
                img = np.frombuffer(image_bytes, dtype=np.uint8).reshape(image.height(), image.width(), 4)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                prediction = predict_image(self.model, img, self.image_size)
                if prediction is not None:
                    if prediction['label'] == 'No Clouds':
                        prediction['value'] = 1 - prediction['value']
                    self.prediction_signal.emit(f"Prediction: {prediction['label']} (Value: {prediction['value']:.4f})", image)
                else:
                    self.error_signal.emit("Prediction failed.")
            else:
                self.error_signal.emit("Error loading image from network reply.")
        else:
            self.error_signal.emit(f"Network error: {reply.errorString()}")


class CloudWatcherGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cloud Watcher")
        self.settings = {}
        self.load_settings()

        self.file_path = self.settings.get("file_path")
        self.model_path = self.settings.get("model_path")
        self.results_file_path = self.settings.get("results_file_path", "cloud_detection_results.txt")
        self.update_frequency = self.settings.get("update_frequency", 5)
        self.cloud_threshold = self.settings.get("cloud_threshold", 3)
        self.no_cloud_threshold = self.settings.get("no_cloud_threshold", 3)
        self.image_size = self.settings.get("image_size", 256)
        self.url = self.settings.get("url")

        self.prediction_thread = None
        self.consecutive_cloud_count = 0
        self.consecutive_no_cloud_count = 0

        self.watching = False
        self.results_file = None
        self.results_file_label = None
        self.current_state = "Unsafe"

        self.initUI()

        if self.file_path:
            self.file_label.setText(os.path.basename(self.file_path))
        if self.url:
            self.webpage_url_input.setText(str(self.url))

    def initUI(self):
        main_layout = QGridLayout()

        # Model and Results File Path
        results_group = QGroupBox("Model and Results File")
        results_layout = QGridLayout()
        self.results_button = QPushButton("Select Results File")
        self.results_button.clicked.connect(self.select_results_file)
        self.results_file_label = QLabel("No file selected")
        if self.results_file_path:
            self.results_file_label.setText(os.path.basename(self.results_file_path))
        self.model_button = QPushButton("Select Model")
        self.model_button.clicked.connect(self.select_model)
        self.model_label = QLabel("No model loaded")
        results_layout.addWidget(self.model_button, 0, 0)
        results_layout.addWidget(self.model_label, 0, 1)
        results_layout.addWidget(self.results_button, 1, 0)
        results_layout.addWidget(self.results_file_label, 1, 1)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group, 0, 0, 1, 2)

        # Radio Buttons for Watch Mode
        self.watch_file_radio = QRadioButton("Watch File")
        self.fetch_webpage_radio = QRadioButton("Fetch from Webpage")
        self.watch_file_radio.setChecked(True)  # Default selection

        radio_layout = QVBoxLayout()
        radio_layout.addWidget(self.watch_file_radio)
        radio_layout.addWidget(self.fetch_webpage_radio)

        radio_group_box = QGroupBox("Watch Mode")
        radio_group_box.setLayout(radio_layout)

        # File Watching Widgets (Grouped)
        self.file_group = QGroupBox("File Watcher")
        file_layout = QGridLayout()
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.file_button, 0, 0)
        file_layout.addWidget(self.file_label, 1, 0)
        self.file_group.setLayout(file_layout)

        # Webpage Fetching Widgets (Grouped)
        self.webpage_group = QGroupBox("Webpage Fetcher")
        webpage_layout = QGridLayout()
        self.webpage_url_label = QLabel("Webpage URL:")
        self.webpage_url_input = QLineEdit()
        self.update_frequency_label = QLabel("Freq. (s):")
        self.update_frequency_input = QSpinBox()
        self.update_frequency_input.setMinimum(1)
        self.update_frequency_input.setValue(self.update_frequency)
        self.update_frequency_input.valueChanged.connect(self.update_update_frequency)

        webpage_layout.addWidget(self.webpage_url_label, 0, 0)
        webpage_layout.addWidget(self.webpage_url_input, 0, 1)
        webpage_layout.addWidget(self.update_frequency_label, 1, 0)
        webpage_layout.addWidget(self.update_frequency_input, 1, 1)
        self.webpage_group.setLayout(webpage_layout)

        main_layout.addWidget(radio_group_box, 1, 0)
        main_layout.addWidget(self.file_group, 1,1)
        main_layout.addWidget(self.webpage_group, 1, 1)

        # Thresholds
        threshold_group = QGroupBox("Thresholds")
        threshold_layout = QGridLayout()
        self.cloud_threshold_edit = QLineEdit(str(self.cloud_threshold))
        self.no_cloud_threshold_edit = QLineEdit(str(self.no_cloud_threshold))
        threshold_layout.addWidget(QLabel("Cloud Threshold:"), 0, 0)
        threshold_layout.addWidget(self.cloud_threshold_edit, 0, 1)
        threshold_layout.addWidget(QLabel("No Cloud Threshold:"), 1, 0)
        threshold_layout.addWidget(self.no_cloud_threshold_edit, 1, 1)
        threshold_group.setLayout(threshold_layout)
        main_layout.addWidget(threshold_group, 2, 0)

        # Image Size
        image_size_group = QGroupBox("Image Size")
        image_size_layout = QHBoxLayout()
        self.image_size_edit = QLineEdit(str(self.image_size))
        image_size_layout.addWidget(QLabel("Image Size:"))
        image_size_layout.addWidget(self.image_size_edit)
        image_size_group.setLayout(image_size_layout)
        main_layout.addWidget(image_size_group, 2, 1)

        # Current State and Count
        state_group = QGroupBox("Current State")
        state_layout = QVBoxLayout()
        self.state_label = QLabel(f"State: {self.current_state}")  # Define state_label here
        self.cloud_count_label = QLabel(f"Cloud Count: {self.consecutive_cloud_count}")
        self.no_cloud_count_label = QLabel(f"No Cloud Count: {self.consecutive_no_cloud_count}")
        state_layout.addWidget(self.state_label)
        state_layout.addWidget(self.cloud_count_label)
        state_layout.addWidget(self.no_cloud_count_label)
        state_group.setLayout(state_layout)
        main_layout.addWidget(state_group, 3, 0, 1, 2)

        # Start/Stop Buttons
        self.watch_button = QPushButton("Start Watching")
        self.watch_button.clicked.connect(self.toggle_watching)
        main_layout.addWidget(self.watch_button, 4, 0, 1, 2)

        # Image Display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Make image label expandable
        main_layout.addWidget(self.image_label, 5, 0, 1, 2)

        # Prediction Text
        self.prediction_label = QLabel("No predictions yet.")
        main_layout.addWidget(self.prediction_label)

        self.setLayout(main_layout)
        if self.model_path:
            self.model_label.setText(os.path.basename(self.model_path))
        self.cloud_threshold_edit.setText(str(self.cloud_threshold))
        self.no_cloud_threshold_edit.setText(str(self.no_cloud_threshold))
        self.image_size_edit.setText(str(self.image_size))

        self.watch_file_radio.toggled.connect(self.update_watch_mode)
        self.fetch_webpage_radio.toggled.connect(self.update_watch_mode)
        self.update_watch_mode()

    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File to Watch", "", "All Files (*)")
        if file_name:
            self.file_path = file_name
            self.save_settings()
            print(f"Selected file: {self.file_path}")  # Print selected file path

    def update_update_frequency(self, value):
        self.update_frequency = value
        self.save_settings()

    def select_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "H5 Files (*.h5);Keras Files (*.keras);;All Files (*)")
        if file_name:
            self.model_path = file_name
            self.model_label.setText(os.path.basename(self.model_path))
            self.save_settings()
            print(f"Selected model: {self.model_path}")  # Print selected model path

    def update_watch_mode(self):
        self.file_group.setVisible(self.watch_file_radio.isChecked())
        self.webpage_group.setVisible(self.fetch_webpage_radio.isChecked())
        if self.watching:
            self.toggle_watching() # stop watching when switching modes

    def start_watching(self):
        if self.watch_file_radio.isChecked():
            if not self.file_path:
                QMessageBox.warning(self, "Error", "Please select a file to watch.")
                self.toggle_watching()  # Reset button state
                return
        elif self.fetch_webpage_radio.isChecked():
            if not self.webpage_url_input.text():
                QMessageBox.warning(self, "Error", "Please enter a webpage URL.")
                self.toggle_watching()  # Reset button state
                return

        if not self.model_path:
            QMessageBox.warning(self, "Error", "Please select a model file.")
            self.toggle_watching()  # Reset button state
            return

        try:
            self.cloud_threshold = int(self.cloud_threshold_edit.text())
            self.no_cloud_threshold = int(self.no_cloud_threshold_edit.text())
            self.image_size = int(self.image_size_edit.text())
            self.save_settings()
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid threshold or image size value.")
            self.toggle_watching()  # Reset button state
            return

        if self.watch_file_radio.isChecked():
            self.prediction_thread = PredictionThread(self.file_path, self.model_path, self.image_size, self)
        elif self.fetch_webpage_radio.isChecked():
            self.url = self.webpage_url_input.text()
            self.save_settings()
            update_frequency = self.update_frequency_input.value()
            self.prediction_thread = WebpagePredictionThread(self.url, self.model_path, self.image_size, update_frequency, self)

        self.prediction_thread.prediction_signal.connect(self.update_prediction)
        self.prediction_thread.error_signal.connect(self.display_error)
        self.prediction_thread.start()

    def stop_watching(self):
        try:
            if self.prediction_thread:
                self.prediction_thread.stop()
                self.prediction_thread.wait()
                self.prediction_thread = None
        except Exception as e:
            print(e)
        self.watch_button.setText("Start Watching")
        self.watch_button.setStyleSheet("")

    def toggle_watching(self):
        self.watching = not self.watching

        if self.watching:
            self.start_watching()
            self.watch_button.setText("Stop Watching")
            self.watch_button.setStyleSheet("color: red;")  # Corrected: Using style sheet
        else:
            self.stop_watching()

    def select_results_file(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Select Results File", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            self.results_file_path = file_name
            self.results_file_label.setText(os.path.basename(self.results_file_path))
            self.save_settings()
            print(f"Selected results file: {self.results_file_path}")

    def update_prediction(self, prediction_text, image):
        self.prediction_label.setText(prediction_text)
        pixmap = QPixmap.fromImage(image).scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

        timestamp = datetime.datetime.now().strftime("%d%m%y_%H%M: ") # Get timestamp
        print(timestamp + prediction_text)

        if "No Clouds" in prediction_text:
            self.consecutive_no_cloud_count += 1
            if self.consecutive_no_cloud_count >= self.no_cloud_threshold:
                self.current_state = "Safe"
            if self.current_state == "Safe":
                self.consecutive_cloud_count = 0  # Reset cloud count
        else:
            self.consecutive_cloud_count += 1
            if self.consecutive_cloud_count >= self.cloud_threshold:
                self.current_state = "Unsafe"
            if self.current_state == "Unsafe":
                self.consecutive_no_cloud_count = 0  # Reset no-cloud count

        self.state_label.setText(f"State: {self.current_state}")
        self.cloud_count_label.setText(f"Cloud Count: {self.consecutive_cloud_count}")
        self.no_cloud_count_label.setText(f"No Cloud Count: {self.consecutive_no_cloud_count}")

        consecutive_text = ""
        if self.current_state == "Unsafe":
            consecutive_text = f"(Consecutive Clouds: {self.consecutive_cloud_count})"
        else:
            consecutive_text = f"(Consecutive No Clouds: {self.consecutive_no_cloud_count})"
        self.write_to_results(f"{self.current_state}\n")

    def write_to_results(self, text):
        if self.results_file_path:
            with open(self.results_file_path, "w") as outfile:
                outfile.write(f"{text}\n")

    def display_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.stop_watching()

    def save_settings(self):
        self.settings["file_path"] = self.file_path
        self.settings["model_path"] = self.model_path
        self.settings["cloud_threshold"] = self.cloud_threshold
        self.settings["no_cloud_threshold"] = self.no_cloud_threshold
        self.settings["image_size"] = self.image_size
        self.settings["results_file_path"] = self.results_file_path
        self.settings["update_frequency"] = self.update_frequency
        self.settings["url"] = self.url

        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(self.settings, f)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_settings(self):
        try:
            with open(SETTINGS_FILE, "r") as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            pass  # Use default settings if the file doesn't exist
        except json.JSONDecodeError:
            print("Error decoding settings file. Using default settings.")
            # Optionally, you might want to create a new empty settings file here:
            # with open(SETTINGS_FILE, "w") as f:
            #     json.dump({}, f)
        except Exception as e:
            print(f"Error loading settings: {e}")

    def closeEvent(self, event):  # Signal handler for window close event
        self.stop_watching()  # Stop the thread before closing
        event.accept()  # Accept the close event

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CloudWatcherGUI()
    gui.show()
    sys.exit(app.exec())