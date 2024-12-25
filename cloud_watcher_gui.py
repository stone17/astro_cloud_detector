import sys
import os
import time
import json
import logging
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QGridLayout,
                             QLineEdit, QVBoxLayout, QHBoxLayout, QGroupBox, QFileDialog, QMessageBox, QCheckBox, QSizePolicy)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QDateTime
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

                            self.prediction_signal.emit(f"Prediction: {prediction['label']} (Probability: {prediction['value']:.4f})", convert_to_qt_format)
                        else:
                            self.error_signal.emit("Could not read image for display.")
                    else:
                        self.error_signal.emit("Prediction failed.")
                except Exception as e:
                    self.error_signal.emit(f"Prediction Error: {e}")
                previous_mtime = current_mtime
            time.sleep(1)


class CloudWatcherGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cloud Watcher")
        self.settings = {}
        self.load_settings()

        self.file_path = self.settings.get("file_path")
        self.model_path = self.settings.get("model_path")
        self.results_file_path = self.settings.get("results_file_path", "cloud_detection_results.txt")

        self.prediction_thread = None
        self.consecutive_cloud_count = 0
        self.consecutive_no_cloud_count = 0
        self.cloud_threshold = self.settings.get("cloud_threshold", 3)
        self.no_cloud_threshold = self.settings.get("no_cloud_threshold", 3)
        self.image_size = self.settings.get("image_size", 256)
        self.watching = False
        self.results_file = None
        self.results_file_label = None
        self.current_state = "Unsafe"

        self.initUI()

        if self.file_path:
            self.file_label.setText(os.path.basename(self.file_path))

    def initUI(self):
        main_layout = QGridLayout()

        # File and Model Selection
        file_group = QGroupBox("File and Model")
        file_layout = QGridLayout() # changed to QVBoxLayout to add the labels below the buttons
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        self.file_label = QLabel("No file selected") # add file label
        self.model_button = QPushButton("Select Model")
        self.model_button.clicked.connect(self.select_model)
        self.model_label = QLabel("No model loaded")

        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.model_button)
        file_layout.addWidget(self.model_label)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group, 0, 0)

        # Results File Path
        results_group = QGroupBox("Results File")
        results_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        self.results_button = QPushButton("Select Results File")
        self.results_button.clicked.connect(self.select_results_file)
        self.results_file_label = QLabel("No file selected")
        if self.results_file_path:
            self.results_file_label.setText(os.path.basename(self.results_file_path))
        button_layout.addWidget(self.results_button)
        results_layout.addLayout(button_layout)
        results_layout.addWidget(self.results_file_label)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group, 0, 1)

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
        main_layout.addWidget(threshold_group, 1, 0)

        # Image Size
        image_size_group = QGroupBox("Image Size")
        image_size_layout = QHBoxLayout()
        self.image_size_edit = QLineEdit(str(self.image_size))
        image_size_layout.addWidget(QLabel("Image Size:"))
        image_size_layout.addWidget(self.image_size_edit)
        image_size_group.setLayout(image_size_layout)
        main_layout.addWidget(image_size_group, 1, 1)

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
        main_layout.addWidget(state_group, 2, 0, 1, 2)

        # Start/Stop Buttons
        self.watch_button = QPushButton("Start Watching")
        self.watch_button.clicked.connect(self.toggle_watching)
        main_layout.addWidget(self.watch_button, 3, 0, 1, 2)

        # Image Display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # Make image label expandable
        main_layout.addWidget(self.image_label, 4, 0, 1, 2)

        # Prediction Text
        self.prediction_label = QLabel("No predictions yet.")
        main_layout.addWidget(self.prediction_label)

        self.setLayout(main_layout)
        if self.model_path:
            self.model_label.setText(os.path.basename(self.model_path))
        self.cloud_threshold_edit.setText(str(self.cloud_threshold))
        self.no_cloud_threshold_edit.setText(str(self.no_cloud_threshold))
        self.image_size_edit.setText(str(self.image_size))

    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File to Watch", "", "All Files (*)")
        if file_name:
            self.file_path = file_name
            self.save_settings()
            print(f"Selected file: {self.file_path}")  # Print selected file path

    def select_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "H5 Files (*.h5);;All Files (*)")
        if file_name:
            self.model_path = file_name
            self.model_label.setText(os.path.basename(self.model_path))
            self.save_settings()
            print(f"Selected model: {self.model_path}")  # Print selected model path

    def start_watching(self):
        if not self.file_path:
            QMessageBox.warning(self, "Error", "Please select a file to watch.")
            self.toggle_watching() # set the button back to start
            return
        if not self.model_path:
            QMessageBox.warning(self, "Error", "Please select a model file.")
            self.toggle_watching() # set the button back to start
            return
        try:
            self.cloud_threshold = int(self.cloud_threshold_edit.text())
            self.no_cloud_threshold = int(self.no_cloud_threshold_edit.text())
            self.image_size = int(self.image_size_edit.text())
            self.save_settings()
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid threshold or image size value.")
            self.toggle_watching() # set the button back to start
            return

        self.prediction_thread = PredictionThread(self.file_path, self.model_path, self.image_size, self)
        self.prediction_thread.prediction_signal.connect(self.update_prediction)
        self.prediction_thread.error_signal.connect(self.display_error)
        self.prediction_thread.start()

    def stop_watching(self):
        if self.prediction_thread and self.prediction_thread.isRunning():
            self.prediction_thread.stop()
            self.prediction_thread.wait()
            self.prediction_thread = None

    def toggle_watching(self):
        self.watching = not self.watching

        if self.watching:
            self.start_watching()
            self.watch_button.setText("Stop Watching")
            self.watch_button.setStyleSheet("color: red;")  # Corrected: Using style sheet
        else:
            self.stop_watching()
            self.watch_button.setText("Start Watching")
            self.watch_button.setStyleSheet("")

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

        print(prediction_text)

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

        #self.write_to_results(f"{QDateTime.currentDateTime().toString(Qt.DateFormat.ISODate)} - {prediction_text} {consecutive_text}\n")
        self.write_to_results(f"{self.current_state}\n")

    def write_to_results(self, text):
        if self.results_file_path:
            with open(self.results_file_path, "w") as outfile:
                outfile.write(f"{text}\n")

    def display_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def save_settings(self):
        self.settings["file_path"] = self.file_path
        self.settings["model_path"] = self.model_path
        self.settings["cloud_threshold"] = self.cloud_threshold
        self.settings["no_cloud_threshold"] = self.no_cloud_threshold
        self.settings["image_size"] = self.image_size
        self.settings["results_file_path"] = self.results_file_path

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