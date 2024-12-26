import os
import numpy as np
import warnings
import traceback
import time
import json

import sys
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QLineEdit, QVBoxLayout, QHBoxLayout, QGroupBox, QFileDialog, QMessageBox, QGridLayout, QTabWidget)
from PyQt6.QtGui import QPalette
from PyQt6.QtCore import QThread, pyqtSignal, QEventLoop

from cloud_functions import load_and_preprocess_image, load_train_data, train_model, save_model, TrainingCallback, BatchUpdateCallback

SETTINGS_FILE = "training_settings.json"  # Separate settings file for training

class TrainingThread(QThread):
    training_finished = pyqtSignal()
    update_plots = pyqtSignal(object, object, object, object, object)
    training_started = pyqtSignal()

    def __init__(self, epochs, batch_size, learning_rate, data_path, image_size, model_output):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_path = data_path
        self.image_size = image_size
        self.model_output = model_output
        self.stop_training_event = None
        self.model = None

    def stop(self):
        if self.model:
            self.model.stop_training = True
            if self.stop_training_event:
                self.stop_training_event.exit()
                self.stop_training_event = None

    def run(self):
        try:
            self.training_started.emit()

            images, labels = load_train_data(self.data_path, self.image_size)
            if not images.size:
                raise ValueError("No images loaded. Check data path.")

            self.stop_training_event = QEventLoop()

            def epoch_end_callback(epoch, logs):
                if logs:
                    train_loss = logs.get('loss')
                    val_loss = logs.get('val_loss')
                    train_accuracy = logs.get('accuracy')
                    val_accuracy = logs.get('val_accuracy')

                    self.update_plots.emit(epoch, train_loss, train_accuracy, val_loss, val_accuracy)

            #training_callback = TrainingCallback(epoch_end_callback)

            def batch_update_callback(step, logs, epoch):
                if logs:
                    train_loss = logs.get('loss')
                    train_accuracy = logs.get('accuracy')
                    val_accuracy = None
                    val_loss = None

                    if step == 'epoch_end':
                        val_accuracy = logs.get('val_accuracy')
                        val_loss = logs.get('val_loss')

                    self.update_plots.emit(
                        epoch,
                        logs.get('loss'),
                        logs.get('accuracy'),
                        logs.get('val_loss'),
                        logs.get('val_accuracy'),
                    )

            batch_callback = BatchUpdateCallback(batch_update_callback)

            # Use the updated train_model function
            self.model = train_model(
                images, labels, image_size=self.image_size, batch_size=self.batch_size, epochs=self.epochs,
                model_output=self.model_output, learning_rate=self.learning_rate, training_callback=batch_callback
            )
            print(f'Training Model finished!')

            if self.model is not None:
                epochs_range = range(len(self.model.history.epoch))
                train_loss = self.model.history.history['loss']
                val_loss = self.model.history.history['val_loss']
                train_accuracy = self.model.history.history['accuracy']
                val_accuracy = self.model.history.history['val_accuracy']
                self.update_plots.emit(list(epochs_range), train_loss, val_loss, train_accuracy, val_accuracy)
            else:
                raise ValueError("Model training failed.")

        except Exception as e:
            print(f"Error in training thread: {e}")
            #QMessageBox.critical(None, "Error", f"An error occurred during training: {e}")
            traceback.print_exc()
        finally:
            if self.stop_training_event:
                self.stop_training_event.exit()
                self.stop_training_event = None
            self.training_finished.emit()

class TrainingGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Training")
        self.training_thread = None
        self.settings = {}
        self.load_settings()
        self.initUI()

    def initUI(self):
        main_layout = QGridLayout()

        # --- Plots (Left Side - using pyqtgraph) ---
        self.train_plot = pg.PlotWidget()
        self.train_plot.addLegend()
        self.train_loss = self.train_plot.plot(pen='b', name='Train Loss')
        self.train_acc = self.train_plot.plot(pen='r', name='Train Acc')
        self.train_plot.setLabel('left', 'Loss')
        self.train_plot.setLabel('bottom', 'Epochs')
        main_layout.addWidget(self.train_plot, 0, 0, 1, 1)

        self.val_plot = pg.PlotWidget()
        self.val_plot.addLegend()
        self.val_acc = self.val_plot.plot(pen='b', name='Val Acc')
        self.val_loss = self.val_plot.plot(pen='r', name='Val Loss')
        self.val_plot.setLabel('left', 'Accuracy')
        self.val_plot.setLabel('bottom', 'Epochs')
        main_layout.addWidget(self.val_plot, 1, 0, 1, 1)

        # --- Input Fields (Right Side) ---
        input_group = QGroupBox("Training Parameters")
        input_layout = QGridLayout()

        # Data Path Selection
        self.data_path_input = QLineEdit(self.settings.get("data_path", ""))
        self.data_path_button = QPushButton("Select Data Path")
        self.data_path_button.clicked.connect(self.select_data_path)
        input_layout.addWidget(self.data_path_button, 0, 0, 1, 2)
        input_layout.addWidget(self.data_path_input, 1, 0, 1, 2)

        # Model Output Path
        self.model_output_input = QLineEdit(self.settings.get("model_output", "cloud_model"))
        self.model_output_button = QPushButton("Select Model Output Path")
        self.model_output_button.clicked.connect(self.select_model_output_path)
        input_layout.addWidget(self.model_output_button, 2, 0, 1, 2)
        input_layout.addWidget(self.model_output_input, 3, 0, 1, 2)

        # Epochs, Batch Size, Learning Rate
        self.epochs_label = QLabel("Epochs:")
        self.epochs_input = QLineEdit(str(self.settings.get("epochs", 30)))
        input_layout.addWidget(self.epochs_label, 4, 0)
        input_layout.addWidget(self.epochs_input, 4, 1)

        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_input = QLineEdit(str(self.settings.get("batch_size", 32)))
        input_layout.addWidget(self.batch_size_label, 5, 0)
        input_layout.addWidget(self.batch_size_input, 5, 1)

        self.learning_rate_label = QLabel("Learning Rate:")
        self.learning_rate_input = QLineEdit(str(self.settings.get("learning_rate", 0.001)))
        input_layout.addWidget(self.learning_rate_label, 6, 0)
        input_layout.addWidget(self.learning_rate_input, 6, 1)

        self.size_label = QLabel("Image Size:")
        self.size_input = QLineEdit(str(self.settings.get("image_size", 256)))
        input_layout.addWidget(self.size_label, 7, 0)
        input_layout.addWidget(self.size_input, 7, 1)

        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        input_layout.addWidget(self.train_button, 8, 0, 1, 2)

        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group, 0, 1, 2, 1)

        self.setLayout(main_layout)

        # Set style sheet for consistent text field appearance
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, palette.color(QPalette.ColorRole.Window))
        self.setPalette(palette)

    def select_data_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory", "")
        if directory:
            self.data_path_input.setText(directory)
            self.settings["data_path"] = directory
            self.save_settings()

    def select_model_output_path(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Select Model Output File", "", "Keras Files (*.keras);;All Files (*)")
        if filename:
            self.model_output_input.setText(filename)
            self.settings["model_output"] = filename
            self.save_settings()

    def start_training(self):
        try:
            epochs = int(self.epochs_input.text())
            batch_size = int(self.batch_size_input.text())
            learning_rate = float(self.learning_rate_input.text())
            image_size = int(self.size_input.text())
            data_path = self.data_path_input.text()
            model_output = self.model_output_input.text()

            self.settings["epochs"] = epochs
            self.settings["batch_size"] = batch_size
            self.settings["learning_rate"] = learning_rate
            self.settings["image_size"] = image_size
            self.save_settings()

        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid input values.")
            return

        if not os.path.isdir(data_path):
            QMessageBox.warning(self, "Error", "Invalid data path. Please select a directory.")
            return

        self.train_data = {'epochs': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        self.train_button.setEnabled(False)
        self.training_thread = TrainingThread(epochs, batch_size, learning_rate, data_path, image_size, model_output)
        self.training_thread.training_finished.connect(self.training_done)
        self.training_thread.update_plots.connect(self.update_plots)
        self.training_thread.training_started.connect(self.training_started)
        self.training_thread.start()

    def training_started(self):
        self.train_button.setText("Training...")
        self.train_button.setStyleSheet("color: orange;")

    def training_done(self):
        self.train_button.setEnabled(True)
        self.train_button.setText("Start Training")
        self.train_button.setStyleSheet("")
        self.training_thread = None

    def update_plots(self, epochs, train_loss, train_acc, val_loss=None, val_acc=None):
        self.train_data['train_loss'].append(train_loss)
        self.train_data['train_acc'].append(train_acc)
        self.train_data['epochs'].append(epochs)
        self.train_loss.setData(self.train_data['epochs'], self.train_data['train_loss'])
        self.train_acc.setData(self.train_data['epochs'], self.train_data['train_acc'])
        if val_loss is not None and val_acc is not None:
            self.train_data['val_loss'].append(val_loss)
            self.train_data['val_acc'].append(val_acc)
            x = list(range(1, len(self.train_data['val_acc']) + 1, 1))
            self.val_acc.setData(x, self.train_data['val_acc'])
            self.val_loss.setData(x, self.train_data['val_loss'])

    def save_settings(self):
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(self.settings, f, indent=4) # Indent for readability
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_settings(self):
        try:
            with open(SETTINGS_FILE, "r") as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            pass # Use default settings if file doesn't exist
        except json.JSONDecodeError:
            print("Error decoding settings file. Using default settings.")
        except Exception as e:
            print(f"Error loading settings: {e}")


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cloud Detector and Trainer")
        self.settings = {}
        self.load_settings()

        self.initUI()

    def initUI(self):
        self.tabs = QTabWidget()
        self.training_gui = TrainingGUI()
        self.tabs.addTab(self.training_gui, "Model Trainer")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def save_settings(self):
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
            pass
        except json.JSONDecodeError:
            print("Error decoding settings file. Using default settings.")
        except Exception as e:
            print(f"Error loading settings: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_widget = MainWidget()
    main_widget.show()
    sys.exit(app.exec())