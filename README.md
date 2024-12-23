# Cloud Detection Project

This project uses a Convolutional Neural Network (CNN) to detect the presence of clouds in images. It provides scripts for training the model, making predictions on single images, and monitoring a directory for new images.

## Project Structure

cloud-detection/
├── cloud_model.py      # Defines the CNN model architecture.
├── cloud_functions.py  # Contains core functions for data handling, model training, and prediction.
├── cloud_detector.py   # Main script for training and single image prediction.
└── cloud_watcher.py    # Script for monitoring a directory and making predictions.
└── requirements.txt    # Lists project dependencies.

## Requirements

*   Python 3.7+
*   TensorFlow 2.x
*   scikit-learn
*   astropy
*   NumPy <2.0

Install dependencies:
```bash
pip install -r requirements.txt
```
Usage
Training the Model

```bash
python cloud_detector.py --train --data_dir path/to/data --image_size 256 --batch_size 32 --epochs 30 --model_output cloud_model
```
--data_dir: Path to training data (organized in clouds and no-clouds subdirectories).
--image_size: Image resize dimension (e.g., 256x256).
--batch_size: Training batch size.
--epochs: Number of training epochs.
--model_output: Output path for the trained model (without extension).

Predicting on a Single Image
```bash
python cloud_detector.py --model path/to/cloud_model.keras --image path/to/image.jpg --image_size 256
```
--model: Path to the trained model (.h5 file).
--image: Path to the image for prediction.
--image_size: Image size used during training.

Monitoring a Directory

```bash
python cloud_watcher.py --file path/to/watch_file --model path/to/cloud_model.keras --output predictions.txt --image_size 256
```
--file: File to watch for changes (touch this file to trigger prediction).
--model: Path to the trained model.
--output: Output file for predictions.
--image_size: Image size used during training.

Data Format
Training data should be in a directory with two subdirectories:

clouds: Contains images with clouds.
no-clouds: Contains images without clouds.
Supported image formats: .png, .jpg, .jpeg, .fits.

FITS Image Handling
The code includes normalization for FITS images based on their data type.

Logging
cloud_watcher.py logs events to watcher.log.

Model Architecture
The CNN architecture (defined in cloud_model.py) is a simple convolutional stack followed by dense layers.

Contributing
Contributions are welcome.

License
MIT License
