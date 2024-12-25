import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from cloud_model import create_cloud_model
from sklearn.utils import class_weight
from astropy.io import fits
import warnings
import traceback
import cv2
import time

# Suppress TensorFlow and Python warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

def load_and_preprocess_image(image_path, image_size, is_training=False):
    """Loads and preprocesses a single image."""
    target_size = (image_size, image_size)
    try:
        if image_path.lower().endswith('.fits'):
            with fits.open(image_path) as hdul:
                img_data = hdul[0].data
                # FITS data normalization
                if img_data.dtype == np.uint16:
                    img_data = img_data.astype(np.float32) / np.iinfo(np.uint16).max
                elif img_data.dtype == np.int16:
                    img_data = img_data.astype(np.float32) / np.iinfo(np.int16).max
                elif img_data.dtype == np.uint8:
                    img_data = img_data.astype(np.float32) / np.iinfo(np.uint8).max
                elif img_data.dtype == np.int8:
                    img_data = img_data.astype(np.float32) / np.iinfo(np.int8).max
                elif img_data.dtype == np.float64:
                    img_data = img_data.astype(np.float32)
                elif img_data.dtype != np.float32:
                    img_data = img_data.astype(np.float32)

                if len(img_data.shape) == 2:
                    img_data = np.stack([img_data, img_data, img_data], axis=-1)
                else:
                    raise ValueError(f"unsupported shape {img_data.shape} for file {image_path}")
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            img_data = cv2.resize(img, target_size)
            if len(img_data.shape) == 2:
                img_data = np.stack([img_data, img_data, img_data], axis=-1)

        img_data = img_data / 255.0

        return img_data # Removed the extra expand_dims here

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        traceback.print_exc()
        return None

def load_data(data_dir, image_size):
    images = []
    labels = []
    for label, folder in enumerate(['no-clouds', 'clouds']):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.fits')):
                image_path = os.path.join(folder_path, filename)
                img_data = load_and_preprocess_image(image_path, image_size) # is_training is False by default
                if img_data is not None:
                    images.append(img_data)
                    labels.append(label)
    return np.array(images), np.array(labels)

def print_model_info(model, input_shape):
    """Prints model summary and input shape information."""
    print("Model Input Shape:", input_shape)
    model.summary()

def train_model(images, labels, image_size, batch_size, epochs, model_output):
    try:
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))
        input_shape = images[0].shape
        model = create_cloud_model(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print_model_info(model, input_shape)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_output + ".keras",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        X_train = np.stack(X_train, axis=0) # Added stacking here
        X_val = np.stack(X_val, axis=0) # Added stacking here

        model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint_callback, early_stopping],
            class_weight=class_weights_dict
        )
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        return None

def save_model(model, save_path):
    try:
        model.save(save_path)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        traceback.print_exc()
        return False

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None

def predict_image(model, image_path, image_size):
    result = None
    try:
        # Start measuring inference time
        start_time = time.time()

        img_array = load_and_preprocess_image(image_path, image_size)
        if img_array is None:
            return result
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)

        # End measuring inference time
        inference_time = time.time() - start_time

        if isinstance(prediction, np.ndarray):
            if prediction.ndim > 0:
                prediction_value = prediction.item() if prediction.size == 1 else prediction[0, 0]
            else:
                prediction_value = prediction
        elif isinstance(prediction, (int, float, np.int64, np.float64)):
            prediction_value = prediction
        else:
            raise TypeError(f"Unexpected prediction type: {type(prediction)}")
            return result

        result = {
            'value': prediction_value,
            'label': "Clouds" if prediction > 0.5 else "No Clouds",
            'inference_time': inference_time,
        }
        return result

    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return result

def print_prediction(prediction):
    if not isinstance(prediction, dict):
        raise TypeError(f"Expectect dict. Unexpected prediction: {type(prediction)}")
        return
    try:
        print(f"Prediction:     {prediction['label']}")
        print(f"Value:          {prediction['value']:1.3f}")
        print(f"Inference Time: {prediction['inference_time']:2.3f}s")
    except Exception as e:
        traceback.print_exc()