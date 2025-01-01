import time
import datetime
import os
import argparse
from cloud_functions import predict_image, load_model
import logging
import sys
import warnings
import requests
from PIL import Image
import io
import numpy as np


def download_image_from_url(url, timeout=5):
    """Downloads an image from a URL."""
    try:
        response = requests.get(url, timeout=timeout, stream=True) # added stream=True
        response.raise_for_status()  # Raise exception for non-200 status codes
        return Image.open(io.BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error opening image: {e}")
        return None

def watch_webpage(image_url, model_path, output_file, image_size=256, check_interval=5):
    """Watches a webpage for image changes (simulated by checking at intervals)."""
    model = load_model(model_path)
    if model is None:
        logging.error(f"Failed to load model from {model_path}")
        return

    previous_image_hash = None

    while True:
        try:
            image = download_image_from_url(image_url)
            if image is None:
                time.sleep(check_interval)
                continue
            
            current_image_hash = hash(image.tobytes())

            if current_image_hash != previous_image_hash:
                logging.info(f"Image on webpage might have changed. Running prediction...")
                prediction = predict_image(model, np.asarray(image), image_size)
                if prediction is not None:
                    timestamp = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S ")
                    with open(output_file, "w") as outfile:
                        outfile.write(f"{timestamp} Prediction: {prediction['label']}\n")
                    logging.info(f"{timestamp} Prediction: {prediction['label']} (Value: {prediction['value']:.4f})")
                    if args.save_folder is not None:
                        save_image(image, args.save_folder, prediction['label'])
                else:
                    logging.error("Prediction failed.")
                previous_image_hash = current_image_hash

            time.sleep(check_interval)

        except KeyboardInterrupt:
            print("Watcher stopped.")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            time.sleep(check_interval)

def save_image(image, save_folder, label, filename_prefix="downloaded_image"):
    """Saves a PIL Image to a subfolder based on the label."""
    try:
        label_folder = os.path.join(save_folder, label)
        os.makedirs(label_folder, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}_image.jpg"
        filepath = os.path.join(label_folder, filename)
        image.save(filepath)
        logging.info(f"Image saved to: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Error saving image: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a webpage for image changes and run prediction.")
    parser.add_argument("--url", type=str, required=True, help="URL of the webpage containing the image.")
    parser.add_argument("--model", type=str, default="cloud_detection_model.keras", help="Path to the trained model (.keras file).")
    parser.add_argument("--output", type=str, default="prediction_result.txt", help="Path to save the prediction result.")
    parser.add_argument("--image_size", type=int, default=256, help="Image size (height and width).")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds.")
    parser.add_argument("--save_folder", type=str, default=None, help="Folder to save downloaded images.")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(filename='watcher.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    watch_webpage(args.url, args.model, args.output, args.image_size, args.interval)