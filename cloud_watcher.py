import time
import os
import argparse
from cloud_functions import predict_image, load_model
import logging
import sys
import warnings


def watch_file(file_path, model_path, output_file, image_size=256):
    """Watches a specific file for changes and runs prediction."""
    model = load_model(model_path)
    if model is None:
        logging.error(f"Failed to load model from {model_path}")
        return

    previous_mtime = None

    while True:
        try:
            if not os.path.exists(file_path):
                logging.info(f"File not found: {file_path}. Waiting...")
                previous_mtime = None
                time.sleep(5)
                continue

            try:
                current_mtime = os.path.getmtime(file_path)
            except OSError as e:
                logging.error(f"OS Error accessing file: {e}")
                time.sleep(5)
                continue

            if current_mtime != previous_mtime:
                logging.info(f"File changed: {file_path}. Running prediction...")
                prediction = predict_image(model, file_path, image_size)
                if prediction is not None:
                    with open(output_file, "w") as outfile:
                        outfile.write(f"Cloud probability: {prediction['value']:.4f}\n")
                        outfile.write(f"Prediction: {prediction['label']}\n")
                    logging.info(f"Prediction: {prediction['label']} (Probability: {prediction['value']:.4f})")
                    logging.info(f"Prediction saved to {output_file}")
                else:
                    logging.error("Prediction failed.")
                previous_mtime = current_mtime

            time.sleep(1)  # Check every second
        except KeyboardInterrupt:
            print("Watcher stopped.")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a file for changes and run prediction.")
    parser.add_argument("--file", type=str, required=True, help="Path to the file to watch.")
    parser.add_argument("--model", type=str, default="cloud_detection_model.keras", help="Path to the trained model (.keras file).")
    parser.add_argument("--output", type=str, default="prediction_result.txt", help="Path to save the prediction result.")
    parser.add_argument("--image_size", type=int, default=256, help="Image size (height and width).")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(filename='watcher.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    watch_file(args.file, args.model, args.output, args.image_size)