import argparse
import sys
from cloud_functions import load_data, train_model, predict_image, load_model, save_model, print_model_info

def main():
    if args.train:
        train(args.data_dir, args.image_size, args.batch_size, args.epochs, args.model_output)
    elif args.image and args.model:
        predict(args.model, args.image, args.image_size)
    else:
        print("Invalid arguments. Please specify either --train or provide both --model and --image arguments.")

def train(data_dir, image_size, batch_size, epochs, model_output):
    images, labels = load_data(data_dir, image_size)
    if images.size == 0:
        print("No images found. Check data directory and file extensions.")
        return
    model = train_model(images, labels, image_size, batch_size, epochs, model_output)
    if model is not None:
        print(f"Model saved to {model_output}.h5")
    else:
        print("Model training failed. Check for errors during training.")

def predict(model_path, image_path, image_size):
    model = load_model(model_path)
    print_model_info(model, [image_size, image_size])
    if model is None:
        print(f"Failed to load model from {model_path}")
        return
    prediction = predict_image(model, image_path, image_size)
    if prediction is not None:
        prediction_label = "Clouds" if prediction > 0.5 else "No Clouds"
        print(f"Cloud probability: {prediction:.4f}")
        print(f"Prediction: {prediction_label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict cloud presence in images.")
    parser.add_argument("--data_dir", default=".", type=str, help="Path to the data directory (for training).")
    parser.add_argument("--model", default="cloud_detection_model.keras", type=str, help="Path to the trained model (.keras file).")
    parser.add_argument("--image", type=str, help="Path to the image to predict.")
    parser.add_argument("--image_size", type=int, default=256, help="Image size (height and width).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--train", action="store_true", help="Enable training mode.")
    parser.add_argument("--model_output", type=str, default="cloud_detection_model", help="Path to save the trained model (without extension).")
    args = parser.parse_args()
    main()