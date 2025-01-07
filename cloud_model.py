import tensorflow as tf

def create_cloud_model(input_shape, large_model=False):
    """Creates a convolutional neural network model for cloud detection.

    Args:
        input_shape: Tuple representing the input image shape (height, width, channels).

    Returns:
        A tf.keras.Model instance.
    """
    if not large_model:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (6, 6), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((4, 4)),
            tf.keras.layers.Conv2D(64, (6, 6), activation='relu'),
            tf.keras.layers.BatchNormalization(), # Batch Normalization
            tf.keras.layers.MaxPooling2D((4, 4)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),  # Dropout for regularization
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
        ])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (6, 6), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((4, 4)),
            tf.keras.layers.Conv2D(64, (6, 6), activation='relu'),
            tf.keras.layers.BatchNormalization(), # Batch Normalization
            tf.keras.layers.MaxPooling2D((4, 4)),
            tf.keras.layers.Conv2D(128, (6, 6), activation='relu'),
            tf.keras.layers.MaxPooling2D((4, 4)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),  # Dropout for regularization
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
        ])

    return model

if __name__ == '__main__':
    # Example usage:
    model = create_cloud_model((256, 256, 3))  # Example input shape
    model.summary()