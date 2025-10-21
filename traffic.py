import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    
    # Debug: Check if data_dir exists and what's in it
    if not os.path.exists(data_dir):
        return images, labels
    
    # Try different approaches to find category directories
    # First, try the standard approach (0, 1, 2, ...)
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        
        if os.path.exists(category_dir):
            try:
                for filename in os.listdir(category_dir):
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        continue
                        
                    image_path = os.path.join(category_dir, filename)
                    
                    try:
                        image = cv2.imread(image_path)
                        if image is None:
                            continue
                        
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                        
                        images.append(image)
                        labels.append(category)
                        
                    except Exception:
                        continue
            except Exception:
                continue
    
    # If no images found with standard approach, try alternative approaches
    if len(images) == 0:
        # Try looking for any subdirectories that might contain images
        try:
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    # Try to parse category number from directory name
                    try:
                        category = int(item)
                        if 0 <= category < NUM_CATEGORIES:
                            for filename in os.listdir(item_path):
                                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                                    continue
                                    
                                image_path = os.path.join(item_path, filename)
                                
                                try:
                                    image = cv2.imread(image_path)
                                    if image is None:
                                        continue
                                    
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                                    
                                    images.append(image)
                                    labels.append(category)
                                    
                                except Exception:
                                    continue
                    except ValueError:
                        # Directory name is not a number, skip
                        continue
        except Exception:
            pass
    
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([
        # First convolutional layer with 32 filters
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        
        # First max pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional layer with 64 filters
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        
        # Second max pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Third convolutional layer with 128 filters
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        
        # Third max pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten units
        tf.keras.layers.Flatten(),
        
        # Add a dropout layer to prevent overfitting
        tf.keras.layers.Dropout(0.5),
        
        # Add a hidden layer with dropout
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        
        # Add another hidden layer
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        
        # Add an output layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    # Train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


if __name__ == "__main__":
    main()
