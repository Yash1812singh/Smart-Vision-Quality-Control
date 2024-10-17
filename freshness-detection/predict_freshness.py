import cv2
import tensorflow as tf
import numpy as np

# Load Model
model = tf.keras.models.load_model('freshness_model.h5')

def predict_freshness(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    return "Fresh" if prediction > 0.5 else "Spoiled"

# Example usage
print(predict_freshness('dataset/fresh/apple.jpg'))
