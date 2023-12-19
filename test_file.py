import pandas as pd
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

path_model = "model/vggmodel_2912.h5"
model = keras.models.load_model(path_model)
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]

def DetectionObject(img_array):
    img_array = img_array / 255.0  # Normalize the image
    img_array = tf.image.resize(img_array, (128, 128))  # Resize the image

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = labels[predicted_class_index]

    confidence_score = predictions[0][predicted_class_index] * 100

    return {'label': predicted_class, 'confidence': float(confidence_score)}

csv_file_path = 'D:/HocMay/drive/test.csv'
image_folder_path = 'D:/HocMay/drive/test'

df = pd.read_csv(csv_file_path)

for index, row in df.iterrows():
    filename = row['file']
    image_path = os.path.join(image_folder_path, filename)

    # Load and preprocess the image
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)

    prediction_result = DetectionObject(img_array)

    df.at[index, 'predict'] = prediction_result['label']
    df.at[index, 'confidence'] = prediction_result['confidence']
    print(filename, prediction_result['label'], prediction_result['confidence'])

df.to_csv(csv_file_path, index=False)

print(f'Đã cập nhật file CSV tại: {csv_file_path}')
