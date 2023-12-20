import cv2
import tkinter as tk
from PIL import Image, ImageTk
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array

final = ""
img_width, img_height = 128, 128
batch_size = 32
path_model = "model/vggmodel_2912.h5"
model = keras.models.load_model(path_model)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
imgWhiteResized = None
imgCrop = None

labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

def button_click():
    global final, imgWhiteResized
    print("Button Click")
    if imgWhiteResized is not None:
        result = DetectionObject(imgWhiteResized)
        final += result.split(' - ')[0]
        print(final)
        update_result_label()

def clear_result():
    global final
    final = ''
    update_result_label()

def DetectionObject(img):
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions[0])
    predicted_class = labels[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index] * 100

    result_string = f"{predicted_class} - {confidence_score:.2f}%"
    return result_string

def show_frame(frame):
    global final, imgWhiteResized, imgCrop
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    label.img = img_tk
    label.config(image=img_tk)

    img_output_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    label_output.img = img_output_tk
    label_output.config(image=img_output_tk)

    if imgWhiteResized is not None:
        img_white_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(imgWhiteResized, cv2.COLOR_BGR2RGB)))
        label_white.img = img_white_tk
        label_white.config(image=img_white_tk)

    if imgCrop is not None:
        img_crop_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)))
        label_crop.img = img_crop_tk
        label_crop.config(image=img_crop_tk)

    cv2.putText(
        frame,
        final,
        (200, 100),
        cv2.FONT_HERSHEY_COMPLEX,
        1.7,
        (255, 255, 255),
        2,
    )

    label.after(10, update_frame)

def update_frame():
    global imgWhiteResized, imgCrop
    _, frame = cap.read()
    imgOutput = frame.copy()

    hands, _ = detector.findHands(frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        result = ""

        if w > 0 and h > 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = frame[y - offset: y + h + offset, x - offset: x + w + offset]

            if not imgCrop.size == 0:
                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    widthCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (widthCal, imgSize))
                    imgResizeShape = imgResize.shape
                    widthGap = math.ceil((imgSize - widthCal) / 2)
                    imgWhite[:, widthGap: widthCal + widthGap] = imgResize

                    imgWhite = cv2.resize(imgWhite, (128, 128))
                    imgWhiteResized = cv2.resize(imgWhite, (128, 128))
                    result = DetectionObject(imgWhiteResized)

                else:
                    k = imgSize / w
                    heightCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, heightCal))
                    imgResizeShape = imgResize.shape
                    heightGap = math.ceil((imgSize - heightCal) / 2)
                    imgWhite[heightGap: heightCal + heightGap, :] = imgResize
                    imgWhite = cv2.resize(imgWhite, (128, 128))
                    imgWhiteResized = cv2.resize(imgWhite, (128, 128))
                    result = DetectionObject(imgWhiteResized)

                cv2.rectangle(
                    imgOutput,
                    (x - offset, y - offset - 50),
                    (x - offset + 80, y - offset - 50 + 50),
                    (255, 0, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    imgOutput,
                    result,
                    (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1.7,
                    (255, 255, 255),
                    2,
                )
                cv2.rectangle(
                    imgOutput,
                    (x - offset, y - offset),
                    (x + w + offset, y + h + offset),
                    (255, 0, 255),
                    4,
                )

        else:
            print("Invalid bounding box size")

    show_frame(imgOutput)

def update_result_label():
    result_label.config(text=final)

root = tk.Tk()
root.title("Hand Gesture Recognition")

label = tk.Label(root)
label.pack(padx=10, pady=10)

label_output = tk.Label(root)

label_crop = tk.Label(root)
label_crop.pack(side=tk.LEFT, padx=10, pady=10)

label_white = tk.Label(root)
label_white.pack(side=tk.LEFT, padx=10, pady=10)

result_label = tk.Label(root, text=final, 
font=("Helvetica", 16))
result_label.pack(side=tk.BOTTOM, pady=10)

button = tk.Button(root, text="Click me!", command=button_click)
button.pack(side=tk.RIGHT, padx=20, pady=20)

clear_button = tk.Button(root, text="Clear", command=clear_result)
clear_button.pack(side=tk.RIGHT, padx=20, pady=20)

update_frame()

root.mainloop()

cv2.destroyAllWindows()
