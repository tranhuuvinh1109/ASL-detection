import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

final = ""
img_width, img_height = 128, 128
batch_size = 32
path_model = "model/vggmodel_2912.h5"
model = keras.models.load_model(path_model)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

counter = 0

labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


def DetectionObject(img):
    img_array = img_to_array(img)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)

    predicted_class_index = np.argmax(predictions[0])
    predicted_class = labels[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index] * 100

    # return {"label": predicted_class, "confidence": float(confidence_score)}
    result_string = f"{predicted_class} - {confidence_score:.2f}%"
    return result_string


while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture frame")
        break

    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        result = ""
        if w > 0 and h > 0:
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

            if not imgCrop.size == 0:

                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    widthCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (widthCal, imgSize))
                    imgResizeShape = imgResize.shape
                    widthGap = math.ceil((imgSize - widthCal) / 2)
                    imgWhite[:, widthGap : widthCal + widthGap] = imgResize

                    imgWhite = cv2.resize(imgWhite, (128, 128))
                    imgWhiteResized = cv2.resize(imgWhite, (128, 128))
                    result = DetectionObject(imgWhiteResized)

                else:
                    k = imgSize / w
                    heightCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, heightCal))
                    imgResizeShape = imgResize.shape
                    heightGap = math.ceil((imgSize - heightCal) / 2)
                    imgWhite[heightGap : heightCal + heightGap, :] = imgResize
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

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

        else:
            print("Invalid bounding box size")

    key = cv2.waitKey(1)

    if key == ord("s"):
        final += result.split(' - ')[0]

    cv2.putText(
            imgOutput,
            final,
            (200, 100),
            cv2.FONT_HERSHEY_COMPLEX,
            1.7,
            (255, 255, 255),
            2,
        )   
    cv2.imshow("Image", imgOutput)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()