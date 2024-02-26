import time

import cv2 as cv
import tensorflow as tf
from nms_face import detect_face


def show_results(img, xyxy, conf, draw=True, name=""):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()
    if draw:
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv.LINE_AA)

        # show 5 circle
        # clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        #
        # for i in range(5):
        #     point_x = int(landmarks[2 * i])
        #     point_y = int(landmarks[2 * i + 1])
        #     cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)
        tf = max(tl - 1, 1)  # font thickness
        label = str(conf)[:5]
        label = label + f" {name}"
        cv.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv.LINE_AA)
    return img


model = tf.lite.Interpreter(model_path="converted_model.tflite")
model.allocate_tensors()
# Get input and output tensors
input_details = model.get_input_details()
output_details = model.get_output_details()

video = cv.VideoCapture(0)
i = 0
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# name = ["Unknown" for i in range(10)]
while True:
    net, frame = video.read()
    img, face, xyxys, confidence = detect_face(frame, model)

    if len(face) != 0:
        for f in range(len(face)):
            img = show_results(img=img, xyxy=xyxys[f], conf=confidence[f])

    i += 1
    cv.imshow('result', img)
    k = cv.waitKey(1)
    if 0xFF == ord('q'): break

video.release()
cv.destroyAllWindows()
