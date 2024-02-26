import cv2
import tensorflow as tf
import numpy as np
# from main import cosine_similar
import os
import cv2 as cv


def cosine_similar(embed1, embed2):
    cos = np.dot(embed2, embed1.T) / (np.linalg.norm(embed1) * np.linalg.norm(embed2))

    return cos


# path = r"C:\Users\daoda\Downloads\Gathering data\test"
# name = os.walk(path)
# image_list = []
# label_list = []
# for r, _, files in os.walk(path):
#     for f in files:
#         image_list.append(os.path.join(r, f))
#
# for i in image_list:
#     label_list.append(i.split("\\")[6])
#
# label_list = np.unique(label_list)
# print(len(label_list))
#
# model = tf.keras.models.load_model("ghostnetv1_w1.3_s1_basic_model_latest.h5", compile=True)
# dic = {}
# for label in label_list:
#     image_list = []
#     for r, _, files in os.walk(f"C:\\Users\\daoda\\Downloads\\Gathering data\\test\\{label}"):
#         for f in files:
#             image_list.append(os.path.join(r, f))
#     embedding_vector = []
#
#     for im in image_list:
#         image = cv.imread(im)
#         if image is not None:
#             if image.shape != (112, 112, 3):
#                 image = cv.resize(image, (112, 112))
#             img_ = tf.convert_to_tensor(image, dtype=tf.float32)
#             img_ = tf.expand_dims(img_, 0)
#             img_ = (img_ - 127.5) * 0.0078125
#             # print(img_)
#             result = model.predict(img_)
#             embedding_vector.append(result)
#     dic[str(label)] = embedding_vector
#
# np.savez_compressed(f"embedding_vectors/test.npz", dic)

emv = np.load("embedding_vectors/train.npz", allow_pickle=True)
train = emv[emv.files[0]].tolist()
emv = np.load("embedding_vectors/test.npz", allow_pickle=True)
test = emv[emv.files[0]].tolist()
a = 0.4
F1 = []
thresdhold = []
acc = []
while a < 1:
    TN, TP, FN, FP = 0, 0, 0, 0
    for key, value in test.items():
        for i in value:
            for k, v in train.items():
                for j in v:
                    cos = cosine_similar(i, j)
                    if cos >= 0.7:
                        if key == k:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if key == k:
                            FN += 1
                        else:
                            TN += 1

    Accuracy = (TP+TN)/(TP+TN+FN+FP)
    print(f"Accuracy: {Accuracy}")
