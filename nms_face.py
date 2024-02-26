import os

import numpy as np
import time
import tensorflow as tf
import cv2

conf_thres = 0.25
iou_thres = 0.45
labels = ()
classes = None
agnostic = False


def show_results(img, xyxy, conf, draw=True):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()
    if draw:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

        # show 5 circle
        # clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        #
        # for i in range(5):
        #     point_x = int(landmarks[2 * i])
        #     point_y = int(landmarks[2 * i + 1])
        #     cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)
        tf = max(tl - 1, 1)  # font thickness
        label = str(conf)[:5]
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = tf.clip_by_value(
        tf.math.reduce_min(box1[:, None, 2:], box2[:, 2:]) - tf.math.reduce_max(box1[:, None, :2], box2[:, :2]), 0,
        float('inf')).numpy().prod(axis=2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = tf.identity(x) if isinstance(x, tf.Tensor) else np.copy(x)
    y_ = tf.Variable(y)
    y_[:, 0].assign(x[:, 0] - x[:, 2] / 2)  # top left x
    y_[:, 1].assign(x[:, 1] - x[:, 3] / 2)  # top left y
    y_[:, 2].assign(x[:, 0] + x[:, 2] / 2)  # bottom right x
    y_[:, 3].assign(x[:, 1] + x[:, 3] / 2)  # bottom right y
    return y_


def yolo_nms(inputs, max_output_size=15, iou_threshold=0.35, score_threshold=0.25):
    inputs = inputs[0][inputs[0, :, -1] > score_threshold]
    xy_center, wh, ppt, cct = inputs[:, :2], inputs[:, 2:4], inputs[:, 4:14], inputs[:, 14]
    xy_start = xy_center - wh / 2
    xy_end = xy_start + wh
    bbt = tf.concat([xy_start, xy_end], axis=-1)
    # print(bbt.shape, cct.shape)
    rr = tf.image.non_max_suppression(bbt, cct, max_output_size=max_output_size, iou_threshold=iou_threshold,
                                      score_threshold=0.0)
    bbs, pps, ccs = tf.gather(bbt, rr, axis=0), tf.gather(ppt, rr, axis=0), tf.gather(cct, rr, axis=0)
    pps = tf.reshape(pps, [-1, 5, 2])
    return bbs.numpy(), pps.numpy(), ccs.numpy()


def save_img(img, xyxy):
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    del_x = x2 - x1
    del_y = y2 - y1
    save_im = img[y1:y2, x1:x2, :]
    # print(save_im.shape)
    if del_y > del_x:
        save_im = cv2.copyMakeBorder(save_im, top=0, bottom=0, left=int((del_y - del_x) / 2),
                                     right=int((del_y - del_x) / 2), borderType=0)
        save_im[:, 0:int((del_y - del_x) / 2), :] += 128
        save_im[:, int((del_y + del_x)/2):del_y, :] += 128
    else:
        save_im = cv2.copyMakeBorder(save_im, top=int((del_x - del_y) / 2), bottom=int((del_x - del_y) / 2), left=0,
                                     right=0, borderType=0)
        save_im[ 0:int((del_x - del_y) / 2), :, :] += 128
        save_im[int((del_y + del_x)/2):del_x, :, :] += 128
    return save_im


def detect_face(frame, model, not_cut_image=True, is_rec=True, label=None):
    if is_rec:
        img = cv2.copyMakeBorder(frame, 80, 80, 0, 0, 0)
    else:
        img = frame
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    img_ = np.transpose(img, [2, 0, 1])
    img_ = tf.convert_to_tensor(img_, dtype=tf.float32)
    img_ = tf.expand_dims(img_, 0)
    img_ /= 255.0
    model.set_tensor(input_details[0]['index'], img_)
    model._interpreter.Invoke()
    prediction = model.get_tensor(output_details[0]['index'])

    nc = prediction.shape[2] - 15
    xc = prediction[..., 4] > 0.25
    prediction = tf.convert_to_tensor(prediction, dtype=tf.float32)

    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [tf.zeros((0, 16), dtype=tf.float32)] * prediction.shape[0]
    # print(prediction.shape)
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = tf.boolean_mask(x, xc[xi])  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = tf.zeros((len(l), nc + 15), dtype=tf.float32)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v = tf.tensor_scatter_nd_update(v, tf.stack([tf.range(len(l)), l[:, 0]], axis=1) + 15, 1.0)  # cls
            x = tf.concat((x, v), 0)

        # If none remain process next image
        if x.shape[0] == 0:
            continue

        # Compute conf
        # print(x.shape)
        x_ = tf.Variable(x)
        x_[:, 15:].assign(x[:, 15:] * tf.expand_dims(x[:, 4], axis=1))  # conf = obj_conf * cls_conf
        # print(x_)
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x_[:, :4])
        # Detections matrix nx6 (xyxy, conf, landmarks, cls)
        if multi_label:
            i, j = tf.where(x_[:, 15:] > conf_thres)
            x = tf.concat(
                (
                    box[i], tf.expand_dims(x_[i, j + 15], axis=1), x_[i, 5:15],
                    tf.expand_dims(tf.cast(j, tf.float32), axis=1)),
                axis=1)
        else:  # best class only
            conf = tf.math.reduce_max(x_[:, 15:], axis=1, keepdims=True)
            j = tf.zeros(conf.shape)
            x = tf.boolean_mask(tf.concat((box, conf, x_[:, 5:15], tf.cast(j, tf.float32)), axis=1),
                                tf.reshape(conf, -1) > conf_thres)

        # Filter by class
        if classes is not None:
            x = tf.boolean_mask(x, tf.reduce_any(tf.equal(x[:, 5:6], tf.constant(classes, dtype=tf.float32)), axis=1))

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if n == 0:
            continue

        # Batched NMS
        c = x[:, 15:16] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # print(scores)
        i = tf.image.non_max_suppression(boxes, scores, max_output_size=5, iou_threshold=iou_thres)  # NMS

        # if i.shape[0] > max_det:  # limit detections
        #    i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = tf.linalg.matmul(weights, x[:, :4]) / tf.expand_dims(weights.sum(1), axis=1)  # merged boxes
            if redundant:
                i = tf.boolean_mask(i, tf.reduce_sum(tf.cast(iou, tf.int32), axis=1) > 1)  # require redundancy

        output[xi] = tf.gather(x, i)
        if (time.time() - t) > time_limit:
            break  # time limit exceeded
    # output = yolo_nms(prediction)
    # print(output)
    save_face = []
    xyxys = []
    confident = []
    for i, det in enumerate(output):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_coords(img_.shape[2:], det[:, :4], img_.shape).round()

            # Print results
            unique, _ = tf.unique(det[:, -1])
            for c in unique:
                n = tf.math.count_nonzero(det[:, -1] == c)  # detections per class

            # det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()

            for j in range(det.shape[0]):
                xyxy = tf.reshape(det[j, :4], -1).numpy().tolist()
                xyxys.append(xyxy)
                conf = det[j, 4].cpu().numpy()
                confident.append(conf)
                # landmarks = det[j, 5:15].view(-1).tolist()
                # class_num = det[j, 15].cpu().numpy()

                img = show_results(img, xyxy, conf, draw=not_cut_image)

                if conf >= 0.85 and not_cut_image == False:
                    save_im = save_img(img, xyxy)
                    save_im = cv2.resize(save_im, (112, 112))
                    if label != None:
                        os.makedirs(f"save_image/{label}", exist_ok=True)
                        cv2.imwrite(f"save_image/{label}/12012024+{round(time.time(), 4)}.jpg", save_im)
                    print(f"saved_image {save_im.shape}")

                if conf >= 0.80 and not_cut_image == True:

                    save_im = save_img(img, xyxy)
                    save_im = cv2.resize(save_im, (112, 112))
                    save_face.append(save_im)
                # multi_face.append(img)
                # multi_conf.append(conf)
    return img, save_face, xyxys, confident


if __name__ == "__main__":
    model = tf.lite.Interpreter(model_path="converted_model.tflite")
    model.allocate_tensors()
    # Get input and output tensors
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    video = cv2.VideoCapture(0)

    while True:
        net, frame = video.read()
        # print(img_.shape)
        # print(out)
        t = time.time()
        img, face = detect_face(frame, model)
        if len(face) != 0:
            cv2.imshow('result', face[0])
        k = cv2.waitKey(1)
        if 0xFF == ord('q'): break
        # print((time.time() - t) * 1000)
