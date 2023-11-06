# https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1

import tensorflow as tf
import cv2 as cv
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

model = tf.saved_model.load('./faster_rcnn_openimages_v4_inception_resnet_v2_1/')


# print(list(model.signatures.keys()))

# detect the cat in each frame using the pretrained model
def frcnn_detect(input_frame):
    # convert frame to right format
    frame_rgb = cv.cvtColor(input_frame, cv.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.astype(np.float32) / 255.0
    # frame_int = tf.cast(frame, dtype=tf.uint8)
    tensor_frame = tf.convert_to_tensor([frame_rgb])

    # use model
    detect = model.signatures['default'](tensor_frame)

    boxes = detect['detection_boxes'].numpy()
    scores = detect['detection_scores'].numpy()
    classes = detect['detection_class_labels'].numpy()

    # filter out bad boxes and return a list of the indices for the good ones
    good_scores = scores > threshold

    # replace all info and return
    boxes, scores, classes = boxes[good_scores], scores[good_scores], classes[good_scores]

    return boxes, scores, classes


# draws correct box around cat
def draw_box(input_frame, tracks, scores):
    if not scores:
        return input_frame

    if len(tracks) > 0:
        # get the most confident box
        highest_conf_index = np.argmax(scores)
        track = tracks[highest_conf_index]
        left, top, right, bottom = track.to_tlbr()
        # draw box
        frame = cv.rectangle(input_frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    return frame


threshold = 0.75
tracker = DeepSort(max_age=5)
vid = cv.VideoCapture('./kittens.mp4')

# get frame dimensions for output
ret, frame = vid.read()
h, w = frame.shape[:2]
out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*'MJPG'), 20.0, (w, h))
i = 1
while vid.isOpened():
    # read frame of video
    ret, frame = vid.read()
    if ret:
        # use frcnn model to detect cat and return important info about the frame
        boxes, scores, classes = frcnn_detect(frame)
        print("frame" + str(i))
        i += 1

        # get dimensions of frame
        h, w = frame.shape[:2]

        # converts coordinates from relative to absolute and matches with score and class for each tuple
        box_data = [([int(box[1] * w), int(box[0] * h), int((box[3] - box[1]) * w), int((box[2] - box[0]) * h)], score,
                class_) for box, score, class_ in zip(boxes, scores, classes)]
        # uses deepsort
        tracks = tracker.update_tracks(box_data, frame=frame)
        # get new scores
        scores = [bbox[1] for bbox in box_data]

        # draws box
        new_frame = draw_box(frame, tracks, scores)

        # outputs to file
        out.write(new_frame)
    else:
        break

vid.release()
out.release()
cv.destroyAllWindows()

"""
# https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1

import tensorflow as tf
import cv2 as cv
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

model = tf.saved_model.load('./faster_rcnn_resnet50_v1_640x640_1/')


# print(list(model.signatures.keys()))


# detect the cat in each frame using the pretrained model
def frcnn_detect(input_frame):
    #   frame_rgb = cv.cvtColor(input_frame, cv.COLOR_BGR2RGB)
    #  frame_rgb = frame_rgb.astype(np.float32) / 255.0

    # convert frame to right format
    frame_int = tf.cast(frame, dtype=tf.uint8)
    tensor_frame = tf.convert_to_tensor([frame_int])

    # use model
    detect = model.signatures['serving_default'](tensor_frame)

    boxes = detect['detection_boxes'].numpy()
    scores = detect['detection_scores'].numpy()
    classes = detect['detection_classes'].numpy()

    # filter out bad boxes and return a list of the indices for the good ones
    good_scores = scores > threshold

    # replace all info and return
    boxes, scores, classes = boxes[good_scores], scores[good_scores], classes[good_scores]

    return boxes, scores, classes


# draws correct box around cat
def draw_box(input_frame, tracks, scores):
    if len(tracks) > 0:
        # get the most confident box
        highest_conf_index = np.argmax(scores)
        track = tracks[highest_conf_index]
        left, top, right, bottom = track.to_tlbr()
        # draw box
        frame = cv.rectangle(input_frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
    return frame


threshold = 0.75
tracker = DeepSort(max_age=8)
vid = cv.VideoCapture('./kittens.mp4')

# get frame dimensions for output
ret, frame = vid.read()
h, w = frame.shape[:2]
out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*'MJPG'), 20.0, (w, h))
i = 1
while vid.isOpened():
    # read frame of video
    ret, frame = vid.read()
    if ret:
        # use frcnn model to detect cat and return important info about the frame
        boxes, scores, classes = frcnn_detect(frame)
        # so i can see that its actually working
        print("frame" + str(i))
        i += 1

        # dimensions of the frame
        h, w = frame.shape[:2]

        bbs = [([int(box[1] * w), int(box[0] * h), int((box[3] - box[1]) * w), int((box[2] - box[0]) * h)], score,
                class_) for box, score, class_ in zip(boxes, scores, classes)]
        scores = [bb[1] for bb in bbs]
        tracks = tracker.update_tracks(bbs, frame=frame)

        new_frame = draw_box(frame, tracks, scores)

        out.write(new_frame)
    else:
        break

# end everything
vid.release()
out.release()

cv.destroyAllWindows()
"""
