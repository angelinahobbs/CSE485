import os
import cv2
from ultralytics import YOLO

# video paths
# TODO: make this a relative path so its not hard coded everytime
video_input_path = r"C:\Users\lukek\College\Capstone\Dataset\Videos\BrownBear.mp4"
video_output_path = '{}_out.mp4'.format(video_input_path)

# reads videos frame by frame and evaluates them using the model
video_capture = cv2.VideoCapture(video_input_path)
ret, frame = video_capture.read()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'MP4V'), int(video_capture.get(cv2.CAP_PROP_FPS)), (W, H))

# change the train number with the most up-to-date model you are testing
model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

# breaks video into frames and adds rectangle and text over any objects detected by the model
while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = video_capture.read()

video_capture.release()
out.release()
cv2.destroyAllWindows()