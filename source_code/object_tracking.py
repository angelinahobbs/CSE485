import cv2
import numpy as np
from detection import ObjectDetection
import math

#SAMPLE 
# Import OpenCV library
#import cv2

# Load a cascade file for detecting faces
#faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Load image
#image = cv2.imread("dog.jpg")

# Convert into grayscale
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Look for faces in the image using the loaded cascade file
#faces = faceCascade.detectMultiScale(gray, 1.2, 5)
#for (x,y,w,h) in faces:
        # Create rectangle around faces
   #cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)

# Create the resizeable window
#cv2.namedWindow('Dog', cv2.WINDOW_NORMAL)

# Display the image
#cv2.imshow('Dog', image)

# Wait until we get a key
#k=cv2.waitKey(0)



od = ObjectDetection()
#dog_rec.mp4, 1, 2, 3
cap = cv2.VideoCapture("dog_rec.mp4")
count = 0
cp_prev_frame = []

tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    cp_cur_frame = []

    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        cp_cur_frame.append((cx, cy))
        #important frame debugging
        #print("FRAME N°", count, " ", x, y, w, h)

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #compare previous and current frame
    if count <= 2:
        for pt in cp_cur_frame:
            for pt2 in cp_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        cp_cur_frame_copy = cp_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in cp_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in cp_cur_frame:
                        cp_cur_frame.remove(pt)
                    continue

            if not object_exists:
                tracking_objects.pop(object_id)

        for pt in cp_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    print("Tracking objects")
    print(tracking_objects)


    print("CUR FRAME LEFT PTS")
    print(cp_cur_frame)


    cv2.imshow("Frame", frame)

    #copy of the points
    center_points_prev_frame = cp_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
