import numpy as np
from transformers import ViTFeatureExtractor, TFAutoModelForImageClassification
import cv2 as cv
from deep_sort_realtime.deepsort_tracker import DeepSort
from yolox.utils import postprocess
from yolox.exp import get_exp
import torch
from yolox.data.data_augment import ValTransform

# init ViT
vit_features = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
vit_model = TFAutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224')

# init Yolox
exp = get_exp(exp_file=None, exp_name="yolox-m")
yolox_model = exp.get_model()
yolox_model.eval()
# https://github.com/Megvii-BaseDetection/YOLOX/blob/main/README.md
checkpoint = torch.load("./yolox_m.pth", map_location="cpu")
yolox_model.load_state_dict(checkpoint["model"])

# init deepsort
tracker = DeepSort(max_age=5)

# open video
vid = cv.VideoCapture('./cat1.mp4')

ret, frame = vid.read()
h, w = frame.shape[:2]
out = cv.VideoWriter('output3.avi', cv.VideoWriter_fourcc(*'MJPG'), 20.0, (w, h))

while vid.isOpened():
    # get frame
    ret, frame = vid.read()
    if not ret:
        break

    
    # preprocess for YOLO
    preproc = ValTransform(legacy=False)
    img, _ = preproc(frame, input_size=(444,444), res=(2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).float()

    # use yolo model
    with torch.no_grad():
        outputs = yolox_model(img_tensor)

    # Post-process outputs
    predictions = postprocess(outputs, num_classes=exp.num_classes, conf_thre=0.8, nms_thre=0.45)
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy().tolist()

    if predictions is not None and all(prediction is not None for prediction in predictions):
        detections = []
        for det in predictions[0]:
            x_min, y_min, x_max, y_max, confidence, _, class_id = det
            x_min = max(0, x_min.item())
            y_min = max(0, y_min.item())
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            detections.append((bbox, confidence, class_id))
            
        # Update DeepSORT tracker
        print(frame.shape)
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if track.is_confirmed():
                bbox = track.to_tlbr()  # Get the bounding box in top-left, bottom-right format
                track_id = track.track_id
                cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        

        # Crop and preprocess for ViT
        print(x_min)
        print(x_max)
        cropped_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        print(cropped_image.shape)
        resized_cropped_image = cv.resize(cropped_image, (224, 224))
        vit_input = vit_features(resized_cropped_image, return_tensors="np")['pixel_values'][0]
        vit_input = np.expand_dims(vit_input, axis=0)

        # Classify with ViT
        vit_predictions = vit_model.predict(vit_input)
        vit_label = np.argmax(vit_predictions.logits, axis=-1)

        # Draw bounding boxes and labels
        cv.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv.putText(frame, f'Class: {class_id}, ViT: {vit_label}', (int(x_min), int(y_min) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    out.write(frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
out.release()
cv.destroyAllWindows()