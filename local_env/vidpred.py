import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'giraffe.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'containertrain','best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

class_name_dict = ["0"
, "1"
, "2"
, "3"
, "4"
, "5"
, "6"
, "7"
, "8"
, "9"
, "A"
, "B"
, "C"
, "D"
, "E"
, "F"
, "G"
, "H"
, "I"
, "J"
, "K"
, "L"
, "M"
, "N"
, "O"
, "P"
, "R"
, "S"
, "T"
, "U"
, "V"
, "W"
, "X"
, "Y"
, "Z"]

while ret:

    results = model.predict(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            """ cv2.putText(frame, class_id, (int(x1), int(y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA) """

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
