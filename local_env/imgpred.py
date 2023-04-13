import os
import argparse
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import easyocr

""" 
parser = argparse.ArgumentParser
parser.add_argument('--source')
arg = parser.parse_args()
 """

reader = easyocr.Reader(['en'])

img=cv2.imread("cont2.jpg")
model_path = os.path.join('.', 'runs', 'detect', 'containernumregion','best.pt')

height,width,_=img.shape

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.3

""" class_name_dict = ["0"
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
, "Z"] """

# new_width = 800
# new_height = 800
# new_size = (new_width, new_height)
# imgresized= cv2.resize(img, new_size)

class_name_dict = ["region"]


results = model.predict(img)[0]

for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # containerNum =[class_name_dict[int(class_id)]]
if score > threshold:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
        cv2.putText(img, class_name_dict[int(class_id)], (int(x1), int(y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA) 
        


# Crop the detected object
center_x = int(x1 + (x2-x1)/2)
center_y = int(y1 + (y2-y1)/2) 
w = int(x2 - x1)         
h = int(y2 - y1)
x = int(center_x - w / 2)
y = int(center_y - h / 2)
imgcrop = img[y:y+h, x:x+w]



# imgcrop = img[y:y+h, x:x+w]
print("the shape of the original image is:", img.shape)         
print("the shape of the cropped image is :", imgcrop.shape) 

""" filename = 'savedImage.jpg' """
  



imgcropgray = cv2.cvtColor(imgcrop, cv2.COLOR_BGR2GRAY)

_, imgcropthresh = cv2.threshold(imgcropgray, 64, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("output", imgcropthresh)
output = reader.readtext(imgcropgray)
cv2.imshow("output", imgcropgray)
containernumber=""
for out in output:
            
            text_bbox, text, text_score = out
            # if text_score > 0.4:
            containernumber+=text
            
containernumber = containernumber.upper()          
print(containernumber)
# Using cv2.imwrite() method
# Saving the image
""" cv2.imwrite(filename, img) """
cv2.putText(img, containernumber , (int(width*0.7), int(height*0.9)),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA) 
cv2.imshow("output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

