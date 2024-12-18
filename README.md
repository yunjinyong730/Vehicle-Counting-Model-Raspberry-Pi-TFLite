# Vehicle-Counting-Model-Raspberry-Pi-TFLite

![이미지](https://github.com/user-attachments/assets/9ef55400-5c48-4401-9f86-e934ab98201c)

- **프로젝트에 사용할 대표 이미지 (차량 예상 인식 수 2대)**

## YOLOv3-416 모델로 이미지 상의 차량 수 탐색 in Raspberry Pi

```kotlin
import cv2
import numpy as np
import time

min_confidence = 0.5
margin = 30
file_name = "image/parking_02.jpg"

# Load Yolo
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
classes = []
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Loading image
start_time = time.time()
img = cv2.imread(file_name)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # Filter only 'car'
        if class_id == 2 and confidence > min_confidence:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))

indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
color = (0, 255, 0)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = '{:,.2%}'.format(confidences[i])
        print(i, label)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
        
text = "Number of Car is : {} ".format(len(indexes))
cv2.putText(img, text, (margin, margin), font, 2, color, 2)

cv2.imshow("Number of Car - "+file_name, img)

end_time = time.time()
process_time = end_time - start_time
print("=== A frame took {:.3f} seconds".format(process_time))

cv2.waitKey(0)
cv2.destroyAllWindows()
```

[화면 기록 2024-12-18 오후 9.33.35.mov](https://prod-files-secure.s3.us-west-2.amazonaws.com/3de099d9-93f3-4629-a573-b788330d4c5a/6a6fe52c-9d1a-4339-98ae-2923817057b3/%E1%84%92%E1%85%AA%E1%84%86%E1%85%A7%E1%86%AB_%E1%84%80%E1%85%B5%E1%84%85%E1%85%A9%E1%86%A8_2024-12-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.33.35.mov)

## YOLOv3-TINY 모델로 경량화 in Raspberry Pi

![스크린샷 2024-12-18 오후 10.12.01.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3de099d9-93f3-4629-a573-b788330d4c5a/fe807d01-80dc-406f-b49c-f7cd6f068c43/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-12-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_10.12.01.png)

![스크린샷 2024-12-18 오후 10.12.43.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/3de099d9-93f3-4629-a573-b788330d4c5a/78add1c6-c5fc-4aa3-a2e9-d52a5be4846b/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-12-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_10.12.43.png)

```kotlin
import cv2
import numpy as np
import time

min_confidence = 0.5
margin = 30
file_name = "image/parking_02.jpg"

# Load Yolo
net = cv2.dnn.readNet("yolo/yolov3-tiny.weights", "yolo/yolov3-tiny.cfg")
classes = []
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Loading image
start_time = time.time()
img = cv2.imread(file_name)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # Filter only 'car'
        if class_id == 2 and confidence > min_confidence:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))

indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
color = (0, 255, 0)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = '{:,.2%}'.format(confidences[i])
        print(i, label)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
        
text = "Number of Car is : {} ".format(len(indexes))
cv2.putText(img, text, (margin, margin), font, 2, color, 2)

cv2.imshow("Number of Car - "+file_name, img)

end_time = time.time()
process_time = end_time - start_time
print("=== A frame took {:.3f} seconds".format(process_time))

cv2.waitKey(0)
cv2.destroyAllWindows()

```

[화면 기록 2024-12-18 오후 9.54.08.mov](https://prod-files-secure.s3.us-west-2.amazonaws.com/3de099d9-93f3-4629-a573-b788330d4c5a/69ef46c0-399a-4b2c-8285-c7bc6424bb2c/%E1%84%92%E1%85%AA%E1%84%86%E1%85%A7%E1%86%AB_%E1%84%80%E1%85%B5%E1%84%85%E1%85%A9%E1%86%A8_2024-12-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.54.08.mov)

### ***14.626 sec → 2.070 sec (감축 성공!)***

## Tensorflow Lite로 모델 경량화

```kotlin
import cv2
import numpy as np
import time
import re
from tflite_runtime.interpreter import Interpreter

min_confidence = 0.5
margin = 30
file_name = "image/parking_02.jpg"
label_name = "coco_labels.txt"
model_name = "detect.tflite"
number_car = 0

def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  print(labels)
  return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

# Load tflite
labels = load_labels(label_name)
interpreter = Interpreter(model_name)
interpreter.allocate_tensors()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

# Loading image
start_time = time.time()
img = cv2.imread(file_name)
height, width, channels = img.shape
image = cv2.resize(img, (300, 300)) 

# Detecting objects
outs = detect_objects(interpreter, image, min_confidence)

font = cv2.FONT_HERSHEY_PLAIN
color = (0, 255, 0)
for out in outs:
    if out['class_id'] == 2 and out['score'] > min_confidence:
        number_car += 1
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = out['bounding_box']
        xmin = int(xmin * width)
        xmax = int(xmax * width)
        ymin = int(ymin * height)
        ymax = int(ymax * height)
        
        label = '{:,.2%}'.format(out['score'])
        print(number_car, label)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, label, (xmin, ymin - 10), font, 1, color, 2)
        
text = "Number of Car is : {} ".format(number_car)
cv2.putText(img, text, (margin, margin), font, 2, color, 2)

cv2.imshow("Number of Car - "+file_name, img)

end_time = time.time()
process_time = end_time - start_time
print("=== A frame took {:.3f} seconds".format(process_time))

cv2.waitKey(0)
cv2.destroyAllWindows()
```

[화면 기록 2024-12-18 오후 11.13.37.mov](https://prod-files-secure.s3.us-west-2.amazonaws.com/3de099d9-93f3-4629-a573-b788330d4c5a/6651cab9-0135-45bd-ac2a-6350cd694025/%E1%84%92%E1%85%AA%E1%84%86%E1%85%A7%E1%86%AB_%E1%84%80%E1%85%B5%E1%84%85%E1%85%A9%E1%86%A8_2024-12-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_11.13.37.mov)

### ***2.070 sec → 0.605 sec (감축  또 성공!!)***
