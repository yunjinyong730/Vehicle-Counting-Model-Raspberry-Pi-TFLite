# Vehicle-Counting-Model-Raspberry-Pi-TFLite
- **프로젝트에 사용할 대표 이미지 (차량 예상 인식 수 2대)**

![parking_02.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/3de099d9-93f3-4629-a573-b788330d4c5a/3f809c5f-ce44-4987-9df6-daf3d2c2c83b/parking_02.jpg)

## YOLOv3-416 모델로 이미지 상의 차량 수 탐색 in Raspberry Pi

```python
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

- **초기 설정**: 탐지 임계값(`min_confidence`), 출력 텍스트 여백, 이미지 파일 경로를 정의
- **YOLO 모델 로드**:
    - `weights`와 `cfg` 파일로 YOLO 모델을 불러옴
    - COCO 데이터셋의 클래스 이름(`coco.names`)을 읽어 객체를 분류
- **이미지 로드 및 전처리**:
    - 이미지를 불러와 크기를 얻음
    - YOLO가 요구하는 입력 형식(정규화 및 크기 변경)으로 변환
- **객체 탐지**:
    - 모델로부터 출력 계층 결과를 가져옴
    - 신뢰도와 클래스 점수를 바탕으로 자동차(class ID 2)만 필터링
    - 탐지된 객체의 바운딩 박스 좌표와 신뢰도 저장
- **중복 제거**:
    - Non-Maximum Suppression(NMS)으로 중복 바운딩 박스를 제거해 최적의 탐지 결과 선택
- **결과 시각화**:
    - 탐지된 자동차의 바운딩 박스와 신뢰도를 이미지에 표시
    - 탐지된 자동차 개수를 텍스트로 추가
- **성능 측정 및 출력**:
    - 탐지에 걸린 시간을 계산해 출
    - 결과 이미지를 화면에 표시



https://github.com/user-attachments/assets/cdcbacb5-9c62-453d-bfd9-f743f7d5c7f5



## YOLOv3-TINY 모델로 경량화 in Raspberry Pi

기존 모델인 YOLOv3-416은 65.86Bn인 것에 비해, YOLOv3-tiny모델은 5.56Bn으로 **연산량이 약 88% 감소하여 경량화**되었다고 볼 수 있음

이로 인해 YOLOv3-tiny 모델은 속도가 크게 향상되어 실시간 처리가 요구되는 애플리케이션에 더 적합하지만, 상대적으로 정확도는 낮아질 수 있음

<img width="652" alt="2" src="https://github.com/user-attachments/assets/ecce2121-8d6b-4cf0-9c2d-76d35bdc0046" />

실제 라즈베리파이에서 모델별로 메모리 크기를 보면

YOLOv3-416 모델의 cfg와 weights 파일이 YOLOv3-tiny 모델보다 큰 것을 알 수 있다

<img width="1022" alt="3" src="https://github.com/user-attachments/assets/dd1db16f-6a54-4754-a8c1-94aeb6fc8b67" />

코드는 모델을 제외하고 기존과 같다

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

### ***14.626 sec → 2.070 sec (감축 성공!)***



https://github.com/user-attachments/assets/d77b2156-9581-4a0d-ae0c-a0bdcd99c037



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

### **초기 설정**

- `min_confidence`: 탐지 신뢰도의 임계값 (0.5 이상만 유효)
- `margin`: 결과 텍스트의 여
- `file_name`: 처리할 이미지 경로.
- `label_name`, `model_name`: COCO 클래스 레이블 파일과 TensorFlow Lite 모델 파일 경로
- `number_car`: 탐지된 자동차 개수를 저장할 변수

### **레이블 로드**

- **`load_labels()`** 함수는 COCO 클래스 레이블 파일을 읽어 레이블을 딕셔너리로 반환
    - 파일에 인덱스와 이름이 함께 있으면 이를 분리
    - 레이블 파일이 단순 문자열로만 되어 있어도 처리 가능

### **TensorFlow Lite 입력 및 출력 처리**

- **`set_input_tensor()`**: TFLite 모델의 입력 텐서를 설정
    - 입력 텐서의 인덱스를 가져와 이미지 데이터를 채웁니다
- **`get_output_tensor()`**: 특정 출력 텐서를 가져오는 함수
    - 모델의 출력 계층에서 필요한 데이터를 가져옵니다
- **`detect_objects()`**: 모델을 호출해 객체를 탐지
    - 출력된 바운딩 박스, 클래스 ID, 신뢰도 등을 기준으로 결과를 정리
    - 신뢰도가 임계값 이상인 객체만 반환

### **모델 초기화**

- COCO 레이블을 로드하고 TensorFlow Lite 모델(`detect.tflite`)을 초기
- 모델 입력 크기를 가져와 이미지 크기 조정에 활용

### **이미지 전처리**

- 이미지를 읽고 원본 크기(`height`, `width`)를 추출
- TFLite 모델이 요구하는 입력 크기(300x300)로 이미지를 리사이즈

### **객체 탐지 및 결과 분석**

- **탐지 루프**:
    - `detect_objects()`의 결과를 순회하며 각 객체를 처리
    - 탐지된 클래스가 자동차(`class_id == 2`)이고 신뢰도가 `min_confidence` 이상인 경우만 처리
- **바운딩 박스 변환**:
    - 상대 좌표로 출력된 바운딩 박스를 원본 이미지 크기로 변환
    - 이미지 해상도를 기준으로 절대 좌표(`xmin`, `ymin`, `xmax`, `ymax`) 계
- **결과 저장 및 표시**:
    - 바운딩 박스를 원본 이미지에 그립니다
    - 신뢰도를 텍스트로 표시

### **결과 시각화**

- **자동차 개수 표시**:
    - 탐지된 자동차 개수를 텍스트로 이미지 상단에 출력
- **이미지 디스플레이**:
    - `cv2.imshow()`를 사용해 이미지 출력
    - 키 입력이 감지되면 창을 닫고 프로그램 종료

### **성능 측정**

- 전체 탐지 및 처리에 걸린 시간을 측정해 출력

### ***2.070 sec → 0.605 sec (감축  또 성공!!)***

https://github.com/user-attachments/assets/191413a5-41da-437e-a285-a78fdf3754e1

