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
