"""
This code uses the onnx model to detect faces from live video or cameras.
"""
import os
import time

import cv2
import numpy as np
import onnx
import imagenet_classes as class_name

# onnx runtime
import onnxruntime as ort

#ShuffleNet PARAM
input_height=224
input_width=224
input_channel=3
mean = [0.485, 0.456, 0.406]
var = [0.229, 0.224, 0.225]
onnx_path = "models/shufflenet.onnx"

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

path = "preprocess_shufflenet_dataset/img"
sum = 0
listdir = os.listdir(path)
sum = 0
for file_path in listdir:
    img_path = os.path.join(path, file_path)
    orig_image = cv2.imread(img_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_width, input_height)) / 255.0
    image = (image - mean) / var
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    time_time = time.time()
    logits = ort_session.run(None, {input_name: image})
    print("cost time:{}".format(time.time() - time_time))
    cls = np.argmax(logits)
    print("class for " + file_path + " is " + class_name.class_names[cls])
