import cv2
import numpy as np
import imagenet_classes as class_name

REF_CLS = ""

logits = np.reshape(np.fromfile("test/output.bin", dtype=np.int8), [1,1000])
cls = np.argmax(logits)

print("class is " + class_name.class_names[cls])
