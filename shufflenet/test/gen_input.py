import cv2
import numpy as np

input_height=224
input_width=224
input_channel=3

img_path = "../deploy/build/ILSVRC2012_val_00000004.bmp"

orig_image = cv2.imread(img_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (input_width, input_height))
image = (image - 127.0) / 1
image = np.expand_dims(image, axis=0)
image = image.astype(np.int8)

image.tofile("input.bin")
print("save to input.bin OK")
