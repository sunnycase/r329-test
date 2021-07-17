import numpy as np
import sys
import os
import cv2

img_dir='./img/'
label_file='./label.txt'

#ShuffleNet PARAM
input_height=224
input_width=224
input_channel = 3
mean = [0.485, 0.456, 0.406]
var = [0.229, 0.224, 0.225]

label_data = open(label_file)
filename_list = []
label_list = []
for line in label_data:
    filename_list.append(line.rstrip('\n').split(' ')[0])
    label_list.append(int(line.rstrip('\n').split(' ')[1]))
label_data.close()
img_num = len(label_list)

images = np.zeros([img_num, input_height, input_width, input_channel], np.float32)
for file_name, img_idx in zip(filename_list, range(img_num)):
    image_file = os.path.join(img_dir, file_name)
    orig_image = cv2.imread(image_file)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_width, input_height)) / 255.0
    image = (image - mean) / var
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    images[img_idx] = image

np.save('dataset.npy', images)

labels = np.array(label_list)
np.save('label.npy', labels)