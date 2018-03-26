import os
import cv2
from io import BytesIO
import hashlib
os.getcwd()


collection = "/Users/minhvu/Downloads/yimotion_training/eval_images"
for i, filename in enumerate(os.listdir(collection)):
    if filename != '.DS_Store':
        img = cv2.imread(collection + '/' + filename)
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sha1 = hashlib.sha1(BytesIO(RGB_img).getvalue()).hexdigest()
        os.rename(collection + '/' + filename, collection + '/' + "image_" + sha1 + ".jpg")