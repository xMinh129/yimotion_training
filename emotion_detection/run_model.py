from keras.models import load_model
import cv2
import numpy as np

model = load_model('model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('dataset/evaluation/test_image_5.jpg')
img = cv2.resize(img,(64,64))
img = np.reshape(img,[1,64,64,3])

classes = model.predict_classes(img)

print 'negative' if x < 1 else 'positive' for x in classes[0]