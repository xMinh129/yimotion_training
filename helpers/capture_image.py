import cv2
from io import BytesIO
import hashlib
import time

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 0.05)
cv2.namedWindow("detection")
img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("detection", frame)
    sha1 = hashlib.sha1(BytesIO(frame).getvalue()).hexdigest()
    img_name = "/Users/minhvu/Downloads/yimotion_training/eval_images/image_{0}.jpg".format(sha1)
    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))

    k = cv2.waitKey(1)

    # ESC pressed
    if k%256 == 27:
        print("Escape hit, closing...")
        break
    # Put a break of 10 seconds between images

    img_counter += 1

cam.release()

cv2.destroyAllWindows()
