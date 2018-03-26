import glob
from object_detection.run_detection_model import run_detection_model

PATH_TO_TEST_IMAGES_DIR = '/Users/minhvu/Downloads/yimotion_training/eval_images/*.jpg'
TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR)

emotion_labels = run_detection_model(TEST_IMAGE_PATHS)
print(emotion_labels)