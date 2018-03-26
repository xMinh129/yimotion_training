import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import json
from keras.models import load_model
import cv2
import datetime
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId

# Establishing connection with Mlab
client = MongoClient('mongodb://test:test@ds159662.mlab.com:59662/medslack')
# Database name: medslack
db = client['medslack']

OUTPUT_PATH = '/Users/minhvu/Downloads/yimotion_training/outputs/'

current_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(current_path))

from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util

MODEL_NAME = 'facial_detection_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = current_path + '/' + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(current_path, 'training', 'facial_label_map.pbtxt')

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def run_detection_model(test_image_paths):
    # cropped_images = []
    all_emotion_labels = []
    for image_index, image_path in enumerate(test_image_paths):
        image = Image.open(image_path)
        image_name = os.path.basename(image_path).split('.jpg')[0]
        w, h = image.size
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = _load_image_into_numpy_array(image)
        # Actual detection.
        output_dict = _run_inference_for_single_image(image_np, detection_graph)
        # Find the detection boxes with best scores > 0.1
        best_scores = list(filter(lambda x: x > 0.1, output_dict['detection_scores']))
        filtered_detection_box = output_dict['detection_boxes'][:len(best_scores)].tolist()
        filtered_detection_masks = output_dict['detection_classes'][:len(best_scores)].tolist()
        # Create a meta data for the images
        _create_meta_data(image_name, filtered_detection_box, filtered_detection_masks)
        for i, j in enumerate(best_scores):
            cropped_image = _crop_image(output_dict, i, image, w, h)
            emotion_label = _run_emotion_detection(cropped_image)
            all_emotion_labels.append(emotion_label)
            _update_meta_data(image_name, i, emotion_label)
            # cropped_images.append(cropped_image)
            # Write image into assigned directory
            # cv2.imwrite('/Users/minhvu/Downloads/yimotion_training/object_detection/outputs/{1}_C_{2}.jpg'.format(image_name, image_index),
            #             cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))  # To use BGR instead of BGR in cv2
        num_positive_labels, num_negative_labels, emotion_labels, detection_boxes, data = _get_info_from_meta_data(image_name)
        _write_analysis_data_into_db(db, data)
        for index, each_detection in enumerate(detection_boxes):
            _draw_detection_box_on_image(image_np, image_name, each_detection, emotion_labels[index], w, h)
        print(
            "There are {0} students with positive emotions and {1} students with negative emotions out of {2} students detected in {3}"
                .format(num_positive_labels, num_negative_labels, len(emotion_labels), image_name))

    return all_emotion_labels


def _load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def _run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                print(real_num_detection)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def _crop_image(output_dict, detection_box_index, image, w, h):
    with tf.Session() as sess:
        cropped_img = sess.run(tf.image.crop_to_bounding_box(image,
                                                             int(output_dict['detection_boxes'][detection_box_index][
                                                                     0] * h),
                                                             int(output_dict['detection_boxes'][detection_box_index][
                                                                     1] * w),
                                                             int(output_dict['detection_boxes'][detection_box_index][
                                                                     2] * h -
                                                                 output_dict['detection_boxes'][detection_box_index][
                                                                     0] * h),
                                                             int(output_dict['detection_boxes'][detection_box_index][
                                                                     3] * w -
                                                                 output_dict['detection_boxes'][detection_box_index][
                                                                     1] * w)
                                                             ))
        return cropped_img


def _create_meta_data(image_name, detection_boxes, detection_labels):
    image_path = OUTPUT_PATH + image_name + '.txt'
    data = {
        "name": image_name,
        "detection_boxes": detection_boxes,
        "emotion_labels": detection_labels,
        "taken_at": str(datetime.datetime.now())
    }
    with open(image_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)


def _run_emotion_detection(img):
    model = load_model('/Users/minhvu/Downloads/yimotion_training/object_detection/model.h5')

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    img = cv2.resize(img, (64, 64))
    img = np.reshape(img, [1, 64, 64, 3])

    classes = model.predict_classes(img)

    return ['negative' if x < 1 else 'positive' for x in classes[0]]


def _update_meta_data(image_name, index_to_update, label):
    image_path = OUTPUT_PATH + image_name + '.txt'
    with open(image_path, 'r+') as f:
        data = json.loads(f.read())
        data['emotion_labels'][index_to_update] = label[0]
        f.seek(0)
        f.write(json.dumps(data, f, ensure_ascii=False))


def _get_info_from_meta_data(image_name):
    image_path = OUTPUT_PATH + image_name + '.txt'
    with open(image_path, 'r+') as f:
        data = json.loads(f.read())
        emotion_labels = data['emotion_labels']
        detection_boxes = data['detection_boxes']
        num_positive_labels = sum(label == 'positive' for label in emotion_labels)
        num_negative_labels = sum(label == 'negative' for label in emotion_labels)
        return num_positive_labels, num_negative_labels, emotion_labels, detection_boxes, data


def _draw_detection_box_on_image(img_np, image_name, detection_box, label, w, h):
    cv2.rectangle(img_np,
                  (int(detection_box[1] * w), int(detection_box[0] * h)),
                  (int(detection_box[3] * w), int(detection_box[2] * h)),
                  (0, 128, 0),
                  1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_np,
                label,
                (int(detection_box[1] * w), int(detection_box[2] * h)),
                font,
                0.3,
                (255, 255, 255),
                1)
    image_path = OUTPUT_PATH + 'analysed_' + image_name + '.jpg'
    cv2.imwrite(image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

def _write_analysis_data_into_db(db, data):
    find_class = db.classes.find_one({"class_name": "Differentiation"})
    find_class['statistics'].append(data)

    try:
        db.classes.save(find_class)
        print("Database updated")
    except:
        print("Failed to update database")


