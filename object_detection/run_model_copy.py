import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import cv2
import base64
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
# import google.oauth2.credentials
# import google_auth_oauthlib.flow
#
# flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
#     '/Users/minhvu/Downloads/yimotion_training/object_detection/client_secrets.json',
#     scopes=['https://www.googleapis.com/auth/drive'])

current_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(current_path))



from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# What model to download.
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


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = 'test_images'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 8) ]
#
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
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


def run_model(image):
    image = Image.open(image)
    w, h = image.size
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    best_scores = list(filter(lambda x: x > 0.1, output_dict['detection_scores']))
    print(best_scores)
    # 3
    cropped_images = []
    for i, j in enumerate(best_scores):
        cropped_image = crop_image(output_dict, i, image, w, h)
        cropped_images.append(cropped_image)
        cv2.imwrite('/Users/minhvu/Downloads/yimotion_training/object_detection/outputs/image_{0}.jpg'.format(i),
                    cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)) # To use BGR instead of BGRA in cv2
        upload_to_drive(drive, 'yimotion_images', '/Users/minhvu/Downloads/yimotion_training/object_detection/outputs/image_{0}.jpg'.format(i))
    img = cropped_images[0]
    return img


def crop_image(output_dict, detection_box_index, image, w, h):
    with tf.Session() as sess:
        cropped_img = sess.run(tf.image.crop_to_bounding_box(image,
                                                             int(output_dict['detection_boxes'][detection_box_index][0] * h),
                                                             int(output_dict['detection_boxes'][detection_box_index][1] * w),
                                                             int(output_dict['detection_boxes'][detection_box_index][2] * h -
                                                                 output_dict['detection_boxes'][detection_box_index][0] * h),
                                                             int(output_dict['detection_boxes'][detection_box_index][3] * w -
                                                                 output_dict['detection_boxes'][detection_box_index][1] * w)
                                                             ))
        return cropped_img


def upload_to_drive(drive, folder_name, image_name):


    #Name of the folder where I'd like to upload images
    upload_folder = folder_name
    #Id of the folder where I'd like to upload images
    upload_folder_id = None

    #Check if folder exists. If not than create one with the given name
    #Check the files and folers in the root foled
    file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file_folder in file_list:
        if file_folder['title'] == upload_folder:
        	#Get the matching folder id
            upload_folder_id = file_folder['id']
            print 'Image is uploaded to EXISTING folder: ' + file_folder['title']
            #We need to leave this if it's done
            break
        else:
            #If there is no mathing folder, create a new one
            file_new_folder = drive.CreateFile({'title': upload_folder,
                "mimeType": "application/vnd.google-apps.folder"})
            file_new_folder.Upload() #Upload the folder to the drive
            print 'New folder created: ' + file_new_folder['title']
            upload_folder_id = file_new_folder['id'] #Get the folder id
            print 'Image is uploaded to the NEW folder: ' + file_new_folder['title']
            break #We need to leave this if it's done

    #Create new file in the upload_folder
    file_image = drive.CreateFile({"parents":  [{"kind": "drive#fileLink","id": upload_folder_id}]})
    file_image.SetContentFile(image_name) #Set the content to the taken image
    file_image.Upload() # Upload it

