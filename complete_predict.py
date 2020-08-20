import argparse
from lib_detection import load_model, detect_lp, im2single
from object_detection.utils import ops as utils_ops
import pathlib
import tensorflow as tf 
import numpy as np 
import cv2 

def load_model_ssd(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    #Download a file from URL if it not already in the cache
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)
    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy()
                    for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
    help='Path to input image')
args = vars(ap.parse_args())


model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
#load ssd_mobilenet modle
detection_model = load_model_ssd(model_name)

#load wpod model
wpod_net_path = 'wpod-net_update1.json'
wpod_net = load_model(wpod_net_path)

#read the image and convert to pillow format to detect
image_cv = cv2.imread(args['image'])
pil_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

output_dict = run_inference_for_single_image(detection_model, np.array(pil_image))

Dmax = 608
Dmin = 288

score_thresh = 0.5
for i in range(output_dict['detection_boxes'].shape[0]):
    if output_dict['detection_classes'][i] in (3,4,6,8):
        if output_dict['detection_scores'][i] > score_thresh:
            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]

            im_height, im_width, _ = image_cv.shape
            (tl_x, br_x, tl_y, br_y) = (int(xmin * im_width), int(xmax * im_width),
                                            int(ymin * im_height), int(ymax * im_height))
            single_vehicle = image_cv[tl_y:br_y, tl_x:br_x]
            ratio = float(max(single_vehicle.shape[:2])) / min(single_vehicle.shape[:2])
            side = int(ratio*Dmin)
            bound_dim = min(side, Dmax)
            
            _, LpImg, lp_type = detect_lp(wpod_net, im2single(single_vehicle), bound_dim, lp_threshold=0.5)
            
            if LpImg is not None:

                # Xử lý đọc biển đầu tiên, các bạn có thẻ sửa code để detect all biển số

                cv2.imshow("Bien so", cv2.cvtColor(LpImg[0],cv2.COLOR_RGB2BGR ))
                cv2.waitKey()
            
            image_cv = cv2.rectangle(image_cv, (tl_x, tl_y), (br_x, br_y), color=(0,255,0), thickness=2)

cv2.imshow('', image_cv)
cv2.waitKey(0)
