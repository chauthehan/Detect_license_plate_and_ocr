import argparse
from lib_detection import load_model, detect_lp, im2single
from object_detection.utils import ops as utils_ops
from modeL_crnn import CRNN
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
import pathlib
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf 
import numpy as np 
import cv2 
import config_crnn as config
import imutils
from PIL import Image

def fastdecode(y_pred, chars):
    results_str = ""
    confidence = 0.0

    for i,one in enumerate(y_pred[0]):

        if one<config.NUM_CLASSES-1 and (i==0 or (one!=y_pred[0][i-1])):
            results_str+= chars[one]

    return results_str

def resize(image):
	width = 160#160
	height = 32 #32
	k1 = width/height
	k2 = image.shape[1]/image.shape[0]
	if k2 < k1:		
		resized = imutils.resize(image, height = height)

		zeros = np.zeros((height, width - resized.shape[1]))
		#zeros = zeros + 255
		results = np.concatenate((resized, zeros), axis=1)
		

	else:
		resized = imutils.resize(image, width = width)
		zeros = np.zeros((height - resized.shape[0], width))
		#zeros = zeros + 255
		results = np.concatenate((resized, zeros), axis=0)
	
	return results


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

def predict_ocr(img):
    gray_lp = resize(img)
    cv2.imshow('', gray_lp)
    cv2.waitKey(0)

    gray_lp = gray_lp.astype(np.float32)
    gray_lp = (gray_lp/255.0)*2.0-1.0            
    gray_lp = gray_lp.T
    print(gray_lp.shape)
    cv2.imshow('', gray_lp)
    cv2.waitKey(0)

    gray_lp = img_to_array(gray_lp, data_format=None)
    gray_lp = np.expand_dims(gray_lp, axis=0)
    predict = model_ocr.predict(gray_lp)
    predict = np.argmax(predict, axis=2)
    res = fastdecode(predict, dic)

    print('The results is: ', res)

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
    help='Path to input image')
args = vars(ap.parse_args())

print('[INFO] loading ssd-mobilenet model...')
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
#load ssd_mobilenet modle
detection_model = load_model_ssd(model_name)

#load wpod model
print('[INFO] loading wpod-net model...')
wpod_net_path = 'wpod-net_update1.json'
wpod_net = load_model(wpod_net_path)

print('[INFO] loading crnn model...')
model_ocr = CRNN.build(width=config.HEIGHT, height=config.WIDTH, depth=1,
		classes=config.NUM_CLASSES, training=0)
model_ocr.load_weights('epoch_100.hdf5')

#create dictionary
dic = {}
dic[0] = ' '
with open('dic.txt', encoding="utf-8") as dict_file:
	for i, line in enumerate(dict_file):
		(key, value) = line.strip().split('\t')
		dic[int(key)] = value
dict_file.close()

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
            
            if len(LpImg):
                
            #car license plate
                if lp_type==1:
                    if LpImg is not None:
                        cv2.imwrite('1.jpg', cv2.cvtColor(LpImg[0]*255.0,cv2.COLOR_RGB2BGR))
                    
                    gray_lp = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
                    predict_ocr(gray_lp)

                #motobyke license plate
                if lp_type==2:
                    if LpImg is not None:
                        cv2.imwrite('1.jpg', cv2.cvtColor(LpImg[0]*255.0,cv2.COLOR_RGB2BGR))
                    
                    gray_lp = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
                    top = gray_lp[0:int(gray_lp.shape[0]/2), :]
                    bot = gray_lp[int(gray_lp.shape[0]/2):gray_lp.shape[0], :]

                    gray_lp = np.concatenate((top, bot), axis=1)
                    gray_lp[:, 260:300] = cv2.blur(gray_lp[:, 260:300], (40, 40))
                
                    predict_ocr(gray_lp)