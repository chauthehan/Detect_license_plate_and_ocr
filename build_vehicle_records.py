from config import dlib_front_rear_config as config 
from pyimagesearch.utils.tfannotation import TFAnnotation
from PIL import Image 
import tensorflow as tf 
import xml.etree.ElementTree as ET
import os 


# def main():
f = open(config.CLASSES_FILE, 'w')

for (k, v) in config.CLASSES.items():
    item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: '" + k +"'\n"
                "}\n")
    f.write(item)

f.close()

datasets = [
    ("train", config.TRAIN_XML, config.TRAIN_RECORD),
    ("test", config.TEST_XML, config.TEST_RECORD)
]

for (dType, inputPath, outputPath) in datasets:
    #build the soup

    print("[INFO] processing '{}'...".format(dType))
    contents = ET.parse(inputPath).find('images')

    writer = tf.io.TFRecordWriter(outputPath)
    total = 0
    
            
    for image in contents.findall('image'):
        # load the input image from disk as a TensorFlow object
        
        p = os.path.sep.join([config.BASE_PATH, image.attrib["file"]])
        encoded = tf.io.gfile.GFile(p, "rb").read()
        encoded = bytes(encoded)

        #load thee image from disk again, this time as a PIL object

        pilImage = Image.open(p)
        (w, h) = pilImage.size[:2]

        # parse the filename and encoding from the input path
        filename = image.attrib["file"].split(os.path.sep)[-1]
        encoding = filename[filename.rfind(".") + 1:]

        tfAnnot = TFAnnotation()
        tfAnnot.image = encoded
        tfAnnot.encoding = encoding
        tfAnnot.filename = filename
        tfAnnot.width = w
        tfAnnot.height = h
        #print(image)

        for box in image.findall("box"):
            #for box in image.select("box"):
            
            if 'ignore' in box.attrib:
                continue
            #print('pass')
            startX = max(0, float(box.attrib["left"]))
            startY = max(0, float(box.attrib["top"]))
            endX = min(w, float(box.attrib["width"]) + startX)
            endY = min(h, float(box.attrib["height"]) + startY)
            label = box.find("label").text

            xMin = startX/w
            xMax = endX/w 
            yMin = startY/h 
            yMax = endY/h 

            if xMin > xMax or yMin > yMax:
                #print('Wrong!')
                continue 
            tfAnnot.xMins.append(xMin)
            tfAnnot.xMaxs.append(xMax)
            tfAnnot.yMins.append(yMin)
            tfAnnot.yMaxs.append(yMax)
            tfAnnot.textLabels.append(label.encode("utf8"))
            tfAnnot.classes.append(config.CLASSES[label])
            tfAnnot.difficult.append(0)

            total += 1
        # encode the data point attributes using the TensorFlow
        # helper functions
        features = tf.train.Features(feature=tfAnnot.build())
        example = tf.train.Example(features=features)
        # add the example to the writer
        writer.write(example.SerializeToString())
    
    writer.close()
    print("[INFO] {} examples saved for '{}'".format(total, dType))
