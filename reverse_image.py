from imutils import paths
import cv2
import os

images = paths.list_images('out/')
for image in images:
    name = os.path.split(image)[-1]
    name = name.split('.')[0]
    name = name + '.txt'
    path = os.path.join('out', name)
    f = open(path, 'w+')
    f.close()
    
