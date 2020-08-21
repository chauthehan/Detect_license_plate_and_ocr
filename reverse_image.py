from imutils import paths
import cv2
import os

images = paths.list_images('out_reverse/')
for image in images:
    name = os.path.split(image)[-1]

    img = cv2.imread(image)

    img = cv2.flip(img, 1)
    path = os.path.join('flip', name)

    cv2.imwrite(path, img)
    
