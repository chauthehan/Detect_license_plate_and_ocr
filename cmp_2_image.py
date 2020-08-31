from imutils import paths
import cv2
import os
import numpy as np 

def dhash(image, hashSize=8):
	# resize the input image, adding a single column (width) so we
	# can compute the horizontal gradient
	resized = cv2.resize(image, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

path = 'TEST/'
paths = list(paths.list_images(path))
count = 0
for i, path1 in enumerate(paths):
    image1 = cv2.imread(path1)
    hash0 = dhash(image1)
    print('hassssh', hash0)

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #hash0 = imagehash.average_hash(image)
    for path2 in paths:
        if path1 != path2:
            image2 = cv2.imread(path2)           
            try:
                hash1 = dhash(image2)
            except:
                continue
            if hash0 == hash1:
                print('FOUND')
                count = count+1
print(count)



