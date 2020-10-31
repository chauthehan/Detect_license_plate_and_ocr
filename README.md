# Detect_license_plate_and_ocr

Phase detect car:
Using the tensorflow 2 object detection api to detect vehicle, using model SSD+MobileNet trained on COCO dataset. Follow this instruction to install the api: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-installation

![alt text](https://github.com/chauthehan/Detect_license_plate_and_ocr/blob/master/demo_images/car.png)

Phase detect license plate:

Using pretrained WPOD-net to extract the license plate from the result of phase 1.

![alt text](https://github.com/chauthehan/Detect_license_plate_and_ocr/blob/master/demo_images/lp.png)
