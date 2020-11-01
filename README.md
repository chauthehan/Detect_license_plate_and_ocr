# Detect_license_plate_and_ocr

Phase detect car:

Using the tensorflow 2 object detection api to detect vehicle, the model I have used is SSD+MobileNet v1 trained on COCO dataset. Follow this instruction to install the api: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-installation

![alt text](https://github.com/chauthehan/Detect_license_plate_and_ocr/blob/master/demo_images/car.png)

Phase detect license plate:

Using pretrained WPOD-NET to extract the license plate from the result of phase 1.

About the wpod-net structures, see: 
https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf

![alt text](https://github.com/chauthehan/Detect_license_plate_and_ocr/blob/master/demo_images/lp.png)
