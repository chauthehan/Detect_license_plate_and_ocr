# Detect_license_plate_and_ocr

Phase detect car:

Using the tensorflow 2 object detection api to detect vehicle, the model I have used is SSD+MobileNet v1 trained on COCO dataset. Follow this instruction to install the api: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-installation

![alt text](https://github.com/chauthehan/Detect_license_plate_and_ocr/blob/master/demo_images/detect.png)

Phase detect license plate:

Using pretrained WPOD-NET to extract the license plate from the result of phase 1.

About the wpod-net structures, see: 
https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf

![alt text](https://github.com/chauthehan/Detect_license_plate_and_ocr/blob/master/demo_images/lp.jpg)

Phase ocr:

Using crnn ocr model : https://arxiv.org/abs/1507.05717

link code train crnn: https://drive.google.com/drive/folders/1iZcVIpDfz1ne7TgHzxlXmJtnzK-pDThO?usp=sharing


![alt text](https://github.com/chauthehan/Detect_license_plate_and_ocr/blob/master/demo_images/ocr.jpg)


