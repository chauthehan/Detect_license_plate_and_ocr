import cv2
from lib_detection import load_model, detect_lp, im2single
import argparse
import os
from imutils import paths
# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', help='path to image')

# args = vars(ap.parse_args())

# img_path = args['image']

wpod_net_path = 'wpod-net_update1.json'
wpod_net = load_model(wpod_net_path)

for i, path in enumerate(paths.list_images('plate')):
    Ivehicle = cv2.imread(path)
    #cv2.imshow('', Ivehicle)
    #cv2.waitKey(0)
    Dmax = 608
    Dmin = 288

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)
    

    
    if lp_type == 1:
        if LpImg is not None:
            LpImg = cv2.cvtColor(LpImg[0], cv2.COLOR_RGB2BGR)
            
            cv2.imwrite('out/{}.jpg'.format(i), LpImg*255.0)
    if lp_type == 2:
        if LpImg is not None:
            LpImg = cv2.cvtColor(LpImg[0], cv2.COLOR_RGB2BGR)
            height, width, depth = LpImg.shape
            cv2.imwrite('out/{}_1.jpg'.format(i), LpImg[0:int(height/2), 0:width]*255.0)
            cv2.imwrite('out/{}_2.jpg'.format(i), LpImg[int(height/2):height, 0:width]*255.0)


        # Xử lý đọc biển đầu tiên, các bạn có thẻ sửa code để detect all biển số

    #     cv2.imshow("Bien so", cv2.cvtColor(LpImg[0],cv2.COLOR_RGB2BGR ))
    #     cv2.waitKey()
    # else:
    #     print('No lp found!')

    # cv2.destroyAllWindows()