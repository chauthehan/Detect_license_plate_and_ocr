import cv2
from lib_detection import load_model, detect_lp, im2single

img_path = "lp.jpg"

wpod_net_path = 'wpod-net_update1.json'
wpod_net = load_model(wpod_net_path)

Ivehicle = cv2.imread(img_path)
cv2.imshow("Anh goc",Ivehicle)
cv2.waitKey(0)

Dmax = 608
Dmin = 288

# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)

_ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

if LpImg is not None:

    # Xử lý đọc biển đầu tiên, các bạn có thẻ sửa code để detect all biển số

    cv2.imshow("Bien so", cv2.cvtColor(LpImg[0],cv2.COLOR_RGB2BGR ))
    cv2.waitKey()
else:
    print('No lp found!')
    
cv2.destroyAllWindows()