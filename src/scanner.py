import cv2, os
import numpy as np

# 이미지 불러와서 변수 생성
ori_img = cv2.imread("img01.jpg")
filename, ext = os.path.splitext(os.path.basename("img01.jpg"))

src = []

# mouse callback handler 함수 생성
def mouse_handler(event, x, y, flags, param):
    # 마우스 좌측버튼
    if event == cv2.EVENT_LBUTTONUP:
        img = ori_img.copy()
        # 좌측버튼 클릭한 곳에 원 생성
        src.append([x, y])
        for xx, yy in src:
            cv2.circle(img, center = (xx, yy), radius = 5, color = (0, 255, 0), thickness = -1, lineType = cv2.LINE_AA)
        cv2.imshow("img", img)

        # perspective transform
        if len(src) == 4:
            src_np = np.array(src, dtype = np.float32)
        width = max(np.linalg.norm(src_np[0] - src_np[1]), np.linalg.norm(src_np[2] - src_np[3]))
        height = max(np.linalg.norm(src_np[0] - src_np[3]), np.linalg.norm(src_np[1] - src_np[2]))

        dst_np = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype = np.float32)

        # dst_np, src_np 값을 getPerspectiveTransform에 넣어준다.
        M = cv2.getPerspectiveTransform(src = src_np, dst = dst_np)
        # M 값을 warpPerspective에 넣어준다.
        result = cv2.warpPerspective(ori_img, M=M, dsize = (int(width), int(height)))

        cv2.imshow("result", result)
        cv2.imwrite("./result/%s_result%s" % (filename, ext), result)

cv2.namedWindow("img") # 윈도우 이름 설정
cv2.setMouseCallback("img", mouse_handler) # 윈도우에 마우스 콜백함수 설정

# 이미지 띄우기
cv2.imshow("img",ori_img)
cv2.waitKey(0)