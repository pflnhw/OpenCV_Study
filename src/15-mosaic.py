import cv2
import numpy as np

# 모자이크 처리
rate = 15
winTitle = 'src'
img = cv2.imread('./res/01.jpeg')

while True:
    # 마우스 드래그 이벤트가 발생 후 종료될때까지 블럭
    x, y, w, h = cv2.selectROI(winTitle, img, False)
    if w and h:
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (w//rate, h//rate))
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
        img[y:y+h, x:x+w] = roi
        cv2.imshow(winTitle, img)
    else:
        break

cv2.destroyAllWindows()