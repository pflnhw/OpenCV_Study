import cv2
import numpy as np

img = cv2.imread('./res/4star.jpg')
rows, cols = img.shape[0:2] # 영상의 크기

# 이동할 픽셀 거리
dx, dy = 100, 50

# 1. 변환 행렬 생성
mtrx = np.float32([[1, 0, dx], [0, 1, dy]])

# 2. 단순 이동
dst = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy))

# 3. 탈락된 외곽 픽셀 파란색으로 보정
dst2 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255, 0, 0))

# 4. 탈락된 외곽 픽셀 원본의 픽셀로 보정
dst3 = cv2.warpAffine(img, mtrx, (cols+dx, rows+dy), None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

cv2.imshow('original', img)
# cv2.imshow('trans', dst)
# cv2.imshow('trans', dst2)
cv2.imshow('trans', dst3)

cv2.waitKey(0)
cv2.destroyAllWindows()