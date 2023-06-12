import cv2
import numpy as np

img = cv2.imread('./res/01.jpeg')
rows, cols = img.shape[:2]

# 1. 라디안 각도 계산(60진법을 호도법으로 변경)
d45 = 45.0 * np.pi / 180
d90 = 90.0 * np.pi / 180

# 2. 회전을 위한 변환 행렬 생성
m45 = np.float32([[np.cos(d45), -1*np.sin(d45), rows//2], [np.sin(d45), np.cos(d45), -1*cols//4]])
m90 = np.float32([[np.cos(d90), -1*np.sin(d90), rows//2], [np.sin(d90), np.cos(d90), 0]])

# 3. 회전 변환 행렬 적용
r45 = cv2.warpAffine(img, m45, (cols, rows))
r90 = cv2.warpAffine(img, m90, (rows, cols))

# 4. 결과
cv2.imshow('original', img)
cv2.imshow('45', r45)
cv2.imshow('90', r90)

cv2.waitKey()
cv2.destroyAllWindows()