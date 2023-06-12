import cv2
import numpy as np

img = cv2.imread('./res/01.jpeg')
rows, cols = img.shape[:2]
center = (cols/2, rows/2)

# 1. 회전을 위한 변환 행렬 구하기
# 회전축 : 중앙, 각도 : 45, 배율 : 0.5

# mtrx = cv2.getRotationMatrix2D(center, angle, scale)
# center: 회전축 중심 좌표 (x, y)
# angle: 회전할 각도, 60진법
# scale: 확대 및 축소비율

m45 = cv2.getRotationMatrix2D(center, 45, 0.5)

img45 = cv2.warpAffine(img, m45, (cols, rows))

m90 = cv2.getRotationMatrix2D(center, 90, 2)

img90 = cv2.warpAffine(img, m90, (rows, cols))

cv2.imshow('origin', img)
# cv2.imshow('45', img45)
cv2.imshow('90', img90)

cv2.waitKey()
cv2.destroyAllWindows()