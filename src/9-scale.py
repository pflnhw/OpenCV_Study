import cv2
import numpy as np

img = cv2.imread('./res/4star.jpg')
height, width = img.shape[:2]

# array, list
# 다수의 자료를 포함, 순서가 있고 시작이 0번, 인덱스가 숫자형식의 번호로 구성


smallFactor = 0.5
bigFactor = 2

# 1. 0.5배 축소 변환 행렬
m_small = np.float32([[smallFactor, 0, 0], [0, smallFactor, 0]])

small_dsize = (int(width*smallFactor), int(height*smallFactor))
dst1 = cv2.warpAffine(img, m_small, small_dsize)

# 2. 2배 변환 행렬
m_big = np.float32([[bigFactor, 0, 0], [0, bigFactor, 0]])

big_dsize = (int(width*bigFactor), int(height*bigFactor))
dst2 = cv2.warpAffine(img, m_big, big_dsize)

# 3. 0.5배 축소 변환 INTER_AREA
dst3 = cv2.warpAffine(img, m_small, small_dsize, None, cv2.INTER_AREA)

# 4. 2배 변환 INTER_CUBIC
dst4 = cv2.warpAffine(img, m_big, big_dsize, None, cv2.INTER_CUBIC)

cv2.imshow('original', img)
# cv2.imshow('small', dst1)
# cv2.imshow('big', dst2)
cv2.imshow('small_INTER_AREA', dst3)
cv2.imshow('big_INTER_CUBIC', dst4)

cv2.waitKey()
cv2.destroyAllWindows()