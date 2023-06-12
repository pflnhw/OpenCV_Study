import cv2
import numpy as np

path = './res/01.jpeg'
img = cv2.imread(path)
rows, cols = img.shape[:2]

# 변환 전, 후 각 3개의 좌표 생성
pts1 = np.float32([[100, 50], [200, 50], [100, 200]])
pts2 = np.float32([[80, 70], [210, 60], [250, 120]])

# 변환 전 좌표를 이미지에 표시
cv2.circle(img, (100, 50), 5, (255, 0, 0), -1)
cv2.circle(img, (200, 50), 5, (0, 255, 0), -1)
cv2.circle(img, (100, 200), 5, (0, 0, 255), -1)

# 변환 행렬
mtrx = cv2.getAffineTransform(pts1, pts2)

# 어핀 변환
dst = cv2.warpAffine(img, mtrx, (int(cols*1.5), rows))

# 결과 출력
cv2.imshow('origini', img)
cv2.imshow('affine', dst)
cv2.waitKey()
cv2.destroyAllWindows()