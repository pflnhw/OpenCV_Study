import cv2
import numpy as np

path = './res/01.jpeg'
img = cv2.imread(path)
rows, cols = img.shape[:2]

# 원근 변환 전 4개의 좌표
# 왼쪽위(0, 0), 왼쪽아래(0, h), 오른쪽위(w, 0), 오른쪽아래(w, h)
pts1 = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
pts2 = np.float32([[100, 50], [10, rows-50], [cols-100, 50], [cols-10, rows-50]])

# 변환 전 좌표를 표시
cv2.circle(img, (0, 0), 10, (255, 0, 0), -1)
cv2.circle(img, (0, rows), 10, (255, 0, 0), -1)
cv2.circle(img, (cols, 0), 10, (255, 0, 0), -1)
cv2.circle(img, (cols, rows), 10, (255, 0, 0), -1)

# 원근 변환 행렬 계산
mtrx = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, mtrx, (cols, rows))

# 결과 출력
cv2.imshow('origini', img)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()