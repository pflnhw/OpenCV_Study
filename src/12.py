# 열화상 카메라로 찍은 이미지를 구역별로 나눠 각 구역의 평균 온도를 파악
# 1. 가로를 10칸으로 나누고 하나의 칸은 정사각형
# 2. 격자로 선을 그어서 공간을 나누기
# 3. 셀을 관리할 수 있는 맵핑 (row, col)좌표로 특정 셀에 접근
# 4. 각 셀의 화소들 색상을 종합해서 평균값을 구함. 해당 평균값은 특정 기준으로 온도의 레벨 정의(0~9)
# 5. 온도레벨을 해당 셀에 문자로 표시

import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread('./res/infrared_road.jpg')
# 가로, 세로 값 정의
rows, cols = img.shape[:2]

dst1 = img.copy()
dx = cols//7
dy = rows//5
# dst1 = cv2.cvtColor(dst1, cv2.COLOR_BGR2RGB)

# 폰트 색상 지정

blue = (255, 0, 0)
green= (0, 255, 0)
red= (0, 0, 255)
white= (255, 255, 255) 

# 폰트 지정
font =  cv2.FONT_HERSHEY_PLAIN

for i in range(5):
    # 가로 선
    cv2.line(dst1, (0, dy*i), (cols, dy*i), (255, 255, 255), thickness=1)
    for j in range(7):
        # 세로 선
        cv2.line(dst1, (dx*j, 0), (dx*j, rows), (255, 255, 255), thickness=1)
        # 관심영역 지정
        roi = dst1[dy*i: dy*(i+1), dx*j: dx*(j+1)]
        # 컬러 평균값
        roiHsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 컬러 범위
        lowerRed = np.array([-50, 0, 100])
        upperRed = np.array([130, 255, 255])
        mask = cv2.inRange(roiHsv, lowerRed, upperRed)

        dst = cv2.copyTo(img, mask = mask)

        data = cv2.mean(mask)
        print(data)
        # 레벨 구하기
        # for k in range(10):
        #     if data[2]>=k*25 and data[2]<=(k+1)*25:
        #         revel = k
        # # 관심영역 가로 세로 값
        # rows2, cols2 = roi.shape[:2]
        # cv2.putText(roi, str(revel), (rows2//2, cols2//2), font, 2, green, 1, cv2.LINE_AA)
        
cv2.imshow('origin', dst1)
cv2.waitKey()
cv2.destroyAllWindows()