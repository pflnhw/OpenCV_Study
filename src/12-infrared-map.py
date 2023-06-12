#열화상 카메라로 찍은 이미지를 구역별로 나눠 각 구역의 평균 온도를 파악!
#요구사항
#1.7x5칸으로 나누기
#2.격자로 선을 그어서 공간을 나누기
#3.셀을 관리할 수 있는 맵핑 (row,col)좌표로 접근해서 특정 셀에 접근
#4.각 셀의 화소들 색상을 종합해서 평균값을 구함. 해당 평균값은 특정 기준으로 온도의 레벨(0-9)
#5.255 -> lv9 => 255/10
#6.온도레벨을 해당 셀에 문자로 표시

import cv2
import numpy as np


def createROI(src, cSize):

    cellH, cellW = cSize

    cH = cellH
    if y+cellH > rows:
        cH = rows % cellH

    cW = cellW
    if x+cellW > cols:
        cW = cols % cellW

    roi = np.zeros((cH, cW, 3), np.uint8)
    # print('roi shape:', roi.shape[0:2])

    # min(a1,a2) 두 값중에 작은값을 반환, 최대값 제한에 사용, x1 = x+100, x2 = 50 min(x1, x2) 
    # max(a1,a2) 두 값중에 큰값을 반환, 최소값 제한에 사용

    bTotal = 0
    for by in range(y, maxRows):
        for bx in range(x, maxCols):
            # print(f'({by},{bx})')
            colors = src[by,bx]

            # 255 + 255 + 255
            # bTotal += ((colors[0] + colors[1] + colors[2]) / 3)
            roi[by-y, bx-x] = colors

    # mean = (bTotal[0] + bTotal[1] + bTotal[2]) / 3
    # mean = bTotal/((cH*cW)*3)

    # cv2.imshow(coordinate, roi)



src = cv2.imread('./res/infrared_road.jpg')
rows, cols = src.shape[0:2] #0,1

src = cv2.resize(src, (cols*2, rows*2)) #2배로 키우기
rows, cols = src.shape[0:2] #0,1

cellRows = 5
cellCols = 7

cellH = rows // cellRows
cellW = cols // cellCols

print('rows: ', rows)
print('cols: ', cols)
print('cellH: ', cellH)
print('cellW: ', cellW)

# 4x3  [y][x]
# 0 0 0 0
# 0 0 0 0
# 0 0 0 0

hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
hsvMask = cv2.inRange(hsv, (-50,0,100), (130,255,255))
masked = cv2.copyTo(src, mask = hsvMask)

heatMap = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

dst = src.copy()

# BGR B:100,G:50,R:50 == B:50,G:50,R:100

for y in range(0, rows, cellH):
    cv2.line(dst, (0, y), (cols, y), color=(255,255,255), thickness=1)
    for x in range(0, cols, cellW):

        pointText = f'({y},{x})'
        # print(pointText)
        cv2.line(dst, (x, 0), (x, rows), color=(255,255,255), thickness=1)

        coordinate = f'({y//cellH},{x//cellW})'
        cv2.putText(dst, coordinate, (x+10,y+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 1)

        maxRows = min(y+cellH, rows)
        maxCols = min(x+cellW, cols)

        roi = heatMap[y:maxRows, x:maxCols]
        mean = np.mean(roi)
        # print('mean:', mean)

        # dst에 평균값을 문자로 표시
        # cv2.putText(dst, f'{int(mean)}', (x+10,y+50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 1)
        cv2.putText(dst, str(int(mean)), (x+10,y+50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 1)


cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
