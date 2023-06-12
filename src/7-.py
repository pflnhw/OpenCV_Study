# 요구사항
# 유튜브에서 특정 영상을 다운받아 기록하는 것을 기반
# 1. 유튜브영상을 스케치로 변경하여 저장
# 2. 유튜브영상내에 특정컬러 추적하여 필터 후 저장
#   2-1. 특정컬러의 영역에 사각테두리를 둘러서 표시
#   2-2. 특정컬러의 영역만 마스킹하여 해당 컬러의 이미지만 색상이 있도록 (배경은 흑백)

# 사용기술
# pafy or cap_from_youtube
# opencv
# hsv 컬러맵에 대한 이해 (yuv)
# 스케치 함수 사용에 대한 이해 (이전 코드 참고)

import cv2
import pafy
import time
import random
import numpy as np

url = "https://www.youtube.com/watch?v=QyBdhdz7XM4"

# 해당 url의 영상 다운로드
videoInfo = pafy.new(url)

best = videoInfo.getbest(preftype='mp4')

# 캡쳐 구성
videoPath = best.url
cap = cv2.VideoCapture(videoPath)
frameRate = cap.get(cv2.CAP_PROP_FPS)


# [중요] k-nearest neighbors 알고리즘 사용하여 배경 영상의 변화를 알아내는 알고리즘 
# 과거 프레임과 최근프레임의 감지해가지고 history동안 얼마나 변했는지를 빼기 연산을 통해서 알려주는 알고리즘 
sub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=100, detectShadows=False)

if cap.isOpened():

    
    # 녹화 정의
    saveFilePath = './record1.avi'
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # cap 내의 있는 영상에 대한 정의 중 크기를 가져옴
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = int(width), int(height)

    # VideoWriter 객체 생성 (어떻게 파일을 저장할건지에 대한 정의)
    out = cv2.VideoWriter(saveFilePath, fourcc, fps, size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 빨간색 범위1
        lowerRed = np.array([0, 120, 70])
        upperRed = np.array([10, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lowerRed, upperRed)

        # 빨간색 범위2
        lowerRed = np.array([130, 120, 70])
        upperRed = np.array([180, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, lowerRed, upperRed)
        
        mask = mask1 + mask2

        # 흑백 효과
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray.resize(width, height)
        mask.resize(width, height)

        result = cv2.bitwise_or(gray, frame, mask=mask)

        # # 뿌옇도록 하는 효과
        # frame2 = cv2.GaussianBlur(frame, ksize=(9,9), sigmaX=0)

        # # 스케치로 변경
        # sketch , color = cv2.pencilSketch(frame2, sigma_s=60, sigma_r=0.05, shade_factor=0.015)

        # sketch = cv2.resize(sketch, dsize=size)

        # # 웹 캠 사이의 차이를 알아내서 mask에 저장한다. 
        # mask = sub.apply(frame)

        # # 5x5 타원형태로 정의해주고 
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # # morphology open은 이미지 바깥의 노이즈를 없애준다
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # # morphology closing은 이미지 안쪽의 노이즈를 없애준다
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # # 원본 이미지를 굵어지해 만들어주는 효과
        # mask = cv2.dilate(mask, kernel, iterations=2)

        # bitwise_and연산을 해주게 되면 합치게 된다. 마스크 형태로
        # result = cv2.bitwise_and(sketch, frame, mask=mask)

        # 캡쳐된 프레임을 표시
        # cv2.imshow('video', frame)
        # cv2.imshow('video', sketch)
        cv2.imshow('video', result)
        # cv2.imshow('video', gray)

        # 캡쳐된 프레임을 기록
        # out.write(frame)
        # out.write(sketch)
        out.write(result)
        # out.write(gray)

        if cv2.waitKey(int(1000/fps)) >=0:
            break


    # 저장하는 버퍼를 닫아줌
    out.release()
    pass

cap.release()
cv2.destroyAllWindows()
