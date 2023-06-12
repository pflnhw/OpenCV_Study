import cv2
import numpy as np
from cap_from_youtube import  cap_from_youtube

def toPencilSketch(origin:cv2.Mat) -> cv2.Mat:
    out = origin.copy()
    # 뿌옇도록 하는 효과
    out = cv2.GaussianBlur(frame, ksize=(9,9), sigmaX=0)

    # 스케치로 변경
    out , color = cv2.pencilSketch(out, sigma_s=60, sigma_r=0.05, shade_factor=0.015)

    return out

def trackingColor(origin:cv2.Mat, ret) -> cv2.Mat:

    frame = origin.copy()

    if ret==1:
        # 특정색을 추적해서 배경은 흑백, 특정색만 컬러로 합성
        bgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 빨간색 범위1
        lowerRed = np.array([0, 120, 70])
        upperRed = np.array([10, 255, 255])
        mask1 = cv2.inRange(frameHSV, lowerRed, upperRed)

        # 빨간색 범위2
        lowerRed = np.array([130, 120, 70])
        upperRed = np.array([180, 255, 255])
        mask2 = cv2.inRange(frameHSV, lowerRed, upperRed)
        
        # cv2.add() => 화소의 값이 255를 넘으면 최대 255로 유지
        # + 연산자 => 화소의 값이 255를 넘으면 초기화 ex: 255 + 1 = 0
        mask = mask1 + mask2

        # bitwise_and() 의 src1과 src2의 색상 채널 수는 같아야한다.
        redFrame = cv2.bitwise_and(frame, frame, mask = mask)
        maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        redFrame = cv2.bitwise_and(frame, maskBGR)

        bgGray = cv2.cvtColor(bgGray, cv2.COLOR_GRAY2BGR)
        out = cv2.bitwise_or(bgGray, redFrame)
    else:
        # 특정색을 추적해서 특정색만 컬러로 합성
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 빨간색 범위1
        lowerRed = np.array([0, 120, 70])
        upperRed = np.array([10, 255, 255])
        mask1 = cv2.inRange(frameHSV, lowerRed, upperRed)

        # 빨간색 범위2
        lowerRed = np.array([130, 120, 70])
        upperRed = np.array([180, 255, 255])
        mask2 = cv2.inRange(frameHSV, lowerRed, upperRed)
        
        # cv2.add() => 화소의 값이 255를 넘으면 최대 255로 유지
        # + 연산자 => 화소의 값이 255를 넘으면 초기화 ex: 255 + 1 = 0
        mask = mask1 + mask2

        # bitwise_and() 의 src1과 src2의 색상 채널 수는 같아야한다.
        redFrame = cv2.bitwise_and(frame, frame, mask = mask)

        sketch = toPencilSketch(frame)

        sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        out = cv2.bitwise_or(sketch, redFrame)
    return out

# 1. 유튜브영상을 스케치로 변경하여 저장

url = "https://www.youtube.com/watch?v=QyBdhdz7XM4"

# 캡쳐 구성

cap = cap_from_youtube(url, '480p')

fps = cap.get(cv2.CAP_PROP_FPS)

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 펜슬 스케치로 변경
        # out = toPencilSketch(frame)
        out = trackingColor(frame, 2)

        cv2.imshow('video', out)
        if cv2.waitKey(int(1000/fps)) >= 0:
            break

        pass

cap.release()