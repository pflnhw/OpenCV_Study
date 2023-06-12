import cv2
import time
import random


cap = cv2.VideoCapture('muyaho.mp4')

# 영상의 가로와 세로 길이가 나중에 필요해서 구해놓음 
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# FPS를 절반으로 줄여서 영상의 속도를 절반으로 느리게했음 
out = cv2.VideoWriter('output_%s.mp4' % time.time(), fourcc, cap.get(cv2.CAP_PROP_FPS) / 2, (w, h))

# 영상을 건너띄우기 위해서 앞을 제거하는 부분 
# cap.set(1, 900) # 무야호 시작

while cap.isOpened():
    # 영상을 한프레임씩 불러오기위한 코드
    # img은 영상을 저장하는 변수, ret은 영상이 끝났을 때 false로 변경됨.
    ret, img = cap.read()
    if not ret:
        break
    
    # 랜덤으로 20%만 뽑아서 랜럼으로 움직이게 하는 코드 
    if random.random() > 0.9:
        # 랜덤으로 영상의 각도를 좌우로 움직여서 애니메이션 효과를 만든 부분 
        theta = random.randint(-3, 3)
        x, y = random.randint(-10, 10), random.randint(-10, 10)

        # 영상을 회전시키는 부분 (https://deep-learning-study.tistory.com/199)
        M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=theta, scale=1.0)
        M[0, 2] += x
        M[1, 2] += y

        # 이미지를 기하학적 변형 
        img = cv2.warpAffine(img, M=M, dsize=(w, h))
    # 뿌옇도록 하는 효과
    img = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=0)

    '''
    sigma_s: Range between 0 to 200. Default 60.
    sigma_r: Range between 0 to 1. Default 0.07.
    shade_factor: Range between 0 to 0.1. Default 0.02.
    '''
    # 연필효과를 주는 함수 
    gray, color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.05, shade_factor=0.015)


    cv2.imshow('gray', gray)
    # cv2.imshow('color', color)

    out.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()