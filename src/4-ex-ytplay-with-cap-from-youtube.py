import cv2

# pypi
# pafy, youtube-dl
# cap_from_youtube
# https://www.youtube.com/watch?v=QyBdhdz7XM4

import pafy
from cap_from_youtube import cap_from_youtube


# TODO: Youtube 에서 영상을 읽어오기

url = "https://www.youtube.com/watch?v=QyBdhdz7XM4"
# url = "https://www.youtube.com/watch?v=Yb4saNDmddU"


# cap_from_youtube 를 사용
# '144p','240p','360p','480p','720p','1080p','1440p','2160p','best'
# cap = cap_from_youtube(url, 'best')
cap = cap_from_youtube(url, '360p')


# pafy를 이용 현재 에러
# video = pafy.new(url)

# print("video title : {}".format(video.title))  # 제목
# print("video rating : {}".format(video.rating))  # 평점
# print("video viewcount : {}".format(video.viewcount))  # 조회수
# print("video author : {}".format(video.author))  # 저작권자
# print("video length : {}".format(video.length))  # 길이
# print("video duration : {}".format(video.duration))  # 길이
# print("video likes : {}".format(video.likes)) # 좋아요
# print("video dislikes : {}".format(video.dislikes)) #싫어요

# cap = cv2.VideoCapture(video.url)
# print('cap.isOpened: ', cap.isOpened)

if cap.isOpened():
    while True:
        ret, frame = cap.read()        

        # 해당 프레임이 유효한지 파악 (유효하지 않으면 반복 탈출)
        if not ret:
            break
        
        # 유효한 프레임은 윈도우에 표시
        cv2.imshow("CAP", frame)

        # 프레임간에 딜레이를 준다 (화면에 유지될 시간)
        # 만약 아무키나 입력한다면 영상플레이를 중단하고 종료
        # cv2.waitKey() 는 지정된 시간안에 키 입력이 없다면 -1을 반환
        if cv2.waitKey(1) >= 0:
            break
        pass
    pass
else:
    print('재생할 영상을 찾지 못함')

cap.release()
cv2.destroyAllWindows()
# cv2.destroyWindow("CAP") #특정 이름의 윈도우를 종료