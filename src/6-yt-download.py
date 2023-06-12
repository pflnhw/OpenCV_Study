import cv2
import pafy

url = "https://www.youtube.com/watch?v=QyBdhdz7XM4"

# 해당 url의 영상 다운로드
videoInfo = pafy.new(url)

best = videoInfo.getbest(preftype='mp4')

# 캡쳐 구성
videoPath = best.url
cap = cv2.VideoCapture(videoPath)
frameRate = cap.get(cv2.CAP_PROP_FPS)

if cap.isOpened():

    # 녹화 정의
    saveFilePath = './record.avi'
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
        # 캡쳐된 프레임을 표시
        cv2.imshow('video', frame)

        # 캡쳐된 프레임을 기록
        out.write(frame)

        if cv2.waitKey(int(1000/fps)) >=0:
            break

    # 저장하는 버퍼를 닫아줌
    out.release()
    pass

cap.release()
cv2.destroyAllWindows()
