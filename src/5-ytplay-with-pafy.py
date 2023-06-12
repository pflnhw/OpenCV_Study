import cv2
import pafy

# pip install pafy
# pip install youtube-dl

# pafy 오류 수정 참조:
# https://syerco0.com/37

url = "https://www.youtube.com/watch?v=QyBdhdz7XM4"

video = pafy.new(url)

best = video.getbest(preftype="mp4")

# 버퍼 열림
cap = cv2.VideoCapture(best.url)

# 파일이 정상적으로 열렸는지 확인
if cap.isOpened():
    while True:
        ret, frame = cap.read()

        if not ret:
            break
    
        cv2.imshow(best.url, frame)
        if cv2.waitKey(25) >= 0:
            break

# 버퍼 닫음
cap.release()
cv2.destroyAllWindows()