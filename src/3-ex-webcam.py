import cv2

# 첫번째 캠을 연결 (index:0)
cap = cv2.VideoCapture(0)

#캡쳐가 연결되었는지 확인
if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow('camera', img)

            # 아무키도 입력되지 않은 반환값 -1
            if cv2.waitKey(1) != -1:
                break
        else:
            print('no frame')
            break
else:
    print("can't open camera.")
cap.release()
cv2.destroyAllWindows()