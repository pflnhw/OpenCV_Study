import cv2

video_file = "./01.mp4"

cap = cv2.VideoCapture(video_file)
if cap.isOpened():
    while True:
        # return tuple ( (0,0), [0,0] ) [0] = ret, [1] = img:Mat
        # 한번에 가져와서 두개의 변수로 나누기
        ret,img = cap.read()
        
        # read 변수에 담아서 사용
        # read = cap.read()
        # read 변수를 두개의 변수로 나누기
        # ret,img = read
        # ret = read[0]
        # img = read[1]
        # print("read:", read)

        if not ret:
            break
        cv2.imshow(video_file, img)
        cv2.waitKey(25) #25ms (40fps 가정) (1/40 = 0.025) (1/60 = ?)
else:
    print("can't open video.")
cap.release()
cv2.destroyAllWindows()