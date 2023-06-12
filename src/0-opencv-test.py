import cv2

# 변수, 함수는 소문자로 시작 ex variable
# 상수는 대문자 띄어쓰기 _로 구분 ex CONST_VALUE
# 클래스는 대문자로 시작 ex Class

imgPath = './res/4star.jpg'

readImg = cv2.imread(imgPath)

if readImg is not None:
    cv2.imshow('img', readImg)
    cv2.waitKey() # block
    cv2.destroyAllWindows()