import cv2

img_file = "./img1.jpeg"
img = cv2.imread(img_file)

# is는 타입비교
if img is not None:
    cv2.imshow('IMG', img)

    # 아무런 값을 넘기지 않으면 무한대기
    cv2.waitKey()
    # 5000 이후 자동으로 빠져나감
    # cv2.waitKey(5000)

    # 모든 윈도우를 제거
    cv2.destroyAllWindows()
    pass
else:
    print("No image file.")