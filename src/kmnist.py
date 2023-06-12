import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 1 - 이미지를 load하고 사이즈를 다시 맞춘다.
img_path = 'img/09.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# 이미지의 사이즈를 20%로 줄인다.
img = cv2.resize(img, dsize=None, fx=0.2, fy=0.2)
# 이미지의 크기를 확인해본다. 
#print(img.shape)
# [실습] 픽셀화된 이미지를 출력한다. 
# plt.figure(figsize=(20, 20))
# plt.axis('off')
# plt.imshow(img, cmap='gray')
# plt.show()

# 2 - 조각(patch)로 사용할 이미지를 미리한번 확인해보는 부분 
# k49-train-img.npz라는 파일을 불러와서 이미지만 사용할것(arr_0)
sample_imgs = np.load('dataset/k49-train-imgs.npz')['arr_0']

# plt.figure(figsize=(20, 10))
# for i in range(80):
#     img_patch = 255 - sample_imgs[i]

#     plt.subplot(5, 16, i+1)
    # 제목에는 이미지 픽셀의 평균값을 확인한다.
    # 픽셀의 평균값과 픽셀안의 값이 일치하면 넣는다.  
    # 평균값이 낮을 수록 글자에 쓰의진게 많다. 
#     plt.title(int(np.mean(img_patch)))
#     plt.axis('off')
#     plt.imshow(img_patch, cmap='gray')
# plt.show()

# 3- 픽셀의 평균값들이 어떠한 분포를 가지고 있는지 보는 부분
# mnist는 글자가 검정이기 때문에 255에서 뺴기를 해서 흰색으로 만든다.
# mean함수를 사용해서 평균을 낸다. 
means = np.mean(255 - sample_imgs, axis=(1, 2))

# plt.figure(figsize=(12, 6))
# # plt의 히스토그램(분포도)을 이용해서 구역을 50개로 나눈 그림을 로그스케일로 그린다. 
# plt.hist(means, bins=50, log=True)
# plt.show()

# 그림해설
# 글자 픽셀의 평균이 90에서 255까지 존대한다는 것의 의미 
# 그림에는 90보다 작은 것들은 90으로 만들어줘야 칸을 다 채울수 있음. 
# 0~255까지의 모두 이용하지 않고 가장 많은 분포를 하고 있는 120~245까지로 분포를 가지도록 조정함. 
# 위 작업을 해주기 위해 opencv의 normalize 작업을 한다. 
# normalize함수에서 alpah는 120, beta는 245으로 조정을 해라 
img = cv2.normalize(img, dst=None, alpha=120, beta=245, norm_type=cv2.NORM_MINMAX)

# plt.figure(figsize=(12, 6))
# plt.hist(img.flatten(), bins=50, log=True)
# plt.show()

# 4-pach이미지랑 고양이의 픽셀 이미지를 매칭하는 작업 
# patch이미지들을 평균값에 맞춰서 bins에 딕셔너리로 정리한다.
# 예를 들어 patch가 120이미지 = ... /121이미지 =  ... / 255이미지 = ,,,
bins = defaultdict(list)

for img_patch, mean in zip(sample_imgs, means):
    bins[int(mean)].append(img_patch)
    
#print(len(bins))

# 5-이미지를 채우는 과정
h, w = img.shape
# 최종이미지인 img_out에다가 0으로 채워넣는다. patch이지가 28*28사이즈를 갖기 때문에 곱한다. 
img_out = np.zeros((h*28, w*28), dtype=np.uint8)
# for 반복을 돌면서 채운다. 
for y in range(h):
    for x in range(w):
        level = img[y, x]
        # 레벨에 해당되는 이미지를 넣는다. 
        b = bins[level]
        # 레벨에 해당되는 이미지가 없는 경우에 그 다음 레벨의 패치를 가져오게한다.
        while len(b) == 0:
            level += 1
            b = bins[level]
        # b에 있는 패치 이미지들 중에 랜덤으로 뽑아서 img_patch라는 변수에 넣는다.
        # 넣을 때 255을 빼준다. (하얀색 바탕에 검은색을 넣기 원하므로)
        img_patch = 255 - b[np.random.randint(len(b))]
        # 이미지 패치를 해당 픽셀에 넣어주는 코드
        img_out[y*28:(y+1)*28, x*28:(x+1)*28] = img_patch
        
# plt.figure(figsize=(20, 20))
# plt.axis('off')
# plt.imshow(img_out, cmap='gray')
# plt.show()

_ = cv2.imwrite('result/%s_bw.jpg' % os.path.splitext(os.path.basename(img_path))[0], img_out)