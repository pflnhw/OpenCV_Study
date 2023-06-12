import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#KMean 클러스터링이란 비슷한 K개의 군집으로 분리해주는 머신러닝
from sklearn.cluster import KMeans
# pairwise ..는 가까운 색깔을 비교할 때 사용 
from sklearn.metrics import pairwise_distances_argmin_min
# shuffle은 랜덤으로 섞는다는 의미
from sklearn.utils import shuffle

# sklearn이 없다고 나오면 설치를 해준다.
# pip install -U scikit-learn

img_path = 'img/01.jpg'
# 이미지를 컬러로 읽는다
img = cv2.imread(img_path)

img = cv2.resize(img, dsize=None, fx=0.2, fy=0.2)
# BGR을 RGB로 변경한다. 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#print(img.shape)

# plt.figure(figsize=(20, 20))
# plt.axis('off')
# plt.imshow(img)
# plt.show()

# 비슷한 원리로 픽셀에 이미지를 넣어준다. 
# 공식 홈페이지를 참고하여 이미지를 넣어준다. 
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        
    data = dict[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float64) / 255.

    return data

x_train_1 = unpickle('dataset/data_batch_1')
x_train_2 = unpickle('dataset/data_batch_2')
x_train_3 = unpickle('dataset/data_batch_3')
x_train_4 = unpickle('dataset/data_batch_4')
x_train_5 = unpickle('dataset/data_batch_5')

# 로드한 다음 sample_imgs에 넣어준다. 
sample_imgs = np.concatenate([x_train_1, x_train_2, x_train_3, x_train_4, x_train_5], axis=0)

# print(sample_imgs.shape)

# plt.figure(figsize=(20, 10))
# for i in range(80):
#     img_patch = sample_imgs[i]

#     plt.subplot(5, 16, i+1)
#     plt.axis('off')
#     plt.imshow(img_patch)
# plt.show()

# kmins클러스터를 쓰는 이유
# 흑백이미지는 0~255까지만 되므로 고려해야될 점이 많지 않다. 
# 심지어 채널이 한개일 때도 0부터 120은 못쓰므로 120~245로 평균화했다.
# 컬러는 경우의 수가 3배이다. 이미지를 아무리 찾아도 없을 수가 있다.
# 완벽히 일치하는 패치를 찾기 힘드니까, 어느정도 가까운 거리에 있는 패치를 찾는다. 
# cluster를 써서 경우의 수를 줄이고 32개의 색상만가지는 이미지를 만든다.
N_CLUSTERS = 32

h, w, d = img.shape
# 이미지를 복사해서 float형태로 만든다음에 255로 나눠준다. 
img_array = img.copy().astype(np.float64) / 255.
# reshape형태로 펴준다.
img_array = np.reshape(img_array, (w * h, d))

# all pixels
# suffle은 옵션인데 이미지 크기가 클 경우에는 전부다 클러스터링 할수 없다. 

img_array_sample = shuffle(img_array, random_state=0)

# 1000개 이미지만 랜덤으로 뽑아서 클러스터링을 한다.(이미지가 클경우) 
# pick random 1000 pixels if want to run faster
# img_array_sample = shuffle(img_array, random_state=0)[:1000]

# KMeans clustering
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(img_array_sample)
# 클러스터 군집화를 확인. 32개로 quartize한 결과 
# print(kmeans.cluster_centers_)

# quantize한 것을 이미지에 찍어본다. 
cluster_centers = kmeans.cluster_centers_

pred_labels = kmeans.predict(img_array)
cluster_labels = pred_labels.reshape((h, w))

img_quantized = np.zeros((h, w, d), dtype=np.float64)

label_idx = 0
for y in range(h):
    for x in range(w):
        label = pred_labels[label_idx]

        img_quantized[y, x] = cluster_centers[label]

        label_idx += 1

# plt.figure(figsize=(20, 20))
# plt.axis('off')
# plt.imshow(img_quantized)
# # 결과를 확인해보면 32개의 색상으로 강제로 quantize한 결과를 확인할수 있다.
# # 옛날 컴퓨터 게임화면 
# plt.show()

# 패치 이미지들 사이에 가장 유사한 것을 골라낸다. 
DISTANCE_THRESHOLD = 0.1

bins = defaultdict(list)

# sample_imgs들을 5만번 돌리면서 패치이미지의 평균값을 mean에 넣어준다.
for img_patch in sample_imgs:
    mean = np.mean(img_patch, axis=(0, 1))

    # compare patch mean and cluster centers
    # a와 b의 데이터를 비교하면서 가장 가까운 데이터의 인덱스와 그 값을 구한다. 
    cluster_idx, distance = pairwise_distances_argmin_min(cluster_centers, [mean], axis=0)
    # distance가 허용하는 범위 이하일 때만 적용한다.  
    # 만약 이미지가 안나오면 DISTANCE_THRESHOLD를 높혀야한다.
    if distance < DISTANCE_THRESHOLD:
        bins[cluster_idx[0]].append(img_patch)

# number of bins must equal to N_CLUSTERS. if not, increase DISTANCE_THRESHOLD
assert(len(bins) == N_CLUSTERS)

# 이미지를 채우기
# cifar10은 32*32의 크기를 가지고 있다. 
img_out = np.zeros((h*32, w*32, d), dtype=np.float64)

for y in range(h):
    for x in range(w):
        label = cluster_labels[y, x]

        b = bins[label]

        img_patch = b[np.random.randint(len(b))]

        img_out[y*32:(y+1)*32, x*32:(x+1)*32] = img_patch
        
# plt.figure(figsize=(20, 20))
# plt.axis('off')
# plt.imshow(img_out)
# plt.show()

img_out2 = cv2.cvtColor((img_out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
_ = cv2.imwrite('result/%s_color.jpg' % os.path.splitext(os.path.basename(img_path))[0], img_out2)