import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr
import json
import numpy as np
from scipy import stats
import os

from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error

sources = ["coco", "bdd", "cityscapes", "detrac", "exdark", "kitti", "self_driving2coco", "roboflow2coco", "udacity2coco", "traffic2coco"]
# sources = ["coco", "cityscapes", "detrac", "exdark", "kitti", "self_driving2coco", "roboflow2coco", "udacity2coco", "traffic2coco"]
# sources = ["bdd", "coco", "detrac", "exdark", "kitti", "self_driving2coco", "roboflow2coco", "udacity2coco", "traffic2coco"]
# sources = ["bdd", "cityscapes", "detrac", "kitti", "self_driving2coco"]
# sources = ["bdd", "cityscapes", "detrac",  "exdark", "kitti", "roboflow2coco"]

index_to_exclude = 0  # 3번째 요소(값: 3)를 제외

train_set = sources[:index_to_exclude] + sources[index_to_exclude+1:]
text_set = [sources[index_to_exclude]]
print(train_set)
print(text_set)
# exit()
dropout_pos = "1_2"
dropou_rate ='0_15'

X = []
y1 = []
y2 = []
metesize = 50

for source in train_set:
    for meta in range(metesize):
        with open('./02.21/r50_retina/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '_s250_n50/' + str(meta) + '.json') as f:
            data = json.load(f)
            X.append(data['0'][0][0]*100)  # mAP
            # y.append(-data['0'][2][0])
            # y.append(-data['0'][2][0]/data['0'][9][0])
            # y.append(data['0'][8][0]/data['0'][3][0])
            y1.append(data['0'][3][0])
            y2.append(data['0'][8][0])


metesize = 1
esti = []
true = []
esti1 = []
esti2 = []
for source in text_set:
    for meta in range(metesize):
        with open('./02.21/r50_retina_ori_dataset/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '_s250_n50/' + str(meta) + '.json') as f:
            data = json.load(f)

            true.append(data['0'][0][0]*100)
            # esti.append(-data['0'][2][0])
            # esti.append(-data['0'][2][0]/data['0'][9][0])
            # esti.append(data['0'][8][0]/data['0'][3][0])
            esti1.append(data['0'][3][0])
            esti2.append(data['0'][8][0])

# mAP와 BS 값을 가정한 데이터 생성 (예시 데이터)
# BS = np.array([0.1, 0.4, 0.5, 0.7, 0.9])  # Box Stability 값 (특징)
# mAP = np.array([0.15, 0.35, 0.55, 0.65, 0.85])  # mAP 값 (목표값)
y1 = np.array(y1).reshape(-1, 1)
y2 = np.array(y2).reshape(-1, 1)
BS = np.hstack((y1, y2))
# print(BS.shape)
# exit()
# BS  = np.array(y)
mAP = np.array(X)
# mAP = mAP.reshape(-1, 1)
# print(mAP.shape)
# exit()
# print(len(BS))
# print(len(mAP))
# 2D array로 변환 (sklearn은 2D 배열을 입력으로 받음)
# BS = BS.reshape(-1, 1)

# 선형 회귀 모델 초기화 및 학습
model = LinearRegression()
model.fit(BS, mAP)
# exit()
# 모델의 기울기 (ω1)와 절편 (ω0) 확인
omega1 = model.coef_[0]
omega0 = model.intercept_

print(f"기울기 (ω1): {omega1}")
print(f"절편 (ω0): {omega0}")

# esti = np.array(esti)
esti1 = np.array(esti1).reshape(-1, 1)
esti2 = np.array(esti2).reshape(-1, 1)
esti = np.hstack((esti1, esti2))
true = np.array(true)

# esti = esti.reshape(-1, 1)
# 예측값 계산 (학습된 모델을 사용하여 mAP 예측)
mAP_pred = model.predict(esti)

# Root Mean Squared Error (RMSE) 계산
rmse = np.sqrt(np.mean((true - mAP_pred) ** 2))
RMSE = mean_squared_error(true, mAP_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Root Mean Squared Error (RMSE): {RMSE}")
exit()

# 결과 시각화
plt.scatter(BS, mAP, color='blue', label='실제 값')
plt.plot(BS, mAP_pred, color='red', label='예측 값')
plt.xlabel('Box Stability (BS)')
plt.ylabel('mAP')
plt.title('Box Stability와 mAP 간의 선형 회귀')
plt.legend()
plt.show()
