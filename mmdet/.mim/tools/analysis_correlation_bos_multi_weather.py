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

# sources = ["coco", "bdd", "cityscapes", "detrac", "exdark", "kitti", "self_driving2coco", "roboflow2coco", "udacity2coco", "traffic2coco"]
sources = ["bdd", "cityscapes", "detrac", "exdark", "kitti", "self_driving2coco", "roboflow2coco", "udacity2coco", "traffic2coco"]
# sources = ["bdd", "cityscapes", "detrac", "kitti", "self_driving2coco"]
# sources = ["bdd", "cityscapes", "detrac",  "exdark", "kitti"]

experts = ["night", "overcast", "snow", "wet"]
sources = ["night", "overcast", "snow", "wet"]
dropout_pos = "1_2"
dropou_rate ='0_15'

X = []
y = []
# metesize = 50
# metesize = 10
# metesize = 1
# metesize = 4
metesize = 38

# 1007_modi_whole
# 1014_consistency_3
# 1014_consistency_whole
# 017_new
# for source in sources:
#     for meta in range(metesize):
#         with open('./res_0909_modi/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '_s250_n50/' + str(meta) + '.json') as f:
#             data = json.load(f)
#             with open('./res_1007_modi_whole/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '_s250_n50/' + str(meta) + '.json') as f:
#                 data2 = json.load(f)
#                 y.append(data['0'][3][0]/data['0'][4][0]/data2['0'][9][0])
#                 # y.append(-data['0'][2][0]/data['0'][10][0]*data2['0'][9][0])
#                 # y.append(data2['0'][9][0])
#             X.append(data['0'][0][0]*100)  # mAP
            # y.append(data['0'][12][0]*data['0'][2][0])
            # y.append(-data['0'][2][0])
            # y.append(data['0'][3][0])   # 이거 좋음 res_0930
            # y.append(data['0'][5][0]) # 이거 좋음 res_1007 
            # y.append(data['0'][5][0])  # 이거 좋음 res_1007_2
            # y.append(data['0'][2][0] / data['0'][4][0])
            # y.append(-data['0'][2][0]/data['0'][9][0])
                
            # if np.isnan(data['0'][3][0]):
                # print(source)
                # print(meta)

            # y.append(data['0'][5][0]/data['0'][6][0])  # 이거 좋음 res_0909_modi
            # y.append(data['0'][3][0]/data['0'][4][0])    # 이거 좋음  res_0909_modi
            # y.append(data['0'][4][0]/data['0'][5][0])   # 이거 좋음 modi전 res_0909
# split_intersection
# split_resize_ref_iou
# res_split_only_intersection
# res_1017_new_all_count_pred_2
# res_split_ablation
# res_1017_new_ablation
# resize_up_down

# experts = ["original"]
experts = ["snow"]
for expert in experts:
    for source in sources:
        for meta in range(metesize):
            with open('./res_multiweather_train_60_model/' + str(expert) + '_expert/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '/' + str(meta+1) + '.json') as f:
                data = json.load(f)        
                X.append(data['0'][0][0]*100) # mAP
                y.append(-data['0'][2][0])
                y.append(-data['0'][8][0])  # y.append(-data['0'][3][0]) : original
                # y.append(data['0'][8][0]/data['0'][2][0])
            # y.append(data['0'][8][0]/data['0'][5][0])
            # y.append(data['0'][8][0])  # res_resize0.7/0.8/0.9는 idx=9
# import torch
# X = torch.tensor(X).reshape(4,4)
# y = torch.tensor(y).reshape(4,4)
# print(X)
# print(y)
# exit()
# sources = ["roboflow2coco"]
# for source in sources:
#     for meta in range(metesize):
#         with open('./res_ffinal/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '_s250_n50/' + str(meta) + '.json') as f:
#             data = json.load(f)        
#             X.append(data['0'][0][0]*100) # mAP
#             # y.append(-data['0'][2][0])
#             # y.append(-data['0'][2][0]/data['0'][6][0])
#             # y.append(data['0'][8][0]/data['0'][5][0])
#             y.append(data['0'][10][0])  # res_resize0.7/0.8/0.9는 idx=9
    
# print(X)
# print(y)
# exit()
# y1 =  [179.2902, 267.9445, 388.7759, 255.9754, 425.3849, 216.2749, 101.7016, 110.8845, 107.4367, 157.6673 ]
# y2 = [121.8111, 127.5396, 124.7858, 112.8348, 317.5028, 35.6759, 40.7502, 90.9525, 70.1252, 88.5168]
# y = [(a + b)/2 for a, b in zip(y1, y2)]
# y = [8.6352, 11.1450, 26.6879, 10.9381, 12.1041, 7.8440, 8.8596, 8.6621, 16.7895, 8.8628]
# y = [ 11.5696, 28.8531, 11.7571, 12.1853, 8.1074, 9.3029, 8.8785, 17.7258, 9.0439]
# y = [2.4299, 5.2535, 1.6002, 1.1264, 1.7532, 3.4615, 1.4729, 5.7016, 1.3847]
# print(y)
rho1, pval1 = stats.spearmanr(X, y)
rho1 = round(rho1, 3)
print('\nRank correlation-rho', rho1)
print('Rank correlation-pval', pval1)

rho2, pval2 = stats.pearsonr(X, y)  # R^2
rho2 = round(rho2, 3)
print('\nPearsons correlation-rho', rho2)
print('Pearsons correlation-pval', pval2)

# palette = sns.color_palette("Paired")
############################################
# palette = sns.color_palette("Paired")
palette = sns.color_palette("deep")
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

robust = True
sns.set()
sns.set(font_scale=1.3)
sns.set_style('darkgrid', {'axes.facecolor': '0.96', 'axes.linewidth': 20, 'axes.edgecolor': '0.15'})

f, ax1 = plt.subplots(1, 1, tight_layout=True)
for i in range(len(sources)):
    idx = i * metesize
    # sns.regplot(ax=ax1, color=palette[i+1], y=X[idx:idx+metesize], x=y[idx:idx+metesize],  scatter_kws={'alpha': 0.5, 's': 30}, label=source)
    sns.scatterplot(ax=ax1, color=palette[i], y=X[idx:idx+metesize], x=y[idx:idx+metesize], alpha=0.5, s=30)
sns.regplot(ax=ax1, color='blue', y=X, x=y, robust=robust, scatter=False, label='{:>8}\n{:>8}'.format(
    r'$R^2$' + '={:.3f}'.format(rho2), r'$ρ$' + '={:.3f}'.format(rho1)))
ax1.legend(loc=2, shadow=True, labelspacing=-0.0, handletextpad=0, borderpad=0.5, markerscale=2,
    prop={'weight': 'medium', 'size': '8'})



# plt.xlabel("Box Stability Score", fontsize=17)
plt.xlabel("Score", fontsize=17)
# plt.ylabel("Detrac (target set) mAP (%)", fontsize=17)
plt.ylabel("mAP (%)", fontsize=17)
# plt.ylabel("Kitti (target set) mAP (%)", fontsize=17)

# ax1.tick_params(axis='x', labelsize=10)  # x축 글자 크기
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# f.savefig('./figs/correlation_mAP_bos_5.pdf')
f.savefig('./figs_1014/correlation_mAP_bos.pdf')
# f.savefig('./figs_1007_modi/correlation_mAP_bos3.pdf')