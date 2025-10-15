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
# sources = ["bdd", "cityscapes", "detrac", "kitti", "self_driving2coco"]
sources = ["bdd", "cityscapes", "detrac", "exdark", "kitti"]

dropout_pos = "1_2"
dropou_rate ='0_15'

X = []
y = []
# metesize = 50
metesize = 10



for source in sources:
    for meta in range(metesize):
        with open('./res_1007_wbf/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '_s250_n50/' + str(meta) + '.json') as f:
            with open('./res1/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '_s250_n50/' + str(meta) + '.json') as f1:
                data = json.load(f)
                data1 = json.load(f1)
                X.append(data['0'][0][0]*100)  # mAP
                y.append(data['0'][9][0])
                # y.append(data1['0'][3][0] / np.sqrt(data['0'][3][0]))  # 0.749
                # y.append(-data1['0'][2][0] ** np.sqrt(data['0'][3][0])  / data1['0'][3][0] )
                # y.append(1 / data1['0'][2][0] * data1['0'][3][0] / np.sqrt(data['0'][3][0]))  #0.853
                # y.append(-data1['0'][2][0] * 1 / (data1['0'][3][0] / np.sqrt(data['0'][3][0])))  # 0.885
                # y.append(data['0'][3][0] ** data1['0'][3][0])
                # y.append(-data['0'][2][0] / data['0'][3][0])
            # y.append(data['0'][3][0] )
            # if np.isnan(data['0'][3][0]):
            #     y.append(imputation)
            # else:
            #     imputation = data['0'][3][0]
            #     y.append(data['0'][3][0])  
    
# print(X)
# print(y)
# exit()
rho1, pval1 = stats.spearmanr(X, y)
rho1 = round(rho1, 3)
print('\nRank correlation-rho', rho1)
print('Rank correlation-pval', pval1)

rho2, pval2 = stats.pearsonr(X, y)
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

# ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# f.savefig('./figs2/correlation_mAP_bos_5.pdf')
# f.savefig('./figs2/correlation_mAP_bos_5_under_over.pdf')
f.savefig('./figs_1007_wbf/correlation_mAP_bos2.pdf')