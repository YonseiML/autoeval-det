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
from matplotlib.lines import Line2D
import torch
import matplotlib.patheffects as pe

sources = ["coco", "bdd", "cityscapes", "detrac", "exdark", "kitti", "self_driving2coco", "roboflow2coco", "udacity2coco", "traffic2coco"] # car
# sources = ["coco", "caltech", "citypersons", "cityscapes", "crowdhuman", "ECP", "ExDark", "kitti", "self_driving"] # person

dropout_pos = "1_2"
dropou_rate ='0_15'

X = []
y = []

datasets = []
meta_ids = []  

metesize = 50
for source in sources:
    for meta in range(metesize):
        with open('./result/car_inc/PCR/r50_retina/' + str(source) + '_s250_n50/' + str(meta) + '.json') as f:
            data = json.load(f)
            X.append(data['0'][0][0] * 100)  # mAP  
            # y.append(1-data['0'][1][0]) # consistency
            y.append(data['0'][2][0]) # reliability

            datasets.append(source)    
            meta_ids.append(meta) 
            

Xx = []
yy = []
metesize2 = 1
for source in sources:
    for meta in range(metesize2):
        with open('./result/car_inc/PCR/r50_retina/swin_retina/cost_droprate_' + dropou_rate + '_' + str(source) + '_droppos_' + dropout_pos + '_s250_n50/' + str(meta) + '.json') as f:
                        data = json.load(f)
                        
                        Xx.append(data['0'][0][0]*100)
                        # yy.append(1-data['0'][1][0]) # consistency
                        yy.append(data['0'][2][0]) # reliability
                        
# Correlation
rho1, pval1 = stats.spearmanr(X, y)
rho2, pval2 = stats.pearsonr(X, y)

print(rho2)
print(rho1)

# Color mapping
palette = sns.color_palette("bright", len(sources))
color_mapping = {source: palette[i] for i, source in enumerate(sources)}

sns.set(font_scale=1.3)
sns.set_style('darkgrid', {'axes.facecolor': '0.96', 'axes.linewidth': 20, 'axes.edgecolor': '0.15'})

fig, ax = plt.subplots(figsize=(10, 8))

X_arr = np.array(X)
y_arr = np.array(y)
datasets_arr = np.array(datasets)
meta_ids_arr = np.array(meta_ids)

i = 0

# Dataset-wise points
for source in sources:
    mask = np.array(datasets) == source
    ax.scatter(np.array(y)[mask], np.array(X)[mask], label=source, color=color_mapping[source], alpha=1, s=70)


i = 0
for source in sources:
    ax.scatter(yy[i], Xx[i], color='black', facecolors=color_mapping[source], linewidths=2.5, alpha=1, s=500, marker='*')
    i += 1

# Regression Line
m, b = np.polyfit(y, X, 1)

line = ax.axline((0, b), slope=m,
                 color='gray', linestyle='-', linewidth=1.5, zorder=10,
                 solid_joinstyle='round', solid_capstyle='round')
line.set_path_effects([
            pe.Stroke(linewidth=9, foreground='black', alpha=0.95,
                    joinstyle='round', capstyle='round'),
            pe.Normal()
        ])

# Legend
line_leg = Line2D([0],[0], color='gray', lw=1.5, linestyle='-',
                      solid_joinstyle='round', solid_capstyle='round',
                      label=f' $r$={rho2:.3f}, \n $\\rho$={rho1:.3f}')
line_leg.set_path_effects([
        pe.Stroke(linewidth=9, foreground='black', alpha=0.95,
                joinstyle='round', capstyle='round'),
        pe.Normal()
    ])
stats_legend= [line_leg]
stats_legend_obj = ax.legend(handles=stats_legend, loc=2, shadow=True, labelspacing=-0.0, handletextpad=0, borderpad=0.5, markerscale=2, prop={'weight': 'medium', 'size': '35'})

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='both', labelsize=20)

ax.set_ylim(-5, 47) # vehicle
# ax.set_ylim(-4, 40) # pedestrian
lo = float(np.min(y))
hi = float(np.max(y))
pad = 0.05
ax.set_xlim(lo-pad, hi+pad)


# 레이아웃 조정
plt.tight_layout()
save_path = './corr_plots/car_r50_retina_rel.pdf'

# 디렉토리 자동 생성
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 그림 저장
plt.savefig(save_path, bbox_inches='tight')

