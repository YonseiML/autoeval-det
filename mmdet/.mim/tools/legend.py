import matplotlib.pyplot as plt
import seaborn as sns

# sources 목록
sources = ["bdd", "cityscapes", "detrac", "exdark", "kitti", 
           "self_driving", "roboflow", "udacity", "traffic"]
# sources = ["night", "overcast", "snow", "wet"]
# Seaborn 색상 팔레트
palette = sns.color_palette("deep")

# 빈 figure 생성
fig, ax = plt.subplots()

# sources에 대한 scatterplot 그리기 (실제로는 보이지 않게 설정)
for i, source in enumerate(sources):
    ax.scatter([], [], color=palette[i], label=source)

# 범례 추가 (legend)
legend = ax.legend(loc='center', shadow=True, labelspacing=0.5, handletextpad=0.5, borderpad=0.5, markerscale=1.5,
                   prop={'size': 12})

# 축과 배경 제거
ax.set_axis_off()

# legend만 저장
# fig.savefig('./IDEA/multi_weather_legend_only.pdf', bbox_inches='tight')
fig.savefig('./IDEA/legend_only.pdf', bbox_inches='tight')
