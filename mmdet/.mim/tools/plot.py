import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 이름
datasets = ["coco", "bdd", "cityscapes", "detrac", "exdark", "kitti", "self_driving2coco", "roboflow2coco", "udacity2coco", "traffic2coco"]

# 각 데이터셋마다 distance 값 (meta-set)
# values_conf_low = [8.9055, 11.5696, 28.8531, 11.7571, 12.1853, 8.1074, 9.3029, 8.8785, 17.7258, 9.0439]
# values_conf_high = [5.5643, 4.7613, 5.4921, 7.3473, 10.8184, 4.6244, 2.6875, 6.0280, 3.1089, 6.5312]

# 각 데이터셋마다 iou 값 (original dataset)
values_conf_low = [0.2502, 0.3929, 0.4418, 0.2969, 0.3022, 0.2847, 0.3548, 0.3011, 0.3213, 0.2206]
values_conf_high = [0.6519, 0.7461, 0.8057, 0.6123, 0.6396, 0.7260, 0.7250, 0.6463, 0.6984, 0.5689]

# x축 좌표
x = np.arange(len(datasets))

# 막대 폭
bar_width = 0.35

# 그래프 그리기
fig, ax = plt.subplots()


# 첫 번째 값 (a)에 대한 막대
bar_a = ax.bar(x - bar_width/2, values_conf_low, bar_width, label='Low Confidence', color='deepskyblue')

# 두 번째 값 (b)에 대한 막대
bar_b = ax.bar(x + bar_width/2, values_conf_high, bar_width, label='High Confidence', color='tomato')

# y축 범위 설정 (최대값을 기준으로 여유 공간 추가)
ax.set_ylim(0, max(values_conf_low + values_conf_high) * 1.2)

# x축에 데이터셋 이름 표시
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha="right")  # x축 레이블 회전

# 레이블 추가
ax.set_xlabel('Datasets')
# ax.set_ylabel('Distance')
ax.set_ylabel('IoU')
# ax.set_title('Distance Values for Each Dataset')
ax.set_title('IoU Values for Each Dataset')

# 범례 표시
ax.legend()

# 막대 위에 값 레이블 표시
ax.bar_label(bar_a, fmt='%.2f', padding=3)
ax.bar_label(bar_b, fmt='%.2f', padding=3)

# 그래프를 원하는 경로에 저장
output_dir = "IDEA/bar_chart_iou.png"  # 저장할 경로와 파일명 설정
plt.tight_layout()  # 레이아웃을 자동으로 조정 (겹침 방지)
plt.savefig(output_dir, dpi=300)  # 고해상도(dpi=300)로 저장