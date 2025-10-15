import matplotlib.pyplot as plt

# 예제 데이터
list1 = [0.5, 0.7, 0.8, 1.0, 1.1]
list2 = [0.3, 0.5, 0.6, 0.9, 1.0]

# x축 값을 위한 인덱스 생성
x_values = range(len(list1))

# 그래프 그리기
plt.figure(figsize=(8, 6))

# List 1 산점도와 점선
plt.scatter(x_values, list1, color='blue', label='List 1')
plt.plot(x_values, list1, color='blue', linestyle='--')

# List 2 산점도와 점선
plt.scatter(x_values, list2, color='orange', label='List 2')
plt.plot(x_values, list2, color='orange', linestyle='--')

# 그래프 제목과 축 레이블 설정
plt.title('Scatter Plot with Connected Points')
plt.xlabel('Index')
plt.ylabel('Values')

# 범례 표시
plt.legend()

# legend만 저장
fig.savefig('./IDEA/resize.pdf', bbox_inches='tight')
