import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_excel(
    r'D:\StudyAndWork\研二\南湖\水体模拟\看代码\WaveParticles\Assets\Scripts\Log\Boat\11-25 17 34data.xlsx')# 第一列是时间，后面是正面和侧面的面积
time = data.iloc[100:900, 0]  # sampling time 第一列
col1 = data.iloc[100:900, 1]
col2 = data.iloc[100:900, 2]

# plt.plot(col1)
plt.ylabel('Area')
plt.xlabel('Time')
plt.title('Areas ')
plt.plot(time, col1, label='正面投影面积',color='#fa0807')
plt.plot(time, col2, label='侧面投影面积',color='#098154')
plt.legend()
plt.show()

# 分析平均值和极差和百分比

mean=np.sum(col1)/len(col1)
print("Mean XArea", mean)
max=np.max(col1)
print("max", max)
diff1=(max-mean)/mean
print("max diff", diff1) # 42%

min=np.min(col1)
print("min", min)
diff2=(mean-min)/mean
print("min diff", diff2) # 57%

# 对于侧面而言
mean=np.sum(col2)/len(col2)
print("Mean YArea", mean)
max=np.max(col2)
print("max", max)
diff1=(max-mean)/mean
print("max diff", diff1) # 63%

min=np.min(col2)
print("min", min)
diff2=(mean-min)/mean
print("min diff", diff2) # 55%


