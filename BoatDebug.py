import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 受力画图
angles = [1.201, 29.580, 56.99, 88.828, 118.97, 149.22, 179.86]
FX = [-16.592, -3.368, 0.651, 0.0367, 56.364, 42.24, 16.564]
FY = [0.446, 39.666, 48.731, 51.103, 59.640, 28.172, 0]
FXOld = [-11.625, -3.385, 0.910, 0.0681, 72.201, 50.987, 25.825]
FYOld = [0.738, 20.773, 41.29, 67.324, 62.655, 31.245, 0.074]
plt.ylabel('FX')
plt.xlabel('风向角')
plt.title('纵向力')
plt.plot(angles, FX, label='本方法', color='#fa0807')
plt.plot(angles, FXOld, label='藤原敏文', color='#098154')
plt.legend()
plt.show()

plt.ylabel('FY')
plt.xlabel('风向角')
plt.title('横向力')
plt.plot(angles, FY, label='本方法', color='#fa0807')
plt.plot(angles, FYOld, label='藤原敏文', color='#098154')
plt.legend()
plt.show()

# 面积统计
filename = r'\11-28 10 28data'
data = pd.read_excel(
    r'D:\StudyAndWork\研二\南湖\水体模拟\看代码\WaveParticles\Assets\Scripts\Log\Boat' + filename + '.xlsx')  # 第一列是时间，后面是正面和侧面的面积
time = data.iloc[100:900, 0]  # sampling time 第一列
col1 = data.iloc[100:900, 1]
col2 = data.iloc[100:900, 2]

# plt.plot(col1)
plt.ylabel('Area')
plt.xlabel('Time')
plt.title('Areas ')
plt.plot(time, col1, label='正面投影面积', color='#fa0807')
plt.plot(time, col2, label='侧面投影面积', color='#098154')
plt.legend()
plt.show()

# 分析平均值和极差和百分比

mean = np.sum(col1) / len(col1)
print("Mean XArea", mean)
max = np.max(col1)
print("max", max)
diff1 = (max - mean) / mean
print("max diff", diff1)  # 42%

min = np.min(col1)
print("min", min)
diff2 = (mean - min) / mean
print("min diff", diff2)  # 57%

# 对于侧面而言
mean = np.sum(col2) / len(col2)
print("Mean YArea", mean)
max = np.max(col2)
print("max", max)
diff1 = (max - mean) / mean
print("max diff", diff1)  # 63%

min = np.min(col2)
print("min", min)
diff2 = (mean - min) / mean
print("min diff", diff2)  # 55%

limit = (max - min) / min
print(" limit diff", limit)

# 面积变化对应的受力变化
data = pd.read_excel(
    r'D:\StudyAndWork\研二\南湖\水体模拟\看代码\WaveParticles\Assets\Scripts\Log\Boat' + filename + '.xlsx',
    sheet_name=1)  # 读取第二个sheet
time = data.iloc[100:900, 0]  # sampling time 第一列
FX = data.iloc[100:900, 1]
FY = data.iloc[100:900, 2]

# plt.plot(col1)
plt.ylabel('FX')
plt.xlabel('Time')
plt.title('纵向力')
plt.plot(time, FX, label='FX', color='#fa0807')
plt.legend()
plt.show()

plt.ylabel('FY')
plt.xlabel('Time')
plt.title('横向力')
plt.plot(time, FY, label='FY', color='#098154')
plt.legend()
plt.show()

# Old方法的FX和Fy
data = pd.read_excel(
    r'D:\StudyAndWork\研二\南湖\水体模拟\看代码\WaveParticles\Assets\Scripts\Log\Boat' + filename + '.xlsx',
    sheet_name=2)  # 读取第二个sheet
time = data.iloc[100:900, 0]  # sampling time 第一列
FX = data.iloc[100:900, 1]
FY = data.iloc[100:900, 2]

# plt.plot(col1)
plt.ylabel('FX')
plt.xlabel('Time')
plt.title('纵向力')
plt.plot(time, FX, label='藤原敏文FX', color='#fa0807')
plt.legend()
plt.show()

plt.ylabel('FY')
plt.xlabel('Time')
plt.title('横向力')
plt.plot(time, FY, label='藤原敏文FY', color='#098154')
plt.legend()
plt.show()

# 位置的图
data = pd.read_excel(
    r'D:\StudyAndWork\研二\南湖\水体模拟\看代码\WaveParticles\Assets\Scripts\Log\Boat\11-28 11 01data.xlsx',
    sheet_name=3)  # 读取第二个sheet
posx = data.iloc[0:900, 1]
posy = data.iloc[0:900, 2]

data2 = pd.read_excel(
    r'D:\StudyAndWork\研二\南湖\水体模拟\看代码\WaveParticles\Assets\Scripts\Log\Boat\11-28 11 07data.xlsx',
    sheet_name=3)  # 读取第二个sheet
posx2 = data2.iloc[0:100, 1]  # 我们的方法
posy2 = data2.iloc[0:100, 2]
# plt.plot(col1)
plt.ylabel('y')
plt.xlabel('x')
plt.title('位置')
plt.plot(posx, posy, label='藤原敏文', color='#fa0807')
plt.plot(posx2, posy2, label='本方法', color='#098154')
plt.legend()
plt.show()
