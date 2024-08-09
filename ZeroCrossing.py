#####
from math import sqrt

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm  # https://blog.csdn.net/qq_36056219/article/details/112118602
from scipy.stats import laplace


# %reset -sf
# 定义瑞利分布函数
def rayleigh_distribution(x, sigma):
    """
    计算瑞利分布的概率密度函数值
    :param x: 自变量
    :param sigma: 尺度参数 α=sqrt(2)*sigma
    :return: 概率密度函数值
    """
    return (x / sigma ** 2) * np.exp(-x ** 2 / (2 * sigma ** 2))


def clean(serie):
    output = serie[(np.isnan(serie) == False) & (np.isinf(serie) == False)]
    return output


type = 1;  # 1 for up-crossing, 2 for down-crossing

data = pd.read_excel(
    r'D:\StudyAndWork\研二\南湖\水体模拟\看代码\WaveParticles\Assets\Scripts\Log\08-08 22 02data.xlsx');  # First column is sampling time, followed by water level columns

row = np.size(data, 0);
col = np.size(data, 1);

time = data.iloc[:, 0];  # sampling time

for i in range(1, col - 1):
    dat = data.iloc[:, i];  # current data column 第i列
    flucs = dat - np.mean(dat);  # fluctuations around the mean
    sgn = np.sign(flucs);  # amplitude signs
    cross = np.zeros((row - 1, 1));  # row-1这么多行，1列

    for ii in range(0, row - 2):  # 0，1，...row-3
        if type == 1:  # up-crossing
            if sgn[ii] == -1 and sgn[ii + 1] == +1:
                cross[ii] = 1;  # putting 1 for zero-cross locations#标记一下zero-cross的点（这个点是负的，下个点就是正的了）
        elif type == 2:  # down-crossing
            if sgn[ii] == +1 and sgn[ii + 1] == -1:
                cross[ii] = 1;  # putting 1 for zero-cross locations

    ind = [];  # zero-crossing indices

    for iii in range(0, row - 1):
        if cross[iii] == 1:
            ind.append(iii)

    n_waves = np.size(ind) - 1;  # number of waves

    ind_n = [];
    H = np.empty((n_waves, 1));  # wave heights
    T = np.empty((n_waves, 1));  # wave periods

    for n in range(0, n_waves):

        start = ind[n] + 1;  # start indice for the current wave #+1，意思就是这个点已经是正的了
        end = ind[(n + 1)];  # end indice for the current wave

        if end - start > 1:
            flucs_n = dat[start:end];  # 这一段wave里面的高度数据都提取出来
            a_cre = np.nanmax(flucs_n);  # max crest amplitude
            a_tro = np.nanmin(flucs_n);  # min trough amplitude

            H[n] = a_cre + np.absolute(a_tro);  # wave heights 最高和最低的差
            T[n] = time[end] - time[start];  # wave periods

    out = np.column_stack((H, T));
    # print(out)
    np.savetxt('Column' + str(i) + '.txt', out, header='(H,T)')
    # print(H)
    plt.plot(H)
    plt.ylabel('H (m)')
    plt.xlabel('Wave no.')
    plt.title('Column ' + str(i))
    plt.show()

    Hdata = H / np.mean(H)
    Hdata = clean(Hdata)
    sns.set_palette("hls")
    # sns.set_style("whitegrid")
    plt.figure(dpi=120)
    sigma = [0.7, 0.75, 0.8,0.85, 0.9]
    x = np.linspace(0, 5, 100)
    # 计算概率密度函数值
    pdf = rayleigh_distribution(x, sigma[0])
    pdf1 = rayleigh_distribution(x, sigma[1])
    pdf2 = rayleigh_distribution(x, sigma[2])
    pdf3 = rayleigh_distribution(x, sigma[3])
    pdf4 = rayleigh_distribution(x, sigma[4])
    # 绘制概率密度函数图
    plt.plot(x, pdf, label="a=0.7")
    plt.plot(x, pdf1, label='a=0.75')
    plt.plot(x, pdf2, label='a=0.8')
    plt.plot(x, pdf3, label='a=0.85')
    plt.plot(x, pdf4, label='a=0.9')
    plt.legend()

    sns.set(style='dark')
    sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
    g = sns.distplot(Hdata,
                     hist=True,
                     kde=True,  # 开启核密度曲线kernel density estimate (KDE)
                     kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29',
                              # 设置外框线属性
                              },
                     # fit=norm,
                     color='#098154',
                     axlabel='H/H mean',  # 设置x轴标题
                     )
    # sns.kdeplot(Hdata)
    kde = stats.gaussian_kde(Hdata)
    # plt.hist(Hdata, density=True, alpha=0.5)
    X_plot = np.linspace(0, 5, 100)  # 和上面用到的x一样，划分成100个点
    predict = kde(X_plot)
    plt.plot(X_plot, predict, label="kde")
    plt.show()
    # 原生实现
    # 衡量线性回归的MSE 、 RMSE、 MAE、r2

    mse = np.sum((pdf - predict) ** 2) / len(pdf)
    rmse = sqrt(mse)
    mae = np.sum(np.absolute(pdf - predict)) / len(pdf)
    mape = 0
    smape = 0
    count = 0
    for i in range(len(pdf)):
        if pdf[i] != 0:
            mape += np.absolute((pdf[i] - predict[i]) / pdf[i])
            smape += np.absolute((pdf[i] - predict[i]) / ( (pdf[i] + predict[i]) / 2 ) )
            count = count + 1
    mape = mape / count
    smape =smape /count
    r2 = 1 - mse / np.var(pdf)  # 均方误差/方差
    print("pdf0 mae:", mae, "mse:", mse, " rmse:", rmse, " r2:", r2, " mape:", mape,"smape:", smape)

    mse1 = np.sum((pdf1 - predict) ** 2) / len(pdf1)
    rmse1 = sqrt(mse1)
    mae1 = np.sum(np.absolute(pdf1 - predict)) / len(pdf1)
    # mape1 = np.sum(np.absolute((pdf1 - predict) / pdf1)) / len(pdf1)
    mape1 = 0
    smape1 = 0
    # MAPE1 = np.zeros_like(pdf1)
    count = 0
    for i in range(len(pdf1)):
        if pdf1[i] != 0:
            mape1 += np.absolute((pdf1[i] - predict[i]) / pdf1[i])
            # MAPE1[i] = np.sum(np.absolute((pdf1[i] - predict[i]) / pdf1[i])) / len(pdf1)
            smape1 += np.absolute((pdf1[i] - predict[i]) / ((pdf1[i] + predict[i]) / 2))
            count = count + 1
    mape1 = mape1 / count
    smape =smape /count
    r21 = 1 - mse / np.var(pdf1)  # 均方误差/方差
    print("pdf1 mae:", mae1, "mse:", mse1, " rmse:", rmse1, " r2:", r21, " mape:", mape1,"smape:", smape1)

    mse2 = np.sum((pdf2 - kde(X_plot)) ** 2) / len(pdf2)
    rmse2 = sqrt(mse2)
    mae2 = np.sum(np.absolute(pdf2 - kde(X_plot))) / len(pdf2)
    mape2 = 0
    smape2 = 0
    count = 0
    for i in range(len(pdf2)):
        if pdf[i] != 0:
            mape2 += np.absolute((pdf2[i] - predict[i]) / pdf2[i])
            smape2 += np.absolute((pdf2[i] - predict[i]) / ((pdf2[i] + predict[i]) / 2))
            count = count + 1
    mape2 = mape2 / count
    r22 = 1 - mse / np.var(pdf2)  # 均方误差/方差
    print("pdf2 mae:", mae2, "mse:", mse2, " rmse:", rmse2, " r2:", r22, " mape:", mape2 ,"smape:", smape2)

    mse3 = np.sum((pdf3 - kde(X_plot)) ** 2) / len(pdf3)
    rmse3 = sqrt(mse3)
    mae3 = np.sum(np.absolute(pdf3 - kde(X_plot))) / len(pdf3)
    mape3 = 0
    smape3 = 0
    count = 0
    for i in range(len(pdf3)):
        if pdf3[i] != 0:
            mape3 += np.absolute((pdf3[i] - predict[i]) / pdf3[i])
            smape3 += np.absolute((pdf3[i] - predict[i]) / ((pdf3[i] + predict[i]) / 2))
            count = count + 1
    mape3 = mape3 / count
    r23 = 1 - mse / np.var(pdf3)  # 均方误差/方差
    print("pdf3 mae:", mae3, "mse:", mse3, " rmse:", rmse3, " r2:", r23, " mape:", mape3 ,"smape:", smape3)

    mse4 = np.sum((pdf4 - kde(X_plot)) ** 2) / len(pdf4)
    rmse4 = sqrt(mse4)
    mae4 = np.sum(np.absolute(pdf4 - kde(X_plot))) / len(pdf4)
    mape4 = 0
    smape4 = 0
    count = 0
    for i in range(len(pdf4)):
        if pdf4[i] != 0:
            mape4 += np.absolute((pdf4[i] - predict[i]) / pdf4[i])
            smape4 += np.absolute((pdf4[i] - predict[i]) / ((pdf4[i] + predict[i]) / 2))
            count = count + 1
    mape4 = mape4 / count
    r24 = 1 - mse / np.var(pdf4)  # 均方误差/方差
    print("pdf3 mae:", mae4, "mse:", mse4, " rmse:", rmse4, " r2:", r24, " mape:", mape4,"smape:", smape4)

    del [dat, flucs, sgn, cross, ind, H, T, Hdata]
