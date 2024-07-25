#####
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm #https://blog.csdn.net/qq_36056219/article/details/112118602
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

type=1; # 1 for up-crossing, 2 for down-crossing

data=pd.read_excel(r'D:\StudyAndWork\研二\南湖\水体模拟\看代码\WaveParticles\Assets\Scripts\Log\06-20 11data.xlsx'); # First column is sampling time, followed by water level columns

row=np.size(data,0);
col=np.size(data,1);

time=data.iloc[:,0]; # sampling time


for i in range(1,col-1):
    dat=data.iloc[:,i]; # current data column 第i列
    flucs=dat-np.mean(dat); # fluctuations around the mean
    sgn=np.sign(flucs); # amplitude signs
    cross=np.zeros((row-1,1));#row-1这么多行，1列

    for ii in range(0,row-2):#0，1，...row-3
        if type==1: #up-crossing
            if sgn[ii]==-1 and sgn[ii+1]==+1:
                cross[ii]=1; # putting 1 for zero-cross locations#标记一下zero-cross的点（这个点是负的，下个点就是正的了）
        elif type==2: #down-crossing
            if sgn[ii]==+1 and sgn[ii+1]==-1:
                cross[ii]=1; # putting 1 for zero-cross locations
            
    ind=[]; # zero-crossing indices
    
    for iii in range(0,row-1):
        if cross[iii]==1:
            ind.append(iii)
            
    n_waves=np.size(ind)-1; # number of waves
            
    ind_n=[];
    H=np.empty((n_waves,1)); # wave heights
    T=np.empty((n_waves,1)); # wave periods

    for n in range(0,n_waves):
        
        start=ind[n]+1; # start indice for the current wave #+1，意思就是这个点已经是正的了
        end=ind[(n+1)]; # end indice for the current wave
        
        if end-start>1:
            flucs_n=dat[start:end];#这一段wave里面的高度数据都提取出来
            a_cre=np.nanmax(flucs_n); # max crest amplitude
            a_tro=np.nanmin(flucs_n); # min trough amplitude
            
            H[n]=a_cre+np.absolute(a_tro); # wave heights 最高和最低的差
            T[n]=time[end]-time[start]; # wave periods
            
    out=np.column_stack((H,T));
    # print(out)
    np.savetxt('Column'+str(i)+'.txt',out,header='(H,T)')

    plt.plot(H)
    plt.ylabel('H (m)')
    plt.xlabel('Wave no.')
    plt.title('Column '+str(i))
    plt.show()

    Hdata=H/np.mean(H);
    sns.set_palette("hls")
    # sns.set_style("whitegrid")
    plt.figure(dpi=120)
    sigma = [0.7, 0.75, 0.8, 0.9]
    x = np.linspace(0, 5, 100)
    # 计算概率密度函数值
    pdf = rayleigh_distribution(x, sigma[0])
    pdf1 = rayleigh_distribution(x, sigma[1])
    pdf2 = rayleigh_distribution(x, sigma[2])
    pdf3 = rayleigh_distribution(x, sigma[3])
    # 绘制概率密度函数图
    plt.plot(x, pdf, label="a=0.7")
    plt.plot(x, pdf1, label='a=0.75')
    plt.plot(x, pdf2, label='a=0.8')
    plt.plot(x, pdf3, label='a=0.9')
    plt.legend()

    sns.set(style='dark')
    sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
    g = sns.distplot(Hdata,
                     hist=True,
                     kde=True,  # 开启核密度曲线kernel density estimate (KDE)
                     kde_kws={'linestyle': '--', 'linewidth': '1', 'color': '#c72e29',
                              # 设置外框线属性
                              },
                     #fit=norm,
                     color='#098154',
                     axlabel='H/H mean',  # 设置x轴标题
                     )
    plt.show()

    del [dat,flucs,sgn,cross,ind,H,T,Hdata]


