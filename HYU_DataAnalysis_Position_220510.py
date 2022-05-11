#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import quad
import statsmodels.api as sm
import statsmodels.formula.api as smf
from glob import glob
from detecta import detect_cusum
from tnorma import tnorma
from detecta import detect_onset
from detecta import detect_peaks
from scipy import signal
from scipy import stats
import os
from mpl_toolkits.mplot3d import Axes3D


# In[237]:


def outlier_iqr(data): 

    # lower, upper 글로벌 변수 선언하기     
    global lower, upper    
    
    # 4분위수 기준 지정하기     
    q25, q75 = np.quantile(data, 0.25), np.quantile(data, 0.75)          
    
    # IQR 계산하기     
    iqr = q75 - q25    
    
    # outlier cutoff 계산하기     
    cut_off = iqr * 1.5          
    
    # lower와 upper bound 값 구하기     
    lower, upper = q25 - cut_off, q75 + cut_off     
    print('IQR은',iqr, '이다.')     
    print('lower bound 값은', lower, '이다.')     
    print('upper bound 값은', upper, '이다.')    
    
    # 1사 분위와 4사 분위에 속해있는 데이터 각각 저장하기     
    data1 = data[data > upper]     
    data2 = data[data < lower]    
    
    # 이상치 총 개수 구하기
    return print('총 이상치 개수는', data1.shape[0] + data2.shape[0], '이다.')


# In[294]:


path = os.getcwd()
path = path.replace('\\', '/')
path
file_list = sorted(glob(path + '/'+'c0011*_07*position*', recursive = True))
file_list


# In[295]:


file = pd.read_csv(file_list[0])
file.head()


# In[ ]:





# In[306]:


# 변인 설정
xs = np.array(file['X_3D_SPINE_CHEST(2)'])/10
ys = np.array(file['Y_3D_SPINE_CHEST(2)'])/10
zs = np.array(file['Z_3D_SPINE_CHEST(2)'])/10
# 이상치 탐지, 아직 보간은 안함
# outlier_iqr(ys)
# ys = np.array(ys.loc[(ys < upper) & (ys > lower)])/10
# outlier_iqr(xs)
# xs = np.array(xs.loc[(xs < upper) & (xs > lower)])/10
# outlier_iqr(zs)
# zs = np.array(zs.loc[(zs < upper) & (zs > lower)])/10
position_name = ["xs", "ys", "zs"]
position_value = [xs, ys, zs]


# In[307]:


from optcutfreq import optcutfreq
from scipy.signal import butter, lfilter, filtfilt
freq = 15
for i,j in zip(position_value, position_name):
        fc_opt = optcutfreq(i, freq=freq, show=True)
        b2, a2 = butter(2, fc_opt/(freq/2), btype = 'low')
        globals()['{}_filt'.format(j)] = filtfilt(b2, a2, i)
    
print(np.ptp(ys_filt))


# In[308]:


fig = plt.figure(figsize=(30, 10))
ax1 = plt.subplot(3,1,1)
ax1.plot(xs_filt)

ax2 = plt.subplot(3,1,2)
ax2.plot(ys_filt)

ax3 = plt.subplot(3,1,3)
ax3.plot(zs_filt)


# In[309]:


# creating figure
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection = "3d")

# 축 크기 조정
# ax.set_xlim(-100, 100)
# ax.set_zlim(-100, 100)

# creating the plot
plot_geeks = ax.scatter(xs, zs, ys, color='green')
# setting title and labels
ax.set_title("3D plot")
ax.set_xlabel('x-axis')
ax.set_ylabel('z-axis')
ax.set_zlabel('y-axis')
  
# displaying the plot
plt.show()


# In[310]:


plt.figure(figsize=(30,10))
plt.subplot(2,1,1)
plt.plot(ys)

plt.subplot(2,1,2)
plt.plot(ys_filt)


# In[311]:


b = detect_peaks(ys_filt, mph=-30, mpd=20, show=True)
print(b)
b = np.array(b)
ys_filt = np.array(ys_filt)


# In[314]:


vel = []
for i in range(len(zs)):
    vel.append(np.sqrt((zs[i]-zs[i-1])**2+(xs[i]-xs[i-1])**2+(ys[i]-ys[i-1])**2)*15)


# In[315]:


plt.figure(figsize=(100,20))
plt.plot(vel)


# In[316]:


b = detect_peaks(vel, mph=20, mpd=15, show=True)
print(b)
b = np.array(b)
vel = np.array(vel)


# In[317]:


v_peak = vel[b]
np.average(v_peak)


# In[157]:


a = detect_onset(vel, 1.4, n_above=10, n_below=1, show=True)
print(a)
len(a)


# In[98]:


a = detect_onset(vel, , n_above=10, n_below=1, show=True)
print(a)
len(a)


# In[16]:


# Game Level, Stage TImeline
Timeline = pd.read_excel('C:/Users/ka4026/Desktop/한양대_게임분석_Event_Timeline.xlsx', sheet_name=1)
Timeline


# In[21]:


Tl_6_2 = []
for i, j in zip(Timeline.iloc[2:,33], Timeline.iloc[2:,34]):
    Tl_6_2.append([i, j])
    print(Tl_6_2)
len(Tl_6_2)


# In[ ]:





# In[ ]:




