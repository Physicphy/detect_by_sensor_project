# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from scipy.fftpack import dct
from scipy.fftpack import idct


# %% [markdown]
# #### -- 모델 준비

# %%
# 원전 내부의 충돌체 정보를 네개의 센서 정보만으로 특정해내기
# 데이터 출처 : https://dacon.io/competitions/official/235614/overview/description/
train_features = pd.read_csv('D:/Data/KAERI_dataset/train_features.csv')
train_target = pd.read_csv('D:/Data/KAERI_dataset/train_target.csv')


# %%
train_features.shape, train_target.shape


# %%
train_features.head()
# %%
# shift를 이용해서, 감지된 신호의 극점들을 찾아 기록하는 함수
def pick_peak(data0):
	data = data0.copy()
	for sensor in ['S1','S2','S3','S4']:
		data[sensor+'_peak'] = -((data[sensor].shift(1)-data[sensor])/abs(data[sensor].shift(1)-data[sensor])+(data[sensor].shift(-1)-data[sensor])/abs(data[sensor].shift(-1)-data[sensor]))/2
		data = data.fillna(0)
		# for i in data['id']:
		# 	last = data[data['id']==0].index[-1]
		# 	data.loc[last,sensor+'_peak'] = 0

	return data
# %%
def first_max_gap(data):
    peak_df = pick_peak(data)
    peak_s_list = []
    for s in ['S1','S2','S3','S4']:
        peak_s = peak_df.copy()
        peak_s = peak_s[peak_s[s+'_peak']!=0][['id',s,s+'_peak']]
        peak_s[s+'_up'] = peak_s[s].shift(-1)
        peak_s['g_'+s] = peak_s[s]*peak_s[s+'_peak']+peak_s[s+'_up']*peak_s[s+'_peak']*(-1)
        peak_s = peak_s[['id','g_'+s]].dropna()
        peak_s['g_'+s+'_peak'] = -((peak_s['g_'+s].shift(1)-peak_s['g_'+s])/abs(peak_s['g_'+s].shift(1)-peak_s['g_'+s])+(peak_s['g_'+s].shift(-1)-peak_s['g_'+s])/abs(peak_s['g_'+s].shift(-1)-peak_s['g_'+s]))/2
        peak_s = peak_s.fillna(0)
        peak_s['fm_gap_'+s] = peak_s['g_'+s]*peak_s['g_'+s+'_peak']
        peak_s = peak_s[peak_s['fm_gap_'+s]>peak_s['fm_gap_'+s].mean()][['id','fm_gap_'+s]].drop_duplicates(['id'],keep='first')
        peak_s = peak_s.reset_index()[['fm_gap_'+s]]
        peak_s_list.append(peak_s)
    peak_df = pd.concat(peak_s_list,axis=1)
    return peak_df
# %%
def first_max_gap_interval(data):
    peak_df = pick_peak(data)
    peak_s_list = []
    for s in ['S1','S2','S3','S4']:
        peak_s = peak_df.copy()
        peak_s = peak_s[peak_s[s+'_peak']!=0][['id','Time',s,s+'_peak']]
        peak_s[s+'_up'] = peak_s[s].shift(-1)
        peak_s[s+'_time_up'] = peak_s['Time'].shift(-1)
        peak_s['g_'+s] = peak_s[s]*peak_s[s+'_peak']+peak_s[s+'_up']*peak_s[s+'_peak']*(-1)
        peak_s['intv_'+s] = abs(peak_s['Time']-peak_s[s+'_time_up'])*10**6
        peak_s = peak_s[['id','g_'+s,'intv_'+s]].dropna()
        peak_s['g_'+s+'_peak'] = -((peak_s['g_'+s].shift(1)-peak_s['g_'+s])/abs(peak_s['g_'+s].shift(1)-peak_s['g_'+s])+(peak_s['g_'+s].shift(-1)-peak_s['g_'+s])/abs(peak_s['g_'+s].shift(-1)-peak_s['g_'+s]))/2
        peak_s = peak_s.fillna(0)
        peak_s['fm_gap_'+s] = peak_s['g_'+s]*peak_s['g_'+s+'_peak']
        peak_s['fm_intv_'+s] = peak_s['intv_'+s]*peak_s['g_'+s+'_peak']
        peak_s = peak_s[peak_s['fm_gap_'+s]>peak_s['fm_gap_'+s].mean()][['id','fm_gap_'+s,'fm_intv_'+s]].drop_duplicates(['id'],keep='first')
        peak_s = peak_s.reset_index()[['fm_gap_'+s,'fm_intv_'+s]]
        peak_s['delta_'+s] = peak_s['fm_gap_'+s]/peak_s['fm_intv_'+s]
        peak_s = peak_s[['fm_gap_'+s,'delta_'+s]]
        peak_s_list.append(peak_s)
    peak_df = pd.concat(peak_s_list,axis=1)
    return peak_df
# %%
def find_firt_min_amp(data0,min_amp=937.55):
    data = data0.copy()
    cond_min = (np.abs(data['S1']) > min_amp) | (np.abs(data['S2']) > min_amp) | (np.abs(data['S3']) > min_amp) | (np.abs(data['S4']) > min_amp)
    data_active = data[cond_min]
    data_active = data_active.drop_duplicates(['id'],keep='first')

    return data_active

# %%
# scipy에서 Discrete Cosine Transform을 사용, 원하는 만큼만 잘라낼 수 있게 함수 설정
def fourier_trsf(data,sensor,idx=10,cutoff=65):
	cond_id = data['id']==idx
	wave = data.loc[cond_id,sensor].values
	time = data.loc[cond_id,'Time']
	fft_wave = dct(wave, type=2,n=time.shape[0],norm='ortho')
	freq = np.fft.fftfreq(wave.size,d=0.000004)
	cw = np.copy(fft_wave)
	cw[cutoff:]=0
	fft_wave_2 = np.real(idct(cw,norm='ortho'))
	
	return {"cw":cw[:cutoff],"fft":fft_wave, "freq":freq, "fft_cutoff":fft_wave_2, "time":time, "wave":wave}

# %%

def find_unique_freq(data0,head=40):
    data = data0.copy()
    id_list = np.array(data['id'].unique())
    set_dict = {}
    n = data[data['id']==0].shape[0]
    nn = int(n/2)+1

    for s in ['S1','S2','S3','S4']:
        min_set = set(range(0,nn))
        for i in id_list:
            fft_wave = fourier_trsf(data=data,sensor=s,id=i)
            freq = fft_wave['freq'][0:nn]
            amp = fft_wave['fft'][0:nn]
            abs_amp = abs(amp)

            df_wave = pd.DataFrame([freq,amp,abs_amp]).T
            df_wave.columns = ['freq','amp','abs_amp']
            set_i = set(df_wave.sort_values(by='abs_amp',ascending=False).head(head).index)

            min_set = min_set - set_i

        set_dict[s]=min_set
    return set_dict

# %%
# 65번째 까지(0~64) 킵해보자
# 일단 feature로 만들어서 넣어주는 함수를 짜자
# column이름은 f1_0~f4_65같은 식으로 넣기
def fourier_feature(data0,cutoff=65):
    data = data0.copy()
    id_list = np.array(data['id'].unique())
    df_id = pd.DataFrame(id_list,columns=['id'])
    df_list = [df_id]

    for s in ['S1','S2','S3','S4']:
        df_s = []
        for i in id_list:
            fft_wave = fourier_trsf(data=data,sensor=s,idx=i,cutoff=cutoff)
            amp = fft_wave['cw']
            
            df_wave = pd.DataFrame(amp).T
            df_wave.columns = [s+'_f'+str(n) for n in range(cutoff)]
            df_s.append(df_wave)
        df_sensor = pd.concat(df_s,axis=0).reset_index(drop=True)
        df_list.append(df_sensor)

    df_tot = pd.concat(df_list,axis=1)

    return df_tot

# %%
# 확인 해보니, S3가 먼저 신호를 받은 경우가 한번도 없는 것으로 나온다
# 그래서 S3가 항상 고려되지 않은채로 분류 된 것...
# 추후에도 S3값이 고려 될 수 있게, 축을 새로 잡아주기
def reset_axis(data0,new_axis=('A','B','C','D')):
    data = data0.copy()
    
    # A=(S1+S2+S3+S4)/4, B=(S1+S2-S3-S4)/4, C=(S1-S2-S3+S4)/4, D=(S1-S2+S3-S4)/4
    ns1,ns2,ns3,ns4 = data['S1'],data['S2'],data['S3'],data['S4']
    data[new_axis[0]] = (ns1+ns2+ns3+ns4)/4
    data[new_axis[1]] = (ns1+ns2-ns3-ns4)/4
    data[new_axis[2]] = (ns1-ns2-ns3+ns4)/4
    data[new_axis[3]] = (ns1-ns2+ns3-ns4)/4
    data = data.drop(['Time','S1','S2','S3','S4'],axis=1)
    
    return data

# %%
# 앞서 수행했던 데이터 전처리 및 feature engineering을 수행해주는 함수
def feature_eng_df(data,cutoff=80):   
    cond_0 = (data['S1'] != 0) | (data['S2'] != 0) | (data['S3'] != 0) | (data['S4'] != 0)
    data_active = data[cond_0]
    data_active = data_active.drop_duplicates(['id'],keep='first')
    
    new_axis = ('A','B','C','D')
    data_new = reset_axis(data,new_axis=new_axis)
    cond_new = (data_new['A'] != 0) | (data_new['B'] != 0) | (data_new['C'] != 0) | (data_new['D'] != 0)  
    data_active_new = data_new[cond_new]
    data_active_new = data_active_new.drop_duplicates(['id'],keep='first')
    
    data_active = data_active.merge(data_active_new,on='id')
    
    for s in ['S1','S2','S3','S4']:
        min_s = data.groupby(by='id').min()[s]
        max_s = data.groupby(by='id').max()[s]
        gap_s = max_s - min_s
        gap_s = gap_s.reset_index()
        gap_s.columns = ['id','gap_'+s]
        data_active = data_active.merge(gap_s,on='id')

    gap_data = first_max_gap_interval(data)
    data_active = pd.concat([data_active,gap_data],axis=1)

    data_active['Time'] = (data_active['Time']*10**6).astype('int')

    data[(data['S2'] != 0)].drop_duplicates(['id'],keep='first')[['id','Time']]

    for s in ['S1','S2','S3','S4']:
        cond_t = (data[s] != 0)
        active_time = data[cond_t].drop_duplicates(['id'],keep='first')[['id','Time']]
        active_time['Time'] = (active_time['Time']*10**6).astype('int')
        active_time.columns = ['id','active_time_'+s]
        data_active = data_active.merge(active_time,on='id')

    data_active['R12'] = (data_active['active_time_S1']+data_active['active_time_S2'])/(data_active['active_time_S3']+data_active['active_time_S4'])
    data_active['R13'] = (data_active['active_time_S1']+data_active['active_time_S3'])/(data_active['active_time_S2']+data_active['active_time_S4'])
    data_active['R14'] = (data_active['active_time_S1']+data_active['active_time_S4'])/(data_active['active_time_S2']+data_active['active_time_S3'])

    data_active['RMS_S'] = (data_active['S1']**2+data_active['S2']**2+data_active['S3']**2+data_active['S4']**2)**0.5
    data_active['RMS_gap'] = (data_active['gap_S1']**2+data_active['gap_S2']**2+data_active['gap_S3']**2+data_active['gap_S4']**2)**0.5
    data_active['RMS_time'] = (data_active['active_time_S1']**2+data_active['active_time_S2']**2+data_active['active_time_S3']**2+data_active['active_time_S4']**2)**0.5
    
    data_fft = fourier_feature(data,cutoff=cutoff)
    data_active = data_active.merge(data_fft,on='id')

    return data_active
# %%
# id 하나만 찍어서 확인, 총 375개
train_features[train_features['id']==0]
# %%
# id가 총 몇개인지 확인, 2800개
len( train_features['id'].unique() )
# %%
# id마다 가지고 있는 행 개수가 다 같을까?
# 375개로 나누어 보니 딱 떨어지긴 한다
train_features.shape[0]/375
# %%
# time은 id마다 다를까? 아니면 동일할까?
# time의 종류도 딱 375개 인걸로 봐선 동일한 시간 간격으로 기록된 것들이다
len(train_features['Time'].unique() )
# %%
# id=10을 예시로 삼아 살펴보자
train_features[train_features['id']==10]
# %%
# 이걸 그대로, 시간에 따라 찍어보면....
# 이런 파형 그래프가 된다. 이런게 2800개씩 있다!
# 이 파형 그래프를 이용해서, 타겟을 예측하면 되는데...
def wave_graph(select_id=0):
    fig = plt.figure(figsize=(15,10))
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)
    ax = gs.subplots()
    fig.suptitle('id = '+str(select_id))

    for r,c,s in [(0,0,'S1'),(0,1,'S2'),(1,0,'S3'),(1,1,'S4')]:
        w0 = fourier_trsf(data=train_features,sensor=s,idx=select_id,cutoff=50)
        ax[r][c].plot(w0['time']*10**6, w0['wave'])
        ax[r][c].set_xlabel('Time')
        ax[r][c].set_ylabel('Amplitude')
        ax[r][c].grid(True)
        ax[r][c].set_title(s)

wave_graph(10)
# %%
# 여기엔 사실 375개의 점이 있다
# 이걸 그대로 다 사용하면, 375*4 = 1500개의 feature가 있어야 한다
# 너무 많다!
fig = plt.figure(figsize=(15,10))
gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)
ax = gs.subplots()
fig.suptitle('id = 10')

selcet_id = 10

for r,c,s in [(0,0,'S1'),(0,1,'S2'),(1,0,'S3'),(1,1,'S4')]:
    w0 = fourier_trsf(data=train_features,sensor=s,idx=selcet_id,cutoff=50)
    ax[r][c].scatter(w0['time']*10**6, w0['wave'])
    ax[r][c].set_xlabel('Time')
    ax[r][c].set_ylabel('Amplitude')
    ax[r][c].grid(True)
    ax[r][c].set_title(s)
# %%
np.pi
# %%
# 이걸 그대로 쓸수는 없으니, 대책이 필요하다
# 그래서 Fourier Transform이란 것을 사용!
# 이걸 이용하면 더 적은 데이터로, 근사적으로 진동을 다룰 수 있다
w0 = fourier_trsf(data=train_features,sensor='S2',idx=10,cutoff=30)

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(w0['time']*10**6, w0['wave'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

markerline, stemlines, baseline = ax2.stem(
    w0['freq'][:int(375/2)+1], w0['fft'][:int(375/2)+1],
    linefmt='k-', markerfmt='D ', basefmt='k-'
    ) 
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Amplitude')
ax2.grid(True)

plt.setp(stemlines, linewidth = 1)
plt.setp(markerline, markersize = 4.5)

plt.show()
# %%
# 구성 성분을 뜯어본 후에, 거기서 가장 세기가 큰 것들 30개만 고르면...
# 대부분 낮은 주파수 쪽에 쏠려 있다
w0 = fourier_trsf(data=train_features,sensor='S2',idx=10,cutoff=30)

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(w0['time']*10**6, w0['wave'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

top_num = 30
top_amp = np.where(abs(w0['fft'][:int(375/2)+1])>=sorted(abs(w0['fft'][:int(375/2)+1]))[-top_num],w0['fft'][:int(375/2)+1],np.nan)
rest_amp = np.where(abs(w0['fft'][:int(375/2)+1])>=sorted(abs(w0['fft'][:int(375/2)+1]))[-top_num],np.nan,w0['fft'][:int(375/2)+1])

markerline1, stemlines, baseline = ax2.stem(
    w0['freq'][:int(375/2)+1], rest_amp,
    linefmt='k-', markerfmt='D ', basefmt='k-'
    ) 
markerline2, stemlines, _ = ax2.stem(
    w0['freq'][:int(375/2)+1], top_amp,
    linefmt='k-', markerfmt='rD ', basefmt='k-'
    ) 
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Amplitude')
ax2.grid(True)

plt.setp(stemlines, linewidth = 1)
plt.setp(markerline1, markersize = 4.5)
plt.setp(markerline2, markersize = 4.5)

plt.show()
# %%
# 모든 id에 대해서, top 30에 들어가는 주파수들의 집합을 확인 해보면...
# 375종류 중에서 가장 작은 80가지라는 걸 알 수 있다
# 여기서 80번째 까지만, 남기면...
w0 = fourier_trsf(data=train_features,sensor='S2',idx=10,cutoff=30)

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(w0['time']*10**6, w0['wave'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

tot_freq = w0['fft'][:int(375/2)+1]
keep_freq = tot_freq.copy()
keep_freq[35:] = np.nan
drop_freq = tot_freq.copy()
drop_freq[:35] = np.nan

markerline1, stemlines, baseline = ax2.stem(
    w0['freq'][:int(375/2)+1], drop_freq,
    linefmt='k-', markerfmt='D ', basefmt='k-'
    ) 
markerline2, stemlines, _ = ax2.stem(
    w0['freq'][:int(375/2)+1], keep_freq,
    linefmt='k-', markerfmt='rD ', basefmt='k-'
    ) 
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Amplitude')
ax2.grid(True)

plt.setp(stemlines, linewidth = 1)
plt.setp(markerline1, markersize = 4.5)
plt.setp(markerline2, markersize = 4.5)

plt.show()
# %%
# 이렇게 근사시킨 형태의 파장을 얻을 수 있다
# 꽤 유사한 형태를 유지하고 있지만, 375개의 정보가 80개로 줄어들었다!
w0 = fourier_trsf(data=train_features,sensor='S2',idx=10,cutoff=35)

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(w0['time']*10**6, w0['wave'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

ax2.plot(w0['time']*10**6, w0['fft_cutoff'], 'r') 
ax2.set_xlabel('Time')
ax2.set_ylabel('Amplitude')
ax2.grid(True)

plt.show()
# %%
# fft의 원리를 간단히 보여주기

fig = plt.figure(figsize=(20,20))
gs = fig.add_gridspec(4, 2, hspace=0.25, wspace=0.2)
ax = gs.subplots()

select_id = 10
select_sensor = 'S2'

for r,nn in enumerate([2,3,5,10,25,50,80,150]):
    ww = fourier_trsf(data=train_features,sensor=select_sensor,idx=select_id,cutoff=nn)
    ax[r//2][r%2].plot(ww['time']*10**6, ww['fft_cutoff'])
    ax[r//2][r%2].set_xlabel('Time')
    ax[r//2][r%2].set_ylabel('Amplitude')
    ax[r//2][r%2].grid(True)
    ax[r//2][r%2].set_title('N='+str(nn))

plt.show()
# %%
# 혹시 train_target들 id 순서로 찍어보면 시계열 처럼 보일까? --> 다행히 아님
# 미리 세트를 세가지로 나누자 0.2 대 0.8로 나눠서 테스트, 훈련및 검증
# 훈련 및 검증을 0.2 대 0.8로 나눠서 검증, 훈련
# %%
sns.scatterplot(x=train_target['id'],y=train_target['X'])
# %%
sns.scatterplot(x=train_target['id'],y=train_target['Y'])
# %%
sns.scatterplot(x=train_target['id'],y=train_target['M'])
# %%
sns.scatterplot(x=train_target['id'],y=train_target['V'])
# %%
df_features = feature_eng_df(train_features,cutoff=80)
df_tot = df_features.merge(train_target,on='id')
print(df_tot.columns)
df_tot
# %%