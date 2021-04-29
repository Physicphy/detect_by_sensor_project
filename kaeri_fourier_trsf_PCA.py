# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from lightgbm import train
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
train_features[['S1','S2','S3','S4']].max()
# %%
train_features[['S1','S2','S3','S4']].min()
# %%
train_features[['S1','S2','S3','S4']].mean()
# %%
train_features[['S1','S2','S3','S4']].mean().mean()
# %%
train_features[np.abs(train_features['S1'])>937.55 ]

# %%
def find_firt_min_amp(data0,min_amp=937.55):
    data = data0.copy()
    cond_min = (np.abs(data['S1']) > min_amp) | (np.abs(data['S2']) > min_amp) | (np.abs(data['S3']) > min_amp) | (np.abs(data['S4']) > min_amp)
    data_active = data[cond_min]
    data_active = data_active.drop_duplicates(['id'],keep='first')

    # for s in ['S1','S2','S3','S4']:
    #         cond_t = (train_features[s] > 937.55)
    #         active_time = data[cond_t].drop_duplicates(['id'],keep='first')[['id','Time']]
    #         active_time['Time'] = (active_time['Time']*10**6).astype('int')
    #         active_time.columns = ['id','active_time_'+s]
    #         data_active = data_active.merge(active_time,on='id')
    return data_active
# %%
(np.abs(find_firt_min_amp(train_features,min_amp=10000)['S4'])>10000).sum()
# %%
# 같은 인덱스(같은 충돌체)의 센서 정보를 시각화 해서 비교
# 이 파장들을 푸리에 트랜스폼 시켜서, 근사적으로 amplitude가 큰 경우만 몇가지 추려내 feature로 쓸 수 있을까?
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.lineplot(data=train_features.loc[train_features['id']==10],x='Time',y='S1')

plt.subplot(2,2,2)
sns.lineplot(data=train_features.loc[train_features['id']==10],x='Time',y='S2')

plt.subplot(2,2,3)
sns.lineplot(data=train_features.loc[train_features['id']==10],x='Time',y='S3')

plt.subplot(2,2,4)
sns.lineplot(data=train_features.loc[train_features['id']==10],x='Time',y='S4')
# %%
train_features.loc[train_features['id']==10,'Time'].shape[0]
# %%
train_features.loc[train_features['id']==10,'S1'].values
# %%
# scipy에서 Discrete Cosine Transform을 사용, 원하는 만큼만 잘라낼 수 있게 함수 설정
def fourier_trsf(data,sensor,id=10,cutoff=65):
	cond_id = data['id']==id
	wave = data.loc[cond_id,sensor].values
	time = data.loc[cond_id,'Time']
	fft_wave = dct(wave, type=2,n=time.shape[0])
	freq = np.fft.fftfreq(wave.size,d=0.000004)
	cw = np.copy(fft_wave)
	cw[cutoff:]=0
	fft_wave_2 = np.real(idct(cw))
	
	return {"cw":cw[:cutoff],"fft":fft_wave, "freq":freq, "fft_cutoff":fft_wave_2, "time":time, "wave":wave}
# %%
amp_w_0_s1 = fourier_trsf(data=train_features,sensor='S1',id=0,cutoff=1)['fft'][0:40]
freq_w_0_s1 = fourier_trsf(data=train_features,sensor='S1',id=0,cutoff=1)['freq'][0:40]
df_w_0_s1 = pd.DataFrame([freq_w_0_s1,amp_w_0_s1]).T
df_w_0_s1.columns = ['frequency','amplitude']
df_w_0_s1
# frequency는 id별로 전부 시간 기록 간격이 동일 하기 때문에 다르지 않음
# frequency정보는 amplitude를 기록하는 순서만 잘 지켜 주면 됨!
# 대신 문제는... 적어도 40개는 잡아줘야 한다는 건데, 이러면 특성만 160개 추가
# id=10 같이 복잡한 케이스는... 50개는 잡아줘야 하네 ㅋㅋ 200개 추가요!
# 전부 잡을 수는 없다. 일단 40개 정도만 넣어주자
# 꼭 다 넣을 필요는 없잖아? 제일 amplitude가 큰것만 넣어봐?

# 전체 id에 대해서, 가장 큰 apmlitude를 갖는 frequency를 30개씩만 뽑기
# 그렇게 뽑은 목록들을 set로 만들어서 교집합이 존재하는지 확인
# 만약에 꽤 많은 교집합이 남는다면, 그게 이 원자로 내부의 형태에 맞는 주파수일 가능성이 높다
# 해당 주파수들만 뽑아서 feature로 삼기!
# %%
w0 = fourier_trsf(data=train_features,sensor='S1',id=10,cutoff=30)

fig, ax = plt.subplots(2, 1)

ax[0].plot(w0['time'], w0['wave'])
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[0].grid(True)

ax[1].plot(w0['time'], w0['fft_cutoff'], 'r') 
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Amplitude')
ax[1].grid(True)

plt.show()
# %%
w0 = fourier_trsf(data=train_features,sensor='S1',id=10,cutoff=30)

fig, ax = plt.subplots(2, 1)

ax[0].plot(w0['time'], w0['wave'])
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[0].grid(True)

ax[1].plot(w0['freq'][:int(375/2)+1], w0['fft'][:int(375/2)+1], 'r') 
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Amplitude')
ax[1].grid(True)

plt.show()
# %%
amp_w_0_s1 = fourier_trsf(data=train_features,sensor='S1',id=0,cutoff=1)['fft'][0:int(375/2)+1]
freq_w_0_s1 = fourier_trsf(data=train_features,sensor='S1',id=0,cutoff=1)['freq'][0:int(375/2)+1]
df_w_0_s1 = pd.DataFrame([freq_w_0_s1,amp_w_0_s1]).T
df_w_0_s1.columns = ['frequency','amplitude']
df_w_0_s1['abs_amp'] = abs(df_w_0_s1['amplitude'])
freq_set = set()
set_0 = set(list(df_w_0_s1.sort_values(by='abs_amp',ascending=False).head(30).index))
# 굳이 frequency로 뽑을 필요도 없겠다. 순서만 알면 됨. 그냥 index번호를 뽑아버리자
# %%
# 함수로 짜보자!
# 상위 40개에서 겹치는 걸 찾으려고 하면... 당연하지만
# 없다! 그런건 없어! 
# 하지만 방법을 반대로 바꿔보자, 상위 40개에 절대 들어오지 않는 freq는 있을까?
# 공간이 제한되어 있기 때문에, 거의 나타나지 않는 freq가 있을 수 있다!
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
# rmv_freq = find_unique_freq(train_features)
# 계산이 좀 걸리지만, 이정도는 오케이
# %%
# 좋아! 있다! 그것도 꽤 많다! 96개
# 일단 188개에서, 96개를 빼버리자, 그러면 남는게 92개
# tot_rmv_freq = rmv_freq['S1'] & rmv_freq['S2'] & rmv_freq['S3'] & rmv_freq['S4']
# len(tot_rmv_freq)
# %%
# 한번 더 해보자, 이번엔 상의 30개에 들어오지 않는 freq를 찾아서 제외시키자
# rmv_freq_und30 = find_unique_freq(train_features,head=30)
# tot_rmv_freq_und30 = rmv_freq_und30['S1'] & rmv_freq_und30['S2'] & rmv_freq_und30['S3'] & rmv_freq_und30['S4']
# len(tot_rmv_freq_und30)
# 오케이! 108개다. 이만큼을 제외 시키면, 80개가 남는다
# 이정도면 해볼만도...? 
# %%
# 게다가 이겈ㅋㅋㅋ 0~79까지가 딱 남았네
# freq_idx_80 = np.array(set(range(0,188)) - tot_rmv_freq_und30)
# freq_idx_80
# %%
# 한번만 더 해보자! 상위 20개
# 112개 제외라, 76개가 남는다. 딱히 더 나아지진 않네
# rmv_freq_und20 = find_unique_freq(train_features,head=20)
# tot_rmv_freq_und20 = rmv_freq_und20['S1'] & rmv_freq_und20['S2'] & rmv_freq_und20['S3'] & rmv_freq_und20['S4']
# len(tot_rmv_freq_und20)
# %%
# freq_idx_76 = np.array(set(range(0,188)) - tot_rmv_freq_und20)
# freq_idx_76
# %%
# 실험 삼아 하나만 더 해보자, 상위 5개!
# rmv_freq_und5 = find_unique_freq(train_features,head=5)
# tot_rmv_freq_und5 = rmv_freq_und5['S1'] & rmv_freq_und5['S2'] & rmv_freq_und5['S3'] & rmv_freq_und5['S4']
# len(tot_rmv_freq_und5)
# %%
# freq_idx_60 = np.array(set(range(0,188)) - tot_rmv_freq_und5)
# freq_idx_60
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
            fft_wave = fourier_trsf(data=data,sensor=s,id=i,cutoff=cutoff)
            amp = fft_wave['cw']
            
            df_wave = pd.DataFrame(amp).T
            df_wave.columns = [s+'_f'+str(n) for n in range(cutoff)]
            df_s.append(df_wave)
        df_sensor = pd.concat(df_s,axis=0).reset_index(drop=True)
        df_list.append(df_sensor)

    df_tot = pd.concat(df_list,axis=1)

    return df_tot
# %%
# df_fft65 = fourier_feature(train_features,cutoff=65)
# df_fft65
# %%

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
def feature_eng_df(data,cutoff=40):   
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
    
    # data_fft = fourier_feature(data,cutoff=cutoff)
    # data_active = data_active.merge(data_fft,on='id')

    return data_active
# %%
df_fft = fourier_feature(train_features,cutoff=80)
# %%
df_fft.iloc[:,1:].shape
# %%
pca_fft = PCA(45)
pca_fft.fit(df_fft.iloc[:,1:])
# %%
pca_trsf = pca_fft.transform(df_fft.iloc[:,1:])
# %%
np.cumsum(pca_fft.explained_variance_ratio_)
# %%
pd.DataFrame(pca_trsf)
# %%
df_features = feature_eng_df(train_features)
df_features.head().T
# %%
df_features = pd.concat([df_features,pd.DataFrame(pca_trsf)],axis=1)
df_features.head().T

# %%
targets = list(train_target.columns)[1:]
features = list(df_features.columns)[1:]

# %%
df = df_features.merge(train_target,on='id')
df.head()


# %%
# 데이터를 학습용, 검증용으로 분리
df_train, df_val = train_test_split(df[1:],test_size=0.2,train_size=0.8,random_state=2)
df_train.shape, df_val.shape, df.shape


# %%
y_train = df_train[targets]
X_train = df_train[features]

y_val = df_val[targets]
X_val = df_val[features]


# %%
# dacon에서 제공하는 평가 지표 함수. 낮을 수록 좋은 값.
def kaeri_metric(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)

### E1과 E2는 아래에 정의됨 ###

def E1(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)

def E2(y_true, y_pred):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]
            
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))


# %%
# Multi output regressor를 이용해서 XGB regressor를 사용
n_estimators = 1000
learning_rate = 0.001
gamma = 0
subsample = 0.75
max_depth = 8

model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate, 
        gamma=gamma, 
        subsample=subsample,
        max_depth=max_depth))

model.fit(X_train,y_train)

y_train_pred_xgb = model.predict(X_train)
y_val_pred_xgb = model.predict(X_val)

# 스코어 계산
xgb_kaeri_score_train = kaeri_metric(y_train,y_train_pred_xgb)
xgb_mae_train = mean_absolute_error(y_train,y_train_pred_xgb)
xgb_mse_train = mean_squared_error(y_train,y_train_pred_xgb)
xgb_kaeri_score_val = kaeri_metric(y_val,y_val_pred_xgb)
xgb_mae_val = mean_absolute_error(y_val,y_val_pred_xgb)
xgb_mse_val = mean_squared_error(y_val,y_val_pred_xgb)

xgb_score = pd.DataFrame(['xgb',xgb_kaeri_score_train,xgb_mae_train, xgb_mse_train, xgb_kaeri_score_val,xgb_mae_val, xgb_mse_val]).T
xgb_score.columns=['idx','kaeri_score_train', 'mae_train', 'mse_train', 'kaeri_score_val', 'mae_val', 'mse_val']
print("N:",n_estimators,"/ L-rate:",learning_rate,"/ gamma:",gamma,"/ subsample:",subsample,"/ max depth:",max_depth)
xgb_score
# %%
# 전체 데이터로 다시 한번 학습
X_data = df[features]
y_data = df[targets]
# %%
X_data
# %%
model.fit(X_data,y_data)
# %%
y_pred = model.predict(X_data)
kaeri_metric(y_data,y_pred)
# %%
test_features = pd.read_csv('D:/Data/KAERI_dataset/test_features.csv')
test_data = feature_eng_df(test_features)
test_fft = fourier_feature(test_features,cutoff=80)
test_pca = pca_fft.transform(test_fft.iloc[:,1:])
test_data = pd.concat([test_data,pd.DataFrame(test_pca)],axis=1)
test_data = test_data[features]
test_data.head()
# %%
y_test_pred = model.predict(test_data)
# %%
submit = pd.read_csv('D:/Data/KAERI_dataset/sample_submission.csv')
for i in range(4):
	submit.iloc[:,i+1] = y_test_pred[:,i]
submit.head()
# %%
submit.to_csv('D:/Data/KAERI_dataset/submission_xgb_fourier_pca.csv', index = False)
# public : 
# private : 
print("Done!")

# %% [markdown]
# #### -- shap 결론
# x에는 R14가, y에는 R12가 가장 큰 영향력을 끼친다.
# 
# 이를 통해 유추해볼 때, x축을 기준으로 좌/우로 나누었을 때는 (S2,S3)/(S1,S4)
# 
# y축을 기준으로 위/아래를 나누었을 때는 (S1,S2)/(S3,S4)가 위치해 있다고 볼 수 있다.
# 
# 
# | S2 | S1 |
# 
# -ㅡ-ㅡ-ㅡ-
# 
# | S3 | S4 |
# 
# ...와 같은 형태
# 
# x,y 좌표에는 주로 시간관련 변수가
# v에는 주로 신호의 세기 관련 변수가 영향을 주었다.
# 
# 각각의 예측을 평가해봤을 때, 어느 것이 부족한지에 따라 추가해야할 변수로 시간을 넣을지 신호의 세기를 넣을지 결정 할 수 있을 것으로 보인다.
# 
# m의 경우에는 두가지 모두가 중요했고, 가장 복잡한 형태를 보여주었다.
# 
# 다른 세가지 변수를 최대한 먼저 최적화 시킨 후에, 질량을 고려하는 것이 좋을 것으로 보인다
# 

# %%
# model_rf = RandomForestRegressor(max_depth=15,min_samples_split=2,min_samples_leaf=1,criterion='mae',max_features='sqrt',n_estimators=1000,oob_score=True,random_state=2,n_jobs=3)
# # %%
# model_rf.fit(X_train,y_train)
# # %%
# y_pred_val_rf = model_rf.predict(X_val)
# kaeri_metric(y_val,y_pred_val_rf)
# # %%
# y_pred_train_rf = model_rf.predict(X_train)
# kaeri_metric(y_train,y_pred_train_rf)
# # %%
# plt.figure(figsize=(10,len(model_rf.feature_importances_)))
# pd.Series(model_rf.feature_importances_,X_train.columns).sort_values().plot.barh()
# # %%
# model_rf2 = RandomForestRegressor(max_depth=None,min_samples_split=2,min_samples_leaf=1,criterion='mae',max_features='sqrt',n_estimators=1000,oob_score=True,random_state=2,n_jobs=3)
# # %%
# model_rf2.fit(X_train,y_train)
# # %%
# y_pred_val_rf2 = model_rf2.predict(X_val)
# kaeri_metric(y_val,y_pred_val_rf)
# # %%
# y_pred_train_rf2 = model_rf2.predict(X_train)
# kaeri_metric(y_train,y_pred_train_rf)
# # %%
# plt.figure(figsize=(10,len(model_rf2.feature_importances_)//2))
# pd.Series(model_rf2.feature_importances_,X_train.columns).sort_values().plot.barh()
# # %%
# test_features = pd.read_csv('D:/Data/KAERI_dataset/test_features.csv')
# test_data = feature_eng_df(test_features)[features]
# test_data.head()
# # %%
# y_test_pred = model_rf2.predict(test_data)
# # %%
# submit = pd.read_csv('D:/Data/KAERI_dataset/sample_submission.csv')
# for i in range(4):
# 	submit.iloc[:,i+1] = y_test_pred[:,i]
# submit.head()
# # %%
# submit.to_csv('D:/Data/KAERI_dataset/submission_rf_fourier_80.csv', index = False)
# # public : 
# # private : 
# print("Done!")
# %%
# model_x = XGBRegressor(n_estimators=200, learning_rate=0.1, gamma=0, subsample=0.75,max_depth=8)

# model_x.fit(X_data,y_data['X'])
# # %%
# y_pred_x = model_x.predict(X_data)
# mean_absolute_error(y_data['X'],y_pred_x)
# # %%
# test_features = pd.read_csv('D:/Data/KAERI_dataset/test_features.csv')
# test_data = feature_eng_df(test_features)[features]
# test_data.head()
# # %%
# y_test_pred_x = model_x.predict(test_data)
# # %%
# submit = pd.read_csv('D:/Data/KAERI_dataset/sample_submission.csv')
# submit.iloc[:,1] = y_test_pred_x
# submit.head()
# # %%
# submit.to_csv('D:/Data/KAERI_dataset/submission_xgb_fourier_40_x.csv', index = False)
# # public : 
# # private : 
# print("Done!")
# %%
# 혹시나, 따로 따로 xgb를 하는 경우랑, 한꺼번에 multioutput을 쓰는게 다를지
# 확인해보니 동일! 그걸 걱정할 필요는 없겠다
# 단지, 각각의 변수에 대해서는 서로 다른 모델이 최적화 될수도 있겠다
# 각각을 따로봐서 효과를 체크해보면서 최적치를 찾아보자