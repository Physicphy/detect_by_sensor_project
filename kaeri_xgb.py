# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score


# %%
# 원전 내부의 충돌체 정보를 네개의 센서 정보만으로 특정해내기
# 데이터 출처 : https://dacon.io/competitions/official/235614/overview/description/
train_features = pd.read_csv('D:/Data/KAERI_dataset/train_features.csv')
train_target = pd.read_csv('D:/Data/KAERI_dataset/train_target.csv')

# %%
train_features.shape, train_target.shape

# %%
train_features.head() #학습 데이터의 feature 확인
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
peak_df = pick_peak(train_features)
peak_df.head()
# %%
train_features.isnull().sum() # 결측치 확인.

# %%
train_target.head() #학습 데이터의 target 확인

# %%
train_target.isnull().sum() # 결측치 확인

# %% [markdown]
# #### -- Data Preprocessing
# %%
# 확인 해보니, S3가 먼저 신호를 받은 경우가 한번도 없는 것으로 나온다
# 그래서 S3가 항상 고려되지 않은채로 분류 된 것...
# 추후에도 S3값이 고려 될 수 있게, 축을 새로 잡아주기
def reset_axis(data0,new_axis=('A','B','C','D')):
	data = data0.copy()
	# A=(S1+S2+S3+S4)/4, B=(S1+S2-S3-S4)/4, C=(S1-S2-S3+S4)/4, D=(S1-S2+S3-S4)/4
	data[new_axis[0]] = (data['S1']+data['S2']+data['S3']+data['S4'])/4
	data[new_axis[1]] = (data['S1']+data['S2']-data['S3']-data['S4'])/4
	data[new_axis[2]] = (data['S1']-data['S2']-data['S3']+data['S4'])/4
	data[new_axis[3]] = (data['S1']-data['S2']+data['S3']-data['S4'])/4
	# data = data.drop(['S1','S2','S3','S4'],axis=1)
	return data
# %%
# 앞서 수행했던 데이터 전처리 및 feature engineering을 수행해주는 함수
def feature_eng_df(data):
	cond = (data['S1'] != 0) | (data['S2'] != 0) | (data['S3'] != 0) | (data['S4'] != 0)

	data_active = data[cond]

	data_active = data_active.drop_duplicates(['id'],keep='first')

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
		cond = (data[s] != 0)
		active_time = data[cond].drop_duplicates(['id'],keep='first')[['id','Time']]
		active_time['Time'] = (active_time['Time']*10**6).astype('int')
		active_time.columns = ['id','active_time_'+s]
		data_active = data_active.merge(active_time,on='id')

	data_active['R12'] = (data_active['active_time_S1']+data_active['active_time_S2'])/(data_active['active_time_S3']+data_active['active_time_S4'])
	data_active['R13'] = (data_active['active_time_S1']+data_active['active_time_S3'])/(data_active['active_time_S2']+data_active['active_time_S4'])
	data_active['R14'] = (data_active['active_time_S1']+data_active['active_time_S4'])/(data_active['active_time_S2']+data_active['active_time_S3'])

	data_active['RMS_S'] = (data_active['S1']**2+data_active['S2']**2+data_active['S3']**2+data_active['S4']**2)**0.5
	data_active['RMS_gap'] = (data_active['gap_S1']**2+data_active['gap_S2']**2+data_active['gap_S3']**2+data_active['gap_S4']**2)**0.5
	data_active['RMS_time'] = (data_active['active_time_S1']**2+data_active['active_time_S2']**2+data_active['active_time_S3']**2+data_active['active_time_S4']**2)**0.5

	new_axis = ('A','B','C','D')
	data_active = reset_axis(data_active,new_axis=new_axis)
	# data = reset_axis(data,new_axis=new_axis)

	for n in data_active.index:
		for s in ['S1','S2','S3','S4']: # new_axis:
			data_active.loc[n,'delta_'+s] = data.loc[n+1,s]-data.loc[n,s]

	data_active['RMS_delta'] = (data_active['delta_S1']**2+data_active['delta_S2']**2+data_active['delta_S3']**2+data_active['delta_S4']**2)**0.5

	for s in ['S1','S2','S3','S4']:
		data_active['abs_'+s] = abs(data_active[s])
		
	return data_active

# %%
df_features = feature_eng_df(train_features)
df_features.head()
# %%
df_features[['active_time_S1','active_time_S2','active_time_S3','active_time_S4']]

# %%
# target과 feature이름들을 모아서 리스트로 만들어 두자
targets = list(train_target.columns)[1:]
features = list(df_features.columns)[1:]


# %%
df = df_features.merge(train_target,on='id')
df.head()


# %%
# 데이터를 학습용, 검증용으로 분리
# 이렇게보니 사이즈가 확 줄었다.
df_train, df_val = train_test_split(df[1:],test_size=0.2,train_size=0.8,random_state=2)
df_train.shape, df_val.shape, df.shape


# %%
y_train = df_train[targets]
X_train = df_train.drop(['id']+targets,axis=1)

y_val = df_val[targets]
X_val = df_val.drop(['id']+targets,axis=1)


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

# %% [markdown]
# #### -- 좀 더 들여다보기

# %%
# 같은 인덱스(같은 충돌체)의 센서 정보를 시각화 해서 비교
# 이 파장들을 푸리에 트랜스폼 시켜서, 근사적으로 amplitude가 큰 경우만 몇가지 추려내 feature로 쓸 수 있을까?
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.lineplot(data=train_features.loc[train_features['id']==644],x='Time',y='S1')

plt.subplot(2,2,2)
sns.lineplot(data=train_features.loc[train_features['id']==644],x='Time',y='S2')

plt.subplot(2,2,3)
sns.lineplot(data=train_features.loc[train_features['id']==644],x='Time',y='S3')

plt.subplot(2,2,4)
sns.lineplot(data=train_features.loc[train_features['id']==644],x='Time',y='S4')


# %%
# numpy에서 Fast Fourier Transform을 사용, 원하는 만큼만 잘라낼 수 있게 함수 설정
def fourier_trsf(data,sensor,id=0,cutoff=20):
	cond_id = data['id']==id
	wave = data.loc[cond_id,sensor]
	time = data.loc[cond_id,'Time']
	fft_wave = np.fft.fft(wave)
	freq = np.fft.fftfreq(time.shape[-1])
	cw = np.copy(fft_wave)
	cw[cutoff:-cutoff]=0
	fft_wave_2 = np.real(np.fft.ifft(cw))
	
	return {"fft":fft_wave, "freq":freq, "fft_cutoff":fft_wave_2, "time":time, "wave":wave}


# %%
# cutoff 15로, 진동수 15개만을 남겨도 꽤 유사한 모습이 된다
# 물론 이렇게 해도, 각 센서별로 변수가 진동수, 진폭, 위상차로 3x15=45가지씩 늘어난다. 총 180개가 추가되는 셈
# 가장 진폭이 큰 진동수와 위상차만 남겨봐야 하나...?
w0 = fourier_trsf(data=train_features,sensor='S1',id=0,cutoff=15)

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

# %% [markdown]
# #### -- Baseline Model
# 
# : X, Y, M, V 각각의 평균 값을 기준 모델로 사용

# %%
# 타겟의 분포를 한번 살펴보자
# 대부분 고르게 퍼져있는 모습을 보여준다
# 기준 모델은 평균으로 잡아서 사용하는 것이 적절해보인다
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.histplot(train_target['X'])
plt.axvline(train_target['X'].mean(),color='r')
plt.subplot(2,2,2)
sns.histplot(train_target['Y'])
plt.axvline(train_target['Y'].mean(),color='r')
plt.subplot(2,2,3)
sns.histplot(train_target['M'])
plt.axvline(train_target['M'].mean(),color='r')
plt.subplot(2,2,4)
sns.histplot(train_target['V'])
plt.axvline(train_target['V'].mean(),color='r')


# %%
# 평균을 기준모델로 잡았다
y_train_base_pred = [y_train.mean()]*len(y_train)
y_val_base_pred = [y_train.mean()]*len(y_val)


# %%
# train data에서 기준모델의 성능 (MAE, MSE)
base_train_mae = mean_absolute_error(y_train,y_train_base_pred)
base_train_mse = mean_squared_error(y_train,y_train_base_pred)
base_train_kaeri = kaeri_metric(y_train,y_train_base_pred)
base_train_mae, base_train_mse, base_train_kaeri


# %%
# validation data에서 기준모델의 성능 (MAE, MSE)
base_val_mae = mean_absolute_error(y_val,y_val_base_pred)
base_val_mse = mean_squared_error(y_val,y_val_base_pred)
base_val_kaeri = kaeri_metric(y_val,y_val_base_pred)
base_val_mae, base_val_mse, base_val_kaeri

# %% [markdown]
# #### -- RandomForest Model

# %%
# regressor와 데이터를 받아서 학습하고, MAE와 MSE, feature importance를 시각화 시켜준다
def make_model(regressor,feature_list,data_train,target_train,data_val,target_val):
	train_X = data_train[feature_list]
	val_X = data_val[feature_list]
	train_y = target_train
	val_y = target_val

	regressor.fit(train_X,train_y)

	train_y_pred = regressor.predict(train_X)
	val_y_pred = regressor.predict(val_X)
	kaeri_score_train = kaeri_metric(train_y,train_y_pred)
	kaeri_score_val = kaeri_metric(val_y,val_y_pred)
	mae_train = mean_absolute_error(train_y,train_y_pred)
	mae_val = mean_absolute_error(val_y,val_y_pred)
	mse_train = mean_squared_error(train_y,train_y_pred)
	mse_val = mean_squared_error(val_y,val_y_pred)

	print("train score \n","kaeri_metric:",kaeri_score_train, "mae:",mae_train,"mse:",mse_train)
	print("val score \n","kaeri_metric:",kaeri_score_val, "mae:",mae_val,"mse:",mse_val)

	plt.figure(figsize=(10,len(regressor.feature_importances_)))
	pd.Series(regressor.feature_importances_,train_X.columns).sort_values().plot.barh()

	return kaeri_score_train, mae_train, mse_train, kaeri_score_val, mae_val, mse_val 


# %%
# RandomForestRegressor를 만든다
model = RandomForestRegressor(criterion='mse',max_features='sqrt',n_estimators=1000,oob_score=True,random_state=2)


# %%
# 주어진 Feature를 넣고서 학습. 결과 확인
model_score = make_model(model,features,X_train,y_train,X_val,y_val)


# %%
# 세가지 score 모두 확실히 RF로 학습한 모델이 훨씬 작은 값을 보여준다
base_vs_rf = pd.DataFrame([['RF']+list(model_score),['Base',base_train_kaeri,base_train_mae, base_train_mse, base_val_kaeri, base_val_mae, base_val_mse]],columns=['idx','kaeri_score_train', 'mae_train', 'mse_train', 'kaeri_score_val', 'mae_val', 'mse_val'])
base_vs_rf

# %% [markdown]
# #### -- Xgboost model

# %%
from xgboost import XGBRegressor
import xgboost
from sklearn.multioutput import MultiOutputRegressor


# %%
# XGB모델 설정 타겟이 네가지라, multioutput regressor를 사용.
multi_xgb = MultiOutputRegressor(XGBRegressor(n_estimators=200, learning_rate=0.1, gamma=0, subsample=0.75,max_depth=8)) #,colsample_bytree=1)
drop_list =['S3','abs_S3','delta_S1','abs_S2','RMS_delta','delta_S3','abs_S1']
# %%
# XGB에 사용하기 위해서, array 형태로 전부 변환
dX_train = X_train.drop(drop_list,axis=1).to_numpy()
dX_val = X_val.drop(drop_list,axis=1).to_numpy()
dy_train = y_train.to_numpy()
dy_val = y_val.to_numpy()


# %%
multi_xgb.fit(dX_train,dy_train)


# %%
y_train_pred_xgb = multi_xgb.predict(dX_train)


# %%
y_val_pred_xgb = multi_xgb.predict(dX_val)


# %%
# 스코어 계산
xgb_kaeri_score_train = kaeri_metric(y_train,y_train_pred_xgb)
xgb_mae_train = mean_absolute_error(y_train,y_train_pred_xgb)
xgb_mse_train = mean_squared_error(y_train,y_train_pred_xgb)
xgb_kaeri_score_val = kaeri_metric(y_val,y_val_pred_xgb)
xgb_mae_val = mean_absolute_error(y_val,y_val_pred_xgb)
xgb_mse_val = mean_squared_error(y_val,y_val_pred_xgb)
xgb_kaeri_score_train,xgb_mae_train, xgb_mse_train, xgb_kaeri_score_val,xgb_mae_val, xgb_mse_val


# %%
xgb_score = pd.DataFrame(['xgb',xgb_kaeri_score_train,xgb_mae_train, xgb_mse_train, xgb_kaeri_score_val,xgb_mae_val, xgb_mse_val]).T
xgb_score.columns=['idx','kaeri_score_train', 'mae_train', 'mse_train', 'kaeri_score_val', 'mae_val', 'mse_val']
xgb_score


# %%
# 베이스라인, RF, XGB의 결과값 비교
# 전반적으로, xgboost를 쓴 것이, RF를 사용한 것 보다, 성능이 향상 되었다.
xgb_vs_rf = pd.concat([xgb_score,base_vs_rf])
xgb_vs_rf

# %% [markdown]
# #### -- Permuataion Importances

# %%
import eli5
from eli5.sklearn import PermutationImportance


# %%
# permuter 생성 후, dX_val, y_val를 넣고 각 feature를 permutation해가면서 확인
permuter = PermutationImportance(
		multi_xgb,
		scoring='neg_mean_absolute_error', 
		n_iter=5, 
		random_state=2
)
permuter.fit(dX_val, y_val);

# %%
# R12, R13, R14가 가장 큰 importance를 보여준다.
# delta_S는 S1하나만 빼고는 0이다
# S3도 0이 나왔다
# 특정 센서가 더 중요할 이유가 없다고 판단하면, 현재 타겟 X, Y의 분포에서 약간 차이나는 부분에서 나온 오차로 같은 종류의 feature에서도 특정 센서만 importance가 다른 것일 거라 추측된다
# delta_S의 경우는 하나만 제외하고는 0이 나왔으니, 제외 해도 괜찮을 것 같다. 특별히 delta_S1이 큰 값을 갖는 것도 아니기 때문에 과적합의 요인이 될 수 있을 것으로 보인다.
pd.Series(permuter.feature_importances_, [col for col in features if col not in drop_list]).sort_values(ascending=False)
# %%
test_features = pd.read_csv('D:/Data/KAERI_dataset/test_features.csv')
test_data = feature_eng_df(test_features)
# %%
# feature_name =  ['Time','S1','S2','S3','S4','delta_S1','delta_S2','delta_S3','delta_S4', 'gap_S1', 'gap_S2', 'gap_S3', 'gap_S4', 'active_time_S1', 'active_time_S2', 'active_time_S3', 'active_time_S4', 'R12', 'R13', 'R14', 'RMS_S', 'RMS_delta', 'RMS_gap', 'RMS_time']

dX_test = test_data[features].drop(drop_list,axis=1).to_numpy()
# %%
y_test_pred = multi_xgb.predict(dX_test)
# %%
submit = pd.read_csv('D:/Data/KAERI_dataset/sample_submission.csv')
for i in range(4):
	submit.iloc[:,i+1] = y_test_pred[:,i]
submit.head()
# %%
submit.to_csv('D:/Data/KAERI_dataset/submission_xgb_new.csv', index = False)
# public : 
# private : 

# %%
y_test_pred = model.predict(test_data[features])
# %%
submit = pd.read_csv('D:/Data/KAERI_dataset/sample_submission.csv')
for i in range(4):
	submit.iloc[:,i+1] = y_test_pred[:,i]
submit.head()
# %%
submit.to_csv('D:/Data/KAERI_dataset/submission_rf_mse_auto.csv', index = False)
# public : 
# private : 
print("Done!")
# %%
peak_df[peak_df['id']==0].index[-1]
# %%
df[df['S3']!=0]
# 이걸 이제야 봤네...
# 처음 온 데이터로만 하니까 S3가 없어져 버렸다
# 이러면 곤란하지... 

# %%
test_data[test_data['S3']!=0]
# 테스트 데이터 에도 없네 이거 함정일수도?
# S3이 실제 private test data에는 있을 수 있다
# S3이 학습할 때 무시되지 않아야 한다
# 서로 조합시켜서, 새로 축을 만들게 해보자
# 예를 들면...
# (S1+S2), (S2-S1), (S3+S4), (S4-S3)
# 아... 이렇게하니까, S3+S4랑 S4-S3 둘이 똑같아 져서 둘중에 하나만 쓰네
# A=(S1+S2+S3+S4)/4, B=(S1+S2-S3-S4)/4, C=(S1-S2-S3+S4)/4, D=(S1-S2+S3-S4)/4
# 이러면, S1 = A+B+C+D, S2 = A+B-C-D, S3=A-B-C+D, S4=A-B+C-D
# 이렇게 바꾸고 나서, delta도 A,B,C,D 값으로 해야 할 것 같다
# %%
# 한번 시도 해보니... 효과는 영 좋지 않네
# 음... 어렵다, 단순히 이렇게 뒤섞다보니 오히려 정보가 왜곡되어서 사라졌나보다
# 
# 추가로 peak time정도도 넣어 볼 수 있을 것 같다

# 추가로, 단순 무식하게...! 모델을 따로 네개 만들어서
# 타겟별로 네가지 학습을 따로 시켜서 결과를 얻을 수도 있을 것 같다
# 각각이 따로 최적화가 될지도 모르니... 한번 해봐?

# 추가로, xgb가 일단 좋아보이기는 하는데... 하이퍼파라미터 조정을 좀 해보자
# 조정할때, cv로 좀더 엄밀하게 확인이 가능하게 세팅해서 해보자