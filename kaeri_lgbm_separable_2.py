# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from lightgbm import train
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
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
# dacon에서 제공하는 평가 지표 함수. 낮을 수록 좋은 값.
def kaeri_metric(y_true, y_pred):    
    return 0.5 * E1(y_true, y_pred) + 0.5 * E2(y_true, y_pred)

def E1(y_true, y_pred):
    _t, _p = np.array(y_true)[:,:2], np.array(y_pred)[:,:2]
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)

def E2(y_true, y_pred):
    _t, _p = np.array(y_true)[:,2:], np.array(y_pred)[:,2:]           
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))
# %%
def cv_check(data,target,features,params,cv=5):
    model = lgb.LGBMRegressor(**params)

    data_feature = data[features] 
    data_target = data[target]

    cv_score = cross_validate(model,data_feature,data_target,cv=cv,scoring=('neg_root_mean_squared_error','neg_mean_absolute_error'))

    rmse = -cv_score['test_neg_root_mean_squared_error'] 
    mae = -cv_score['test_neg_mean_absolute_error']

    return {'mean_rmse':rmse.mean(),'std_rmse':rmse.std(),'mean_mae':mae.mean(),'std_mae':mae.std()}

# %%
def make_model(data,test_data,target,features,params):
    model = lgb.LGBMRegressor(**params)

    data_feature = data[features] 
    data_target = data[target]

    test_data_feature = test_data[features] 

    model.fit(data_feature,data_target)
    test_pred = model.predict(test_data_feature)

    return test_pred

# %%
def multi_model(data,test_data,submit0,x_feature,y_feature,m_feature,v_feature,x_params,y_params,m_params,v_params):
    pred_x = make_model(data,test_data,'X',x_feature,x_params)
    pred_y = make_model(data,test_data,'Y',y_feature,y_params)
    pred_m = make_model(data,test_data,'M',m_feature,m_params)
    pred_v = make_model(data,test_data,'V',v_feature,v_params)

    submit = submit0.copy()

    submit['X'] = pred_x
    submit['Y'] = pred_y
    submit['M'] = pred_m
    submit['V'] = pred_v

    return submit
# %%
def cutoff_features(data,cutoff):
    feature_list = list(data.columns)[1:24]
    if cutoff > 0:
        for s in ['S1','S2','S3','S4']:
            feature_list = feature_list+[s+'_f'+str(n) for n in range(cutoff)]
    return feature_list    

# %%
x_params = {
    'learning_rate':0.21,
    'n_estimators':1000,
    'boosting_type':'dart',
    'random_state':2
}

y_params = {
    'learning_rate':0.36,
    'n_estimators':1000,
    'boosting_type':'dart',
    'random_state':2
}

m_params = {
    'learning_rate':0.2,
    'n_estimators':1000,
    'boosting_type':'dart',
    'random_state':2
}

v_params = {
    'learning_rate':0.25,
    'n_estimators':1000,
    'boosting_type':'dart',
    'random_state':2
}
# %%
df_features = feature_eng_df(train_features,cutoff=80)

targets = list(train_target.columns)[1:]
features = list(df_features.columns)[1:]

df = df_features.merge(train_target,on='id')

# 데이터를 학습용, 검증용으로 분리
df_train, df_val = train_test_split(df[1:],test_size=0.2,train_size=0.8,random_state=2)
df_train.shape, df_val.shape, df.shape

simple_features = np.array(df.columns)[1:24]

# %%
cv_check(df_train,'M',features,m_params,cv=5)
# df_train을 train + val로 생각해서 cv에 사용
# df_val을 test로 생각해서 마지막에 최종 점검에 사용
# %%
def find_opt_cutoff(target_col,target_params,train_data,target_data,min_c=10,max_c=100,step_c=5,cv=5):
    data=train_data.copy()
    data_features = feature_eng_df(data,cutoff=max_c)
    data_tot = data_features.merge(target_data,on='id')
    data_train, _ = train_test_split(data_tot[1:],test_size=0.2,train_size=0.8,random_state=2)
    cv_dict = {}

    for nn in range(min_c,max_c+step_c,step_c):       
        feature_list = list(data_features.columns)[1:24]   
        for s in ['S1','S2','S3','S4']:
            feature_list = feature_list+[s+'_f'+str(n) for n in range(nn)]
        cv_result = cv_check(data_train,target_col,feature_list,target_params,cv=cv)
        cv_dict[nn]=cv_result

    return cv_dict
# %%
# M : 5~100까지 fft cutoff 개수를 바꿔가며 확인 해보니, 50이 제일 결과가 좋다
%time

cv_m_dict = find_opt_cutoff('M',m_params,train_features,train_target,min_c=5,max_c=100,step_c=5,cv=5)
cv_m_dict

# %%
# V : 5~100까지 fft cutoff 개수를 바꿔가며 확인 해보니, 65가 제일 결과가 좋다
%time
cv_v_dict = find_opt_cutoff('V',v_params,train_features,train_target,min_c=5,max_c=100,step_c=5,cv=5)
cv_v_dict
# %%
m_features = cutoff_features(df,50)
v_features = cutoff_features(df,65)

pred_val = multi_model(
    df_train,df_val,df_val[targets],x_feature=simple_features,y_feature=simple_features,
    m_feature=m_features,v_feature=v_features,
    x_params=x_params,y_params=y_params,m_params=m_params,v_params=v_params
    )

mae_score = mean_absolute_error(df_val[targets],pred_val)
rmse_score = mean_squared_error(df_val[targets],pred_val)**0.5

print('MAE:',mae_score,'RMSE:',rmse_score)
# MAE: 1.3715553820262516 RMSE: 4.120108207367386

# %%
m_pred = make_model(df_train,df_val,'M',m_features,m_params)

m_mae_score = mean_absolute_error(df_val['M'],m_pred)
m_rmse_score = mean_squared_error(df_val['M'],m_pred)**0.5
print('MAE:',m_mae_score,'RMSE:',m_rmse_score)
# MAE: 4.147320936615687 RMSE: 5.837224812798168
# %%
pred_val[['X','Y','V']]
# %%
pred_train = multi_model(
    df_train,df_train,df_train[targets],x_feature=simple_features,y_feature=simple_features,
    m_feature=m_features,v_feature=v_features,
    x_params=x_params,y_params=y_params,m_params=m_params,v_params=v_params
    )
pred_train[['X','Y','V']]
# %%
df_train_2 = pd.concat([df_train,pred_train[['X','Y','V']]],axis=1)
df_train_2.columns = list(df_train.columns)+['pred_X','pred_Y','pred_V']
df_val_2 = pd.concat([df_val,pred_val[['X','Y','V']]],axis=1)
df_val_2.columns = list(df_val.columns)+['pred_X','pred_Y','pred_V']
# %%
m_features_2 = m_features+['pred_X','pred_Y','pred_V']
# %%
# 음 넣는거 오히려 떨어지네
m_pred_2 = make_model(df_train_2,df_val_2,'M',m_features_2,m_params)

m_mae_score_2 = mean_absolute_error(df_val_2['M'],m_pred_2)
m_rmse_score_2 = mean_squared_error(df_val_2['M'],m_pred_2)**0.5
print('MAE:',m_mae_score_2,'RMSE:',m_rmse_score_2)
# MAE: 4.924853491008904 RMSE: 6.961037403057198
# %%
# 
m_pred_3 = make_model(df_train,df_val,'M',features,m_params)

m_mae_score_3 = mean_absolute_error(df_val['M'],m_pred_3)
m_rmse_score_3 = mean_squared_error(df_val['M'],m_pred_3)**0.5
print('MAE:',m_mae_score_3,'RMSE:',m_rmse_score_3)
# MAE: 4.3359049402771985 RMSE: 6.07783864731487
# %%
v_features
# %%
test_features = pd.read_csv('D:/Data/KAERI_dataset/test_features.csv')
test_data = feature_eng_df(test_features)[features]

# # %%
# pred_val = multi_model(
#     df_train,df_val,df_val[targets],simple_features,features,
#     x_params=x_params,y_params=y_params,m_params=m_params,v_params=v_params
#     )

# mae_score = mean_absolute_error(df_val[targets],pred_val)
# rmse_score = mean_squared_error(df_val[targets],pred_val)**0.5

# print('MAE:',mae_score,'RMSE:',rmse_score)
# # 1.418707318235794, 4.2061886118650085
# # %%
# 데이콘 제출용으로 한번 계산
submit_csv = pd.read_csv('D:/Data/KAERI_dataset/sample_submission.csv')
submit = multi_model(
    df,test_data,submit_csv,
    x_feature=simple_features,y_feature=simple_features,
    m_feature=m_features,v_feature=v_features,
    x_params=x_params,y_params=y_params,m_params=m_params,v_params=v_params
    )
submit
# %%
submit.to_csv('D:/Data/KAERI_dataset/submission_lgbm_separable_sep_feature.csv', index = False)
print("Done!")
# # 기록 갱신! 0.0836479346, 0.0872165547


# %%
# X,Y는 fft를 안쓰는게, M,V는 fft를 쓰는게 확실히 성능이 좋다
# 설명은 mae로 하기가 편하지만, 큰오차를 줄이려면 rmse로 최적화 시킬 필요가 있다
# M이 상태가 많이 안좋네, X,Y,V 예측한 것도 정말 넣고 해봐?
# X_mae는 simple_feature, n=1000, learning rate = 0.33, dart일 때 좋았다.
## X_rmse는 simple_feature, n=1000, learning rate = 0.21, dart일 때 좋았다.
# Y_mae는 simple_feature, n=1000, learning rate = 0.35, dart일 때 좋았다.
## Y_rmse는 simple_feature, n=1000, learning rate = 0.36, dart일 때 좋았다.
# M_mae는 full_feature, n=1000, learning rate = 0.2 , dart일 때 좋았다.
## M_rmse는 full_feature, n=1000, learning rate =0.2 , dart일 때 좋았다.
# V_mae는 full_feature, n=1000, learning rate =0.24 , dart일 때 좋았다.
## V_rmse는 full_feature, n=1000, learning rate =0.25 , dart일 때 좋았다.

# 좀 귀찮으니까 함수를 하나 짜두자, 변수 하나 범위 찍어주면 그 범위따라
# rmse, mae를 cv로 계산해서 그래프로 찍어주고, 최고값 찾아주는 걸로

# params = {}
# params['n_estimators'] = 1000 #1000 # 100
# params['learning_rate'] = 0.26 #0.02 #0.09
# #params['max_depth'] = 4 # 3 # 4
# #params['num_leaves'] = 15 # 7 # 15
# params['boosting_type'] = 'dart' # 'gbdt' #'goss'
# params['random_state'] = 2
# # params['bagging_fraction'] = 0.7
# #params['reg_alpha'] = 2
# #params['reg_lambda'] = 0.5
# # params['objective'] ='quantile'
# # params['alpha'] = 0.1
# print('LGBM\n',params)

# model_x = lgb.LGBMRegressor(**params)
# model_y = lgb.LGBMRegressor(**params)
# model_m = lgb.LGBMRegressor(**params)
# model_v = lgb.LGBMRegressor(**params)

# # model_x.fit(X_train,y_train_x)
# # model_y.fit(X_train,y_train_y)
# # model_m.fit(X_train,y_train_m)
# model_v.fit(X_train,y_train_v)
# y_true_a = y_train_v
# y_true_b = y_val_v
# model_a = model_v

# y_train_pred = model_a.predict(X_train)
# y_val_pred = model_a.predict(X_val)

# # cv check
# cv_score = -cross_val_score(model_a,X_train,y_true_a,cv=5,scoring='neg_root_mean_squared_error')
# print("\n rmse:",cv_score.mean(), cv_score)

# cv_score = -cross_val_score(model_a,X_train,y_true_a,cv=5,scoring='neg_mean_absolute_error')
# print("\n mae:",cv_score.mean(), cv_score)

# # 스코어 계산
# mae_train = mean_absolute_error(y_true_a,y_train_pred)
# rmse_train = mean_squared_error(y_true_a,y_train_pred)**0.5
# mae_val = mean_absolute_error(y_true_b,y_val_pred)
# rmse_val = mean_squared_error(y_true_b,y_val_pred)**0.5


# score = pd.DataFrame([round(mae_val/mae_train,2),mae_train, mae_val, rmse_train, rmse_val]).T
# score.columns=['mae_ratio(v/t)', 'mae_train','mae_val', 'rmse_train', 'rmse_val']

# score
# %%
# fft해서 넣은 feature를 전부 사용 했을 때 최선
# params = {}
# params['n_estimators'] = 1000 #1000 # 100
# params['learning_rate'] = 0.22 #0.02 #0.09
# #params['max_depth'] = 4 # 3 # 4
# #params['num_leaves'] = 15 # 7 # 15
# params['boosting_type'] = 'dart' # 'gbdt' #'goss'
# params['random_state'] = 2
# # params['bagging_fraction'] = 0.7
# #params['reg_alpha'] = 2
# #params['reg_lambda'] = 0.5
# # params['objective'] ='quantile'
# # params['alpha'] = 0.1
# print('LGBM\n',params)

# model_x = lgb.LGBMRegressor(**params)
# model_y = lgb.LGBMRegressor(**params)
# model_m = lgb.LGBMRegressor(**params)
# model_v = lgb.LGBMRegressor(**params)

# model_x.fit(X_train,y_train_x)

# y_train_pred_x = model_x.predict(X_train)
# y_val_pred_x = model_x.predict(X_val)

# # cv check
# cv_score = -cross_val_score(model_x,X_train,y_train_x,cv=5,scoring='neg_mean_absolute_error')
# print("\n",cv_score.mean(), cv_score)

# # 스코어 계산
# mae_train = mean_absolute_error(y_train_x,y_train_pred_x)
# rmse_train = mean_squared_error(y_train_x,y_train_pred_x)**0.5
# mae_val = mean_absolute_error(y_val_x,y_val_pred_x)
# rmse_val = mean_squared_error(y_val_x,y_val_pred_x)**0.5

# score = pd.DataFrame([round(mae_val/mae_train,2),mae_train, rmse_train, mae_val,rmse_val]).T
# score.columns=['mae_ratio(v/t)', 'mae_train', 'rmse_train', 'mae_val', 'rmse_val']

# score

# %%
# 전체 데이터로 다시 한번 학습
# X_data = df[features]
# y_data = df[targets]

# model.fit(X_data,y_data)
# y_pred = model.predict(X_data)
# print(kaeri_metric(y_data,y_pred))

# test_features = pd.read_csv('D:/Data/KAERI_dataset/test_features.csv')
# test_data = feature_eng_df(test_features)[features]
# y_test_pred = model.predict(test_data)
# # %%
# submit = pd.read_csv('D:/Data/KAERI_dataset/sample_submission.csv')
# for i in range(4):
# 	submit.iloc[:,i+1] = y_test_pred[:,i]
# submit.head()
# # %%
# submit.to_csv('D:/Data/KAERI_dataset/submission_lgbm.csv', index = False)
# # public : 
# # private : 
# print("Done!")
