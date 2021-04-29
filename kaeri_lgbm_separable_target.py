# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
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
def find_opt_cutoff(target_col,target_params,train_data,target_data,min_c=10,max_c=100,step_c=5,cv=5):
    data=train_data.copy()
    data_features = feature_eng_df(data,cutoff=max_c)
    data_tot = data_features.merge(target_data,on='id')
    cv_dict = {}

    for nn in range(min_c,max_c+step_c,step_c):       
        feature_list = list(data_features.columns)[1:24]   
        for s in ['S1','S2','S3','S4']:
            feature_list = feature_list+[s+'_f'+str(n) for n in range(nn)]
        cv_result = cv_check(data_tot,target_col,feature_list,target_params,cv=cv)
        cv_dict[nn]=cv_result

    return cv_dict
# %%
def cv_multi_model(data,x_feature,y_feature,m_feature,v_feature,x_params,y_params,m_params,v_params):
    kf = KFold(n_splits=5)
    mae_score_list = []
    rmse_score_list = []
    for train_index, test_index in kf.split(data):
        train_d = data.loc[train_index]
        test_d = data.loc[test_index]
        test_pred = multi_model(train_d,test_d,test_d[targets],x_feature,y_feature,m_feature,v_feature,x_params,y_params,m_params,v_params)
        mae_score = mean_absolute_error(test_d[targets],test_pred)
        rmse_score = mean_squared_error(test_d[targets],test_pred)**0.5
        mae_score_list.append(mae_score)
        rmse_score_list.append(rmse_score)
    return {'mae_mean':np.mean(mae_score_list),'rmse_mean':np.mean(rmse_score_list)}
# %%
select_objective = 'regression'
boosting_type = 'gbdt' # 'dart'
learning_rate = [0.015,0.03,0.015,0.02] # [0.21,0.36,0.2,0.25]

x_params = {'learning_rate':learning_rate[0],'n_estimators':1000,
    'boosting_type':boosting_type,'random_state':2,'objective':select_objective
}

y_params = {'learning_rate':learning_rate[1],'n_estimators':1000,
    'boosting_type':boosting_type,'random_state':2,'objective':select_objective
}

m_params = {'learning_rate':learning_rate[2],'n_estimators':1000,
    'boosting_type':boosting_type,'random_state':2,'objective':select_objective
}

v_params = {'learning_rate':learning_rate[3],'n_estimators':1000,
    'boosting_type':boosting_type,'random_state':2,'objective':select_objective
}
# %%
%%time
cutoff = 80
df_features = feature_eng_df(train_features,cutoff=cutoff)

targets = list(train_target.columns)[1:]
features = list(df_features.columns)[1:]

df = df_features.merge(train_target,on='id')

# 데이터를 학습용, 검증용으로 분리
df_train, df_val = train_test_split(df[1:],test_size=0.2,train_size=0.8,random_state=2)
df_train.shape, df_val.shape, df.shape

simple_features = np.array(df.loc[:,'Time':'RMS_time'].columns)

print(len(simple_features))

cv_multi_model(
    df,x_feature=simple_features,y_feature=simple_features,
    m_feature=features,v_feature=features,
    x_params=x_params,y_params=y_params,m_params=m_params,v_params=v_params
    )
# fm_gap 추가, cutoff 80 - 'mae_mean': 1.4411159812090908, 'rmse_mean': 4.234911490857837
# fm_gap, fm_delta 추가, cutoff 80 - 'mae_mean': 1.442362658623638, 'rmse_mean': 4.215226553328656
# fm_gap, fm_delta를 추가하기 전 점수
# cutoff 40 - 'mae_mean': 1.4663333369856257, 'rmse_mean': 4.244145073672463
# cutoff 50 - 'mae_mean': 1.412775858523892, 'rmse_mean': 4.156136394483022
# cutoff 60 - 'mae_mean': 1.4405273952757174, 'rmse_mean': 4.216893853079586
# cutoff 70 - 'mae_mean': 1.4382540486185822, 'rmse_mean': 4.217362530264165
# cutoff 80 - 'mae_mean': 1.4451553593726918, 'rmse_mean': 4.258674321015219
# cutoff 90 - 'mae_mean': 1.4546257965245264, 'rmse_mean': 4.269507878957171
# cutoff 100 - 'mae_mean': 1.4504806434451525, 'rmse_mean': 4.260296924640494
# %%
%%time
pred_val = multi_model(
    df_train,df_val,df_val[targets],x_feature=simple_features,y_feature=simple_features,
    m_feature=features,v_feature=features,
    x_params=x_params,y_params=y_params,m_params=m_params,v_params=v_params
    )

mae_score = mean_absolute_error(df_val[targets],pred_val)
rmse_score = mean_squared_error(df_val[targets],pred_val)**0.5

print('MAE:',mae_score,'RMSE:',rmse_score)
# fm_gap, fm_intv를 추가하기 전 점수
# cutoff 40 -  MAE: 1.4498378114314339 RMSE: 4.374771597610147
# cutoff 50 -  MAE: 1.3714717833129992 RMSE: 4.120107771648472
# cutoff 50,65 - MAE: 1.3715553820262516 RMSE: 4.120108207367386
# cutoff 50,70 - MAE: 1.371536916194218 RMSE: 4.120107351517129
# cutoff 60 -  MAE: 1.397167810555709 RMSE: 4.157952315647328
# cutoff 70 -  MAE: 1.3997160130449866 RMSE: 4.166293611531264
# cutoff 80 -  MAE: 1.418707318235794 RMSE: 4.2061886118650085
# cutoff 90 -  MAE: 1.4133144669178783 RMSE: 4.209081412744488
# cutoff 100 -  MAE: 1.4257778761440079 RMSE: 4.272305099414357
# %%
%%time
test_features = pd.read_csv('D:/Data/KAERI_dataset/test_features.csv')
test_data = feature_eng_df(test_features,cutoff=cutoff)[features]

submit_csv = pd.read_csv('D:/Data/KAERI_dataset/sample_submission.csv')
submit = multi_model(
    df,test_data,submit_csv,
    x_feature=simple_features,y_feature=simple_features,
    m_feature=features,v_feature=features,
    x_params=x_params,y_params=y_params,m_params=m_params,v_params=v_params
    )
submit
# %%
# submit.to_csv('D:/Data/KAERI_dataset/submission_lgbm_dart_separable_cutoff_'+str(cutoff)+'.csv', index = False)
# print("Done!")
# cutoff 50일때, 약간 애매! 0.082789476, 0.0876419825
# cutoff 80일때, 기록 갱신! 0.0836479346, 0.0872165547
# cutoff 100일때, 기록 하락 0.0847085837, 0.0900441972	


# %%
m_features = cutoff_features(df,80)
v_features = cutoff_features(df,80)
# %%
%%time
submit2 = multi_model(
    df,test_data,submit_csv,
    x_feature=simple_features,y_feature=simple_features,
    m_feature=m_features,v_feature=v_features,
    x_params=x_params,y_params=y_params,m_params=m_params,v_params=v_params
    )
submit2
# %%
%%time
pred_val = multi_model(
    df_train,df_val,df_val[targets],x_feature=simple_features,y_feature=simple_features,
    m_feature=m_features,v_feature=v_features,
    x_params=x_params,y_params=y_params,m_params=m_params,v_params=v_params
    )

mae_score = mean_absolute_error(df_val[targets],pred_val)
rmse_score = mean_squared_error(df_val[targets],pred_val)**0.5

print('MAE:',mae_score,'RMSE:',rmse_score)
# MAE: 1.3715553820262516 RMSE: 4.120108207367386
# cutoff 80 : MAE: 1.4542291404146797 RMSE: 4.3559285199636575

x_mae_score = mean_absolute_error(df_val['X'],pred_val['X'])
x_rmse_score = mean_squared_error(df_val['X'],pred_val['X'])**0.5

print('MAE:',x_mae_score,'RMSE:',x_rmse_score)
# MAE: 1.001612061153801 RMSE: 5.792512362668784

y_mae_score = mean_absolute_error(df_val['Y'],pred_val['Y'])
y_rmse_score = mean_squared_error(df_val['Y'],pred_val['Y'])**0.5

print('MAE:',y_mae_score,'RMSE:',y_rmse_score)
# MAE: 0.42663566680104825 RMSE: 1.9661926421424363

m_mae_score = mean_absolute_error(df_val['M'],pred_val['M'])
m_rmse_score = mean_squared_error(df_val['M'],pred_val['M'])**0.5

print('MAE:',m_mae_score,'RMSE:',m_rmse_score)
# cutoff 80 : 4.3749755535011765 RMSE: 6.202961299832963

v_rmse_score = mean_squared_error(df_val['V'],pred_val['V'])**0.5
v_mae_score = mean_absolute_error(df_val['V'],pred_val['V'])

print('MAE:',v_mae_score,'RMSE:',v_rmse_score)
# cutoff 80 : 0.013693280202692991 RMSE: 0.024722848135062835
# %%
# 결론 : x,y는 simple_feature를 사용, m,v는 cutoff=80을 사용
# 열심히 잘라보고 했으나 큰 차이는 없네!
# 여기에 추가 feature로 first_max_gap을 넣자

# %%
first_max_gap_interval(train_features)