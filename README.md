# Detect_by_sensor_data

- 사용한 데이터 : https://dacon.io/competitions/official/235614/overview/description/


### 목표
- 원자로 내부에서 움직이는 "충돌체"가 있을 경우, 그 충돌체의 "위치/질량/속도" 정보를 "진동 센서 데이터"를 이용한 머신러닝으로 예측하기.

### 사용한 방식
- LightGBM 사용
- scipy의 fftpack을 이용, 푸리에변환 하는 것으로 feature 수를 줄였다
- 각 센서별로 "첫 신호가 감지될때까지 걸린 시간"을 이용해서 위치정보를 추정하는 주요 feature를 추출
- shap과 pdpbox를 이용, 예측된 결과에 영향을 준 feature를 확인하고 의미를 분석
- RMSE와 전체 데이터의 표준편차 사이의 비율로, 얼마나 정확하게 예측했는지의 기준으로 사용 (RMSE/std)

### 결과
- RMSE/std로 X좌표, Y좌표, M(질량), V(속력)의 예측값을 평가한 결과 각각 0.021, 0.0075, 0.11, 0.063.
- 위치의 경우 작은 오차로 예측
- 질량과 속력의 경우 상대적으로 큰 오차를 보임
