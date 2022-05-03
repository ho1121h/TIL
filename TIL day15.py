import numpy as np
X = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
Y = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )

from sklearn.model_selection import train_test_split #모델 가져옴
t_x,tt_x,t_y,tt_y = train_test_split(X,Y,random_state= 43 )#트레인,테스트,트레인타겟,테스트 타겟
print(t_x.shape,tt_x.shape)#트레인과 테스트/쉐이프로 둘다 1차원 데이터임을 알 수있다,.
n_t_x = t_x.reshape(-1,1)
n_tt_x = tt_x.reshape(-1,1)
print(n_t_x.shape, n_tt_x.shape)#차원이 추가됨을 볼수있다
from sklearn .neighbors import KNeighborsRegressor
#모델 생성
knr=KNeighborsRegressor()
knr.fit(n_t_x,t_y) # 훈련메소드(트레인 데이터, 트레인 타겟)
'''
오차(차이점)란 정답 -결과
머신러닝 Y=WX -학습-> 정답 = W 입력- > y=w(학습이 되어 값을 갖고 있다.)
 xy=wx 에 x입력 하면 y값을 구할 수 있다 w는 이미 학습된 값을 가짐

'''
from sklearn.metrics import mean_absolute_error
end_tt_y = knr.predict(n_tt_x)#학습된 모델에 입력을 줌 fit으로 인해 학습됨

mea=mean_absolute_error(tt_y,end_tt_y)#평균 절대값 함수 사용
print(mea)#평균 절대값의 오차 출력


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
X = np.array([174,152,138,128,186])
Y = np.array([71,55,46,38,88])

from sklearn.model_selection import train_test_split #모델 가져옴
t_x,tt_x,t_y,tt_y = train_test_split(X,Y,random_state= 1 )
n_t_x = t_x.reshape(-1,1)
n_tt_x = tt_x.reshape(-1,1)
lr = LinearRegression().fit(n_t_x,t_y)#학습
#0. 트레인 데이터로 그려보기
plt.scatter(n_t_x, t_y)
plt.plot([128,186],[128*lr.coef_+lr.intercept_,186*lr.coef_+lr.intercept_])

#1. 사이킷런을 이용해 165cm 일때 예상몸무게 출력
print(lr.predict([[165]])) #:[67.58428165]
#다른풀이
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
X = np.array([174,152,138,128,186])
Y = np.array([71,55,46,38,88])
t_x = X.reshape(-1,1)

lr = LinearRegression().fit(t_x,Y)
print(lr.predict([[165]]))
