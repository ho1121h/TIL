## 의사결정트리
- 어떤 규칙을 하나의 트리(tree) 형태로 표현한 후 이를 바탕으로 분류나 회귀 문제를 해결
- 트리 구조의 마지막 노드에는 분류 문제에서 클래스, 회귀 문제에서는 예측치가 들어감
- 의사결정트리는 딥러닝 기반을 제외한 전통적인 통계 기반의 머신러닝 모델 중 효과와실용성이 가장 좋음
- 앙상블(ensemble) 모델이나 부스팅(boosting) 같은 새로운 기법들이 모델들의 성능을 대폭향상시키고 있음
- 분할 속성:부모노드에 들어가는 조건들
- 어떤 분할 속성이 가장 모호성을 줄일 것인지 파악
## 엔트로피
- 어떤 목적의 달성을 위한 경우의 수를 정량적으로 표현하는 수치
- 현재 정보 제공 상태를 결정 
- 낮은 엔트로피 = 경우의 수가 적다 =낮은 불확실성
- 높은 엔트로피 = 경우의 수가 높다 = 높은 불확실성

엔트로피를 측정하는 방법 ex)y 값이 True, False 가 있을때 의사결정, 전체 m
전체 엔트로피 h(D) = -len(True)/m*log2*len(True)/m + (-len(False)/m*log2*len(False)/m ) =엔트로피
```python
#전체 엔트로피값
import pandas as pd
import numpy as np
data = pd.read_csv('day6_data2.csv')
def get_info(df):
    buy = df.loc[df['class_buys_computer']=='yes']
    not_buy = df.loc[df['class_buys_computer']=='no']
    x=np.array([len(buy)/len(df),len(not_buy)/len(df)])
    y=np.log2(x[x!=0])
    info_all = -sum(x[x!=0]*y)
    return info_all
get_info(data)

```
## 정보 이득 
- 엔트로피를 사용하여 속성별 분류 시 데이터가 얼마나 순수한지(impurity)를 측정
전체 엔트로피 - 속성별 엔트로피 = 속성별 정보 이득

```py
#속성 엔트로피를 구하는 함수
def get_attribute_info(df,attribute_name):
    att_v = data[attribute_name].unique()
    get_infos = []
    for i in att_v:
        split_df = data.loc[data[attribute_name]==i]
        get_infos.append((len(split_df)/len(df)) * get_info(split_df))
    return sum(get_infos)
# 속성정보이득 (Gain)= 전체엔트로피 - 속성엔트로피
get_info(data) - get_attribute_info(data,'age')
get_info(data) - get_attribute_info(data,'income')
get_info(data) - get_attribute_info(data,'student')
get_info(data) - get_attribute_info(data,'credit_rating')
```
## 의사 결정 트리 구현

```py
#트리모듈 불러옴
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state= 42)
dt.fit(s_t_x,s_t_y)
dt.score(s_t_x,s_t_y), dt.score(v_t_x,v_t_y)

from sklearn.model_selection import cross_validate #교차 검증을 위한 전단계
sc = cross_validate(dt,t_x,t_y)
np.mean(sc['test_score'])
#>>0.855300214703487

from sklearn.model_selection import StratifiedKFold#교차검증
sc1= cross_validate(dt,t_x,t_y,cv=StratifiedKFold())
#>>0.855300214703487

#하이퍼 파라미터 를 이용한 검증
sc_ck= StratifiedKFold(n_splits=10,shuffle= True, random_state= 42)
sc1= cross_validate(dt,t_x,t_y,cv=sc_ck)
#>>0.8574181117533719
```
```py
sc_ck= StratifiedKFold(n_splits=10,shuffle= True, random_state= 42)
sc2= cross_validate(dt,t_x,t_y,cv=sc_ck)
np.mean(sc2['test_score'])
rom sklearn.model_selection import GridSearchCV#하이퍼 파라미터-교차검증-오ㅓ차
params = {'min_impurity_decrease':[0.0001,0.0002,0.0003,0.0004,0.0005]}
#검증을 위한 데이터
#다음 클래스로 실행 ,매개변수,-1은 모두 실행
gs = GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)
gs.fit(t_x,t_y)#검증할 데이터 학습

#최적으로 검증된 트리/성능 테스트 서로의 차이가 좁으므로 최적화 됨ㅇㅇ
dt = gs.best_estimator_
dt.score(t_x,t_y),dt.score(tt_x,tt_y)

gs.best_params_#이상적인 파라미터 :{'min_impurity_decrease': 0.0001}



```

---
- 모듈 사용 예
```py
from sklearn.tree import DecisionTreeClassifier#의사결정트리사용
from sklearn.model_selection import cross_val_score#교차검증
from sklearn.metrics import accuracy_score #정확도
t_l = []#빈리스트
tt_l = []#빈리스트
for i in range(3,20):
    dt = DecisionTreeClassifier(min_samples_leaf=i)
    acc = cross_val_score(dt,t_x,y,scoring='accuracy',cv=5).mean() #cv는 쪼갬
    t_l.append(accuracy_score(dt.fit(t_x,y).predict(t_x),y))#두개의 y값이 얼마나 정확한지 정확도 계산
    tt_l.append(acc)
r = pd.DataFrame(t_l,index=range(3,20),columns=['train'])
r['test'] = tt_l


```
## 앙상블

```py
from sklearn.ensemble import * #로불러옴
학습한변수.feature_importances_# 데이터 3개의 특징이 도출(각 중요도)
```