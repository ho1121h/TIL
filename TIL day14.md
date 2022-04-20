# 딥러닝 기초
## 사이킷 런
```python
from sklearn.datasets import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.svm import SVC #사이킷런의 svm 모듈에서 SCV 클래스 사용 (소프트 벡터머신
t1=load_iris()
t1
f=[2,3]#2행,3열
X=t1.data[:,f]#행 전부
Y=t1.target
m=SVC(kernel='linear',random_state=0)
m.fit(X,Y)#훈련 메서드

X_min=X[:,0].min() -1
X_max=X[:,0].max() +1
Y_min=X[:,1].min() -1
Y_max=X[:,1].max() +1
XX,YY=np.meshgrid(np.linspace(X_min,X_max,1000),np.linspace(Y_min,Y_max,1000))
ZZ=m.predict(np.c_[XX.ravel(),YY.ravel()]).reshape(XX.shape)
plt.contourf(XX,YY,ZZ)#영역 면으로 구분
plt.contour(XX,YY,ZZ)#영역  선으로 구분
plt.scatter(X[Y== 0,0],X[Y== 0,1] ,s=20,label=t1.target_names[0])
plt.scatter(X[Y== 1,0],X[Y== 1,1] ,s=20,label=t1.target_names[1])
plt.scatter(X[Y== 2,0],X[Y== 2,1] ,s=20,label=t1.target_names[2])
 
 
```
위그래프 다른방법으로 표현
```python
from sklearn.cluster import AffinityPropagation

X , _ = make_blobs(n_features=2, centers=3 ,random_state=1)

m=AffinityPropagation().fit(X)

plt.scatter(X[:,0],X[:,1])
```
다른방법 2
```python
plt.scatter(X[:,0],X[:,1])
for k in range(3):
    c=X[m.cluster_centers_indices_[k]]
    for i in X[m.labels_ == k]:
        plt.plot([c[0],i[0]],[c[1],i[1]])
        

```

## 데이터 전처리(정규화)
데이터를 학습하는데 비어있는 값은 데이터 분석에 영향을 주니 처리하는게 좋다.

```python
import numpy as np
import pandas as pd
data = {
    '이름':["길동",'둘리',np.nan,"또치",'희동',np.nan],
    '나이':[40,np.nan,15,np.nan,5,np.nan],
    '성별':['남자',np.nan,'여자','여자','남자',np.nan],
    '시험점수':[np.nan,20,80,10,2,np.nan]
}
df=pd.DataFrame(data,columns=['이름','나이','성별','시험점수'])

df.isnull().sum()/len(df)#결측치가 있는 비율
'''
이름      0.333333
나이      0.500000
성별      0.333333
시험점수    0.333333
dtype: float64
'''
df2=df.dropna()#결측치가나온 행을 삭제
#기본적으로 하나만 있어도 삭제 희동빼고 전부삭제
df3=df.dropna(how="all")#all은 조건이 전부 Nan 이면 삭제 (앤드연산
df.dropna(axis=1,how="all")#열의 조건이 전부 null 이면 삭제


df4.fillna(0)#결측치가 있는 곳에 대입
df4['이름'].fillna('희선',inplace=True) #이름 칸에 비어있는곳을 희선으로 대입 후 변경
df4['나이'].fillna(df4['나이'].mean(),inplace=True)#나이칸에 비어있는곳을 평균값으로 대입후 변경

df4.groupby('성별')['시험점수'].transform('mean')#성별끼리 점수 평균
df4["시험점수"].fillna(df4.groupby('성별')['시험점수'].transform('mean'),inplace=True)
df4#성별의 데이터 별로 평균값을 시험점수칸에 대입 후 변경


pd.get_dummies(data2)#원 핫 인 코 딩 => 0과 1로 나타냄



###스케일링
#변수 범위가 다를 때 스케일링을함
data = pd.DataFrame(df2,columns=['나이','시험점수'])
data
(data['시험점수']-data['시험점수'].mean()) /(data['시험점수'].std())#표준화
```

---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
f_l=[i for i in os.listdir() if '.csv' in i]
f_l.reverse()
f_l
add_l=[]
for i in f_l:
    add_l.append(pd.read_csv(i))
new_df=pd.concat(add_l)#파일을 합침

test_d=pd.read_csv('test.csv')#테스트 파일
t_d=pd.read_csv('train.csv')#훈련 파일

df= pd.concat([t_d,test_d])#df=융합파일
df=df.reset_index(drop=True)
df.T

n_of_t_d_dataset = df.Survived.notnull().sum() 
#891 결측치가 없는 행
n_of_test_d_dataset = df.Survived.isnull().sum()
#418 결측치가 있는 행

df.isnull().sum()/len(df) *100
'''결측치 비율
PassengerId    0.00
Pclass         0.00
Name           0.00
Sex            0.00
Age           20.09
SibSp          0.00
Parch          0.00
Ticket         0.00
Fare           0.08
Cabin         77.46
Embarked       0.15
dtype: float64

'''
#결측치를 채울 평균값구하기
df[df['Age'].notnull()].groupby(['Sex'])['Age'].mean()
'''
Sex
female   28.69
male     30.59
Name: Age, dtype: float64
'''
df[df['Age'].notnull()].groupby(['Pclass'])['Age'].mean()
'''
Pclass
1   39.16
2   29.51
3   24.82
Name: Age, dtype: float64

'''
df['Age'].fillna(df.groupby('Pclass')['Age'].transform('mean'),inplace=True)
'''
PassengerId    0.00
Pclass         0.00
Name           0.00
Sex            0.00
Age            0.00
SibSp          0.00
Parch          0.00
Ticket         0.00
Fare           0.08
Cabin         77.46
Embarked       0.15
'''
df.loc[61,'Embarked']='S'
df.loc[829,'Embarked']='S'
df.isnull().sum()/len(df) *100
'''
PassengerId    0.00
Pclass         0.00
Name           0.00
Sex            0.00
Age            0.00
SibSp          0.00
Parch          0.00
Ticket         0.00
Fare           0.08
Cabin         77.46
Embarked       0.00
'''
##데이터 타입
범주형=['PassengerId','Pclass','Name','Sex','Ticket','Cabin','Embarked'] #오븢젝트
숫자형=['Age','SibSp','Parch','Fare']#플롯

for i in 범주형:
    df[i]=df[i].astype(object)
for i in 숫자형:
    df[i]=df[i].astype(float)
df['SibSp']=df['SibSp'].astype(int)
df['Parch']=df['Parch'].astype(int)
```