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
Y=t1.target#t1의 y값을 가져옴 영역을 시각화하기위함
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
df info()
'''
#   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  1309 non-null   int64  
 1   Pclass       1309 non-null   int64  
 2   Name         1309 non-null   object 
 3   Sex          1309 non-null   object 
 4   Age          1309 non-null   float64
 5   SibSp        1309 non-null   int64  
 6   Parch        1309 non-null   int64  
 7   Ticket       1309 non-null   object 
 8   Fare         1308 non-null   float64
 9   Cabin        295 non-null    object 
 10  Embarked     1309 non-null   object 
dtypes: float64(2), int64(4), object(5)

'''

for i in 범주형:
    df[i]=df[i].astype(object)#범주형 전부 오브젝트로 변환
for i in 숫자형:
    df[i]=df[i].astype(float)#숫자형 전부 실수형으로 변환
df['SibSp']=df['SibSp'].astype(int)#정수형으로변환
df['Parch']=df['Parch'].astype(int)#정수형으로 변환
'''
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  1309 non-null   object 
 1   Pclass       1309 non-null   object 
 2   Name         1309 non-null   object 
 3   Sex          1309 non-null   object 
 4   Age          1309 non-null   float64
 5   SibSp        1309 non-null   int32  
 6   Parch        1309 non-null   int32  
 7   Ticket       1309 non-null   object 
 8   Fare         1308 non-null   float64
 9   Cabin        295 non-null    object 
 10  Embarked     1309 non-null   object 

'''
def f(ldf, rdf, on, how='inner',index=None):
    if index is True:
        return pd.merge(ldf,rdf,how=how,left_index=True,right_index=True)
    else:
         return pd.merge(ldf,rdf,how=how,on=on)

one_hot_df=f(
df,pd.get_dummies(df['Sex'],prefix='Sex'),on=None,index=True)
one_hot_df=f(
one_hot_df,pd.get_dummies(df['Pclass'],prefix='Pclass'),on=None,index=True)
one_hot_df=f(
one_hot_df,pd.get_dummies(df['Embarked'],prefix='Embarked'),on=None,index=True)
one_hot_df
#원핫 더미데이터 추가 성별 ,클래스, Embarked가 분류됨
#참 1, 거짓 0
one_hot_df.columns.to_list()
'''
['PassengerId',
 'Pclass',
 'Name',
 'Sex',
 'Age',
 'SibSp',
 'Parch',
 'Ticket',
 'Fare',
 'Cabin',
 'Embarked',
 'Sex_female',
 'Sex_male',
 'Pclass_1',
 'Pclass_2',
 'Pclass_3',
 'Embarked_C',
 'Embarked_Q',
 'Embarked_S']

'''
Y_true #트레인 데이터

ck1=['Sex','Pclass','Embarked']
for i in ck1:
    ck_df=pd.merge(one_hot_df[i],Y_true,left_index=True,right_index=True)
    sns.countplot(x='Survived',hue=i,data=ck_df)
    plt.show()
#ck1(성별에 따른 생존수,클래스에 따른 생존수,Embarked에 따른 생존수)에 대한 시각화
ck1=['Sex','Pclass','Embarked']
ck_df2=pd.merge(one_hot_df[ck1],Y_true,left_index=True,right_index=True)
g=sns.catplot(x='Sex',hue='Pclass',col='Survived',kind='count' ,data=ck_df2)
#왼쪽표:p클래스에 따른 남녀 사망수 ,오른쪽:남녀 생존수
#남자가 사망 수가 더많고 클래스가 낮을 수록 사망수가 많음을 볼 수 있음

[name.split('_')[0] for name in one_hot_df.columns.to_list()]
'''
['PassengerId',
 'Pclass',
 'Name',
 'Sex',
 'Age',
 'SibSp',
 'Parch',
 'Ticket',
 'Fare',
 'Cabin',
 'Embarked',
 'Sex',
 'Sex',
 'Pclass',
 'Pclass',
 'Pclass',
 'Embarked',
 'Embarked',
 'Embarked']'''#  _가 사라짐
 [name for name in one_hot_df.columns.to_list() 
          if name.split('_')[0] in ck1 
          and '_' in name
         ]+['Sex']
'''
['Sex_female',
 'Sex_male',
 'Pclass_1',
 'Pclass_2',
 'Pclass_3',
 'Embarked_C',
 'Embarked_Q',
 'Embarked_S',
 'Sex']'''# _가 들어간것 dummies 컬럼이 따로 분류됨
```
## 딥러닝 실습 단계 

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#1.data의 수집
data=pd.read_csv('data.csv')
X = pd.DataFrame(data,columns=['D_length','D_weight'])# 보통 속성 열 행
Y = pd.DataFrame(data,columns=['y'])# 보통 값, 정답
np_X=np.array(X)#넘파이로 정리한 2차원 데이터
np_Y=np.array(Y['y'], dtype=int)#정수로 타입 변환
#2.data 전처리
mean=np.mean(np_X,axis=0)
std=np.std(np_X,axis=0)
sc_t_X=(np_X-mean)/std
t_x,tt_x,t_y,tt_y = train_test_split(sc_t_X,np_Y,random_state=10)
#3.모델 생성 및 학습
kn=KNeighborsClassifier().fit(t_x,t_y)
#4.테스트 및 검증
kn.score(tt_x,tt_y)
```